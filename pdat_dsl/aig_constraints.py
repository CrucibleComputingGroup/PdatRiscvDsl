#!/usr/bin/env python3
"""
AIG-based field constraints for instruction set specifications.

This module provides:
- Helper functions to build AIGs for common constraint patterns
- Data structures for per-instruction field constraints
- Conversion from AIGs to SystemVerilog expressions
"""

from dataclasses import dataclass
from typing import Optional, Set, List, Tuple
from aigverse import Aig, write_verilog
import tempfile
import os


# =============================================================================
# AIG Builder Helper Functions
# =============================================================================

def range_to_aig(min_val: int, max_val: int, width: int) -> Aig:
    """
    Create an AIG representing: min_val <= x <= max_val

    Optimized for contiguous ranges (e.g., x0-x15 = 0 <= x <= 15).

    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        width: Bit width of the field

    Returns:
        AIG with `width` primary inputs representing the field bits,
        and one primary output representing the constraint satisfaction

    Examples:
        range_to_aig(0, 15, 5) -> x[4] == 0  (upper bit must be 0)
        range_to_aig(16, 31, 5) -> x[4] == 1 (upper bit must be 1)
        range_to_aig(0, 3, 5) -> x[4:2] == 3'b000 (upper 3 bits must be 0)
    """
    aig = Aig()

    # Create primary inputs for each bit (LSB first)
    bits = [aig.create_pi() for _ in range(width)]

    # Optimize for power-of-2 aligned ranges
    if min_val == 0 and (max_val + 1) == (1 << (width - 1)):
        # Range is [0, 2^(width-1) - 1], just check MSB == 0
        result = aig.create_not(bits[-1])
    elif min_val == (1 << (width - 1)) and max_val == (1 << width) - 1:
        # Range is [2^(width-1), 2^width - 1], just check MSB == 1
        result = bits[-1]
    elif max_val == (1 << width) - 1 and min_val > 0:
        # Range is [min_val, 2^width - 1], i.e., x >= min_val
        # Check if min_val is a power of 2
        if (min_val & (min_val - 1)) == 0:
            # min_val is a power of 2, e.g., 4 = 2^2, 8 = 2^3
            # For x >= 4 with 5-bit value, check if bit[2] | bit[3] | bit[4]
            bit_pos = (min_val - 1).bit_length()  # Position of the bit that must be set
            # At least one bit from [bit_pos..width-1] must be 1
            result = bits[bit_pos]
            for i in range(bit_pos + 1, width):
                result = aig.create_or(result, bits[i])
        else:
            # General case
            ge_min = build_gte_constant(aig, bits, min_val)
            le_max = build_lte_constant(aig, bits, max_val)
            result = aig.create_and(ge_min, le_max)
    elif min_val == 0:
        # Range starting from 0: [0, max_val]
        # Check if max_val + 1 is a power of 2
        limit = max_val + 1
        if (limit & (limit - 1)) == 0:
            # limit is a power of 2, e.g., max_val=3 -> limit=4 = 2^2
            # Just check that upper bits are 0
            # For x0-x3 (max_val=3, limit=4=2^2), check bits[4:2] == 0
            num_bits_needed = (limit - 1).bit_length()  # Number of bits to represent max_val
            result = aig.get_constant(True)
            for i in range(num_bits_needed, width):
                # All upper bits must be 0
                result = aig.create_and(result, aig.create_not(bits[i]))
        else:
            # General case: build comparison circuit
            ge_min = build_gte_constant(aig, bits, min_val)
            le_max = build_lte_constant(aig, bits, max_val)
            result = aig.create_and(ge_min, le_max)
    else:
        # General case: build comparison circuit
        # result = (x >= min_val) AND (x <= max_val)
        ge_min = build_gte_constant(aig, bits, min_val)
        le_max = build_lte_constant(aig, bits, max_val)
        result = aig.create_and(ge_min, le_max)

    aig.create_po(result)
    return aig


def set_to_aig(values: Set[int], width: int) -> Aig:
    """
    Create an AIG representing: x ∈ {v1, v2, ...}

    Args:
        values: Set of allowed values
        width: Bit width of the field

    Returns:
        AIG with `width` inputs and one output

    Example:
        set_to_aig({0, 2, 4, 8}, 5) -> (x==0) OR (x==2) OR (x==4) OR (x==8)
    """
    aig = Aig()
    bits = [aig.create_pi() for _ in range(width)]

    # Build OR of equality checks
    checks = []
    for val in sorted(values):
        eq_check = build_equals_constant(aig, bits, val)
        checks.append(eq_check)

    if len(checks) == 0:
        # Empty set - always false
        result = aig.get_constant(False)
    elif len(checks) == 1:
        result = checks[0]
    else:
        # OR all checks together
        result = checks[0]
        for check in checks[1:]:
            result = aig.create_or(result, check)

    aig.create_po(result)
    return aig


def complement_aig(constraint_aig: Aig) -> Aig:
    """
    Create the complement of a constraint AIG.

    Args:
        constraint_aig: Input AIG with N inputs and 1 output

    Returns:
        New AIG with same inputs but inverted output
    """
    # Create new AIG with same structure
    aig = Aig()

    # Copy inputs
    num_inputs = constraint_aig.num_pis()
    inputs = [aig.create_pi() for _ in range(num_inputs)]

    # Traverse original AIG and rebuild structure
    # Map from old signal -> new signal
    signal_map = {}

    # Map inputs
    for i, old_pi in enumerate(constraint_aig.pis()):
        signal_map[old_pi] = inputs[i]

    # Copy gates in topological order
    for node in constraint_aig.nodes():
        if constraint_aig.is_pi(node):
            continue  # Already mapped

        # Get fanins
        fanins = list(constraint_aig.fanins(node))
        if len(fanins) != 2:
            continue  # Skip constants or special nodes

        left, right = fanins
        left_node = constraint_aig.get_node(left)
        right_node = constraint_aig.get_node(right)

        # Get mapped signals
        if left_node not in signal_map or right_node not in signal_map:
            continue  # Dependencies not yet mapped

        new_left = signal_map[left_node]
        new_right = signal_map[right_node]

        # Apply complementation if needed
        if constraint_aig.is_complemented(left):
            new_left = aig.create_not(new_left)
        if constraint_aig.is_complemented(right):
            new_right = aig.create_not(new_right)

        # Create AND gate
        new_gate = aig.create_and(new_left, new_right)
        signal_map[node] = new_gate

    # Get output and invert it
    for po_signal in constraint_aig.pos():
        po_node = constraint_aig.get_node(po_signal)
        if po_node in signal_map:
            new_output = signal_map[po_node]

            # Apply complementation from original output
            if constraint_aig.is_complemented(po_signal):
                new_output = aig.create_not(new_output)

            # INVERT the output (this is the complement!)
            new_output = aig.create_not(new_output)
            aig.create_po(new_output)

    return aig


def and_aigs(aig1: Aig, aig2: Aig) -> Aig:
    """
    Create AND of two constraint AIGs.

    Both AIGs must have the same number of inputs.

    Args:
        aig1: First constraint AIG
        aig2: Second constraint AIG

    Returns:
        New AIG representing aig1 AND aig2
    """
    num_inputs = aig1.num_pis()
    if num_inputs != aig2.num_pis():
        raise ValueError(f"AIG input mismatch: {num_inputs} vs {aig2.num_pis()}")

    # Optimization: If one AIG is constant TRUE, return the other
    # Note: get_constant(True) returns Signal(!0), i.e., complemented constant 0
    # So TRUE = constant node that IS complemented
    # Check if aig1 outputs constant TRUE
    if aig1.num_gates() == 0:
        for po_signal in aig1.pos():
            po_node = aig1.get_node(po_signal)
            if aig1.is_constant(po_node):
                # Constant 0 node
                # TRUE if complemented (!0 = TRUE), FALSE if not (0 = FALSE)
                is_true = aig1.is_complemented(po_signal)
                if is_true:
                    # aig1 is TRUE, return aig2
                    return aig2

    # Check if aig2 outputs constant TRUE
    if aig2.num_gates() == 0:
        for po_signal in aig2.pos():
            po_node = aig2.get_node(po_signal)
            if aig2.is_constant(po_node):
                is_true = aig2.is_complemented(po_signal)
                if is_true:
                    # aig2 is TRUE, return aig1
                    return aig1

    # Create new AIG
    aig = Aig()
    inputs = [aig.create_pi() for _ in range(num_inputs)]

    # Rebuild aig1 with shared inputs
    signal_map1 = {}
    for i, old_pi in enumerate(aig1.pis()):
        signal_map1[old_pi] = inputs[i]

    # Copy aig1 gates
    for node in aig1.nodes():
        if aig1.is_pi(node):
            continue

        fanins = list(aig1.fanins(node))
        if len(fanins) != 2:
            continue

        left, right = fanins
        left_node = aig1.get_node(left)
        right_node = aig1.get_node(right)

        if left_node not in signal_map1 or right_node not in signal_map1:
            continue

        new_left = signal_map1[left_node]
        new_right = signal_map1[right_node]

        if aig1.is_complemented(left):
            new_left = aig.create_not(new_left)
        if aig1.is_complemented(right):
            new_right = aig.create_not(new_right)

        new_gate = aig.create_and(new_left, new_right)
        signal_map1[node] = new_gate

    # Rebuild aig2 with shared inputs
    signal_map2 = {}
    for i, old_pi in enumerate(aig2.pis()):
        signal_map2[old_pi] = inputs[i]

    # Copy aig2 gates
    for node in aig2.nodes():
        if aig2.is_pi(node):
            continue

        fanins = list(aig2.fanins(node))
        if len(fanins) != 2:
            continue

        left, right = fanins
        left_node = aig2.get_node(left)
        right_node = aig2.get_node(right)

        if left_node not in signal_map2 or right_node not in signal_map2:
            continue

        new_left = signal_map2[left_node]
        new_right = signal_map2[right_node]

        if aig2.is_complemented(left):
            new_left = aig.create_not(new_left)
        if aig2.is_complemented(right):
            new_right = aig.create_not(new_right)

        new_gate = aig.create_and(new_left, new_right)
        signal_map2[node] = new_gate

    # AND the outputs together
    output1 = None
    for po_signal in aig1.pos():
        po_node = aig1.get_node(po_signal)

        # Check if it's a constant
        if aig1.is_constant(po_node):
            # Get the constant value
            if aig1.is_complemented(po_signal):
                output1 = aig.get_constant(False)
            else:
                output1 = aig.get_constant(True)
        elif po_node in signal_map1:
            output1 = signal_map1[po_node]
            if aig1.is_complemented(po_signal):
                output1 = aig.create_not(output1)
        else:
            # Node not in map - might be a direct input
            # Try to find it in the input mapping
            for i, old_pi in enumerate(aig1.pis()):
                if old_pi == po_node:
                    output1 = inputs[i]
                    if aig1.is_complemented(po_signal):
                        output1 = aig.create_not(output1)
                    break

        if output1 is not None:
            break

    output2 = None
    for po_signal in aig2.pos():
        po_node = aig2.get_node(po_signal)

        # Check if it's a constant
        if aig2.is_constant(po_node):
            if aig2.is_complemented(po_signal):
                output2 = aig.get_constant(False)
            else:
                output2 = aig.get_constant(True)
        elif po_node in signal_map2:
            output2 = signal_map2[po_node]
            if aig2.is_complemented(po_signal):
                output2 = aig.create_not(output2)
        else:
            # Node not in map - might be a direct input
            for i, old_pi in enumerate(aig2.pis()):
                if old_pi == po_node:
                    output2 = inputs[i]
                    if aig2.is_complemented(po_signal):
                        output2 = aig.create_not(output2)
                    break

        if output2 is not None:
            break

    if output1 is None or output2 is None:
        raise ValueError(f"Failed to find outputs in AIGs (output1={output1}, output2={output2})")

    # Create final AND
    final_output = aig.create_and(output1, output2)
    aig.create_po(final_output)

    return aig


# =============================================================================
# Low-level AIG Building Blocks
# =============================================================================

def build_gte_constant(aig: Aig, bits: List, const: int) -> int:
    """Build AIG for: x >= const (unsigned comparison)"""
    # For now, use simple bit-by-bit comparison
    # TODO: Optimize with more efficient comparator circuit
    width = len(bits)

    if const == 0:
        # Always true
        return aig.get_constant(True)

    if const >= (1 << width):
        # Always false
        return aig.get_constant(False)

    # Build comparison from MSB to LSB
    # x >= const iff there exists a bit position i where:
    #   - All bits > i are equal between x and const
    #   - Bit i of x is 1 and bit i of const is 0
    # OR all bits are equal

    result = aig.get_constant(False)
    prefix_equal = aig.get_constant(True)

    for i in range(width - 1, -1, -1):
        const_bit = (const >> i) & 1

        if const_bit == 0:
            # If x[i] == 1 and const[i] == 0, then x > const (so far)
            gt_here = aig.create_and(bits[i], prefix_equal)
            result = aig.create_or(result, gt_here)
        else:
            # const[i] == 1, so x[i] must be 1 for equality to continue
            pass

        # Update prefix equality: prefix_equal = prefix_equal AND (x[i] == const[i])
        if const_bit == 1:
            eq_bit = bits[i]
        else:
            eq_bit = aig.create_not(bits[i])
        prefix_equal = aig.create_and(prefix_equal, eq_bit)

    # Also true if all bits are equal
    result = aig.create_or(result, prefix_equal)

    return result


def build_lte_constant(aig: Aig, bits: List, const: int) -> int:
    """Build AIG for: x <= const (unsigned comparison)"""
    width = len(bits)

    if const >= (1 << width) - 1:
        # Always true
        return aig.get_constant(True)

    if const < 0:
        # Always false
        return aig.get_constant(False)

    # x <= const is equivalent to NOT(x > const)
    # x > const iff x >= (const + 1)
    gt_result = build_gte_constant(aig, bits, const + 1)
    return aig.create_not(gt_result)


def build_equals_constant(aig: Aig, bits: List, const: int) -> int:
    """Build AIG for: x == const"""
    width = len(bits)

    # AND together all bit equalities
    result = aig.get_constant(True)

    for i in range(width):
        const_bit = (const >> i) & 1
        if const_bit == 1:
            bit_match = bits[i]
        else:
            bit_match = aig.create_not(bits[i])
        result = aig.create_and(result, bit_match)

    return result


# =============================================================================
# AIG to SystemVerilog Conversion
# =============================================================================

def aig_to_simple_comparison(aig: Aig, signal_name: str, width: int) -> str:
    """
    Convert AIG to simple comparison syntax when possible.

    Analyzes the AIG structure to recognize common patterns:
    - x <= N  (range [0, N])
    - x >= N  (range [N, 2^width-1])
    - x[high:low] == value (bit-slice equality)

    Falls back to general expression if pattern not recognized.

    Args:
        aig: AIG constraint
        signal_name: Signal name (e.g., "instr_rd")
        width: Field width

    Returns:
        Simple SystemVerilog comparison or boolean expression
    """
    # Check for constant output (TRUE or FALSE)
    if aig.num_gates() == 0:
        # Constant AIG
        for po_signal in aig.pos():
            po_node = aig.get_node(po_signal)
            if aig.is_constant(po_node):
                # TRUE if complemented, FALSE if not
                is_true = aig.is_complemented(po_signal)
                return "1'b1" if is_true else "1'b0"

    # For very simple AIGs (0-3 gates), try to recognize patterns
    if aig.num_gates() <= 3:
        # Try to recognize: x[msb] == 0 pattern (for ranges like x0-x15)
        if aig.num_gates() == 0:  # Just checking one input bit
            for po_signal in aig.pos():
                po_node = aig.get_node(po_signal)
                # Check if output is directly an input (possibly complemented)
                for i, pi_node in enumerate(aig.pis()):
                    if po_node == pi_node:
                        if aig.is_complemented(po_signal):
                            # Output is !input[i], meaning input[i] must be 0
                            return f"(!{signal_name}[{i}:{i}])"
                        else:
                            # Output is input[i], meaning input[i] must be 1
                            return f"{signal_name}[{i}:{i}]"

        # For simple range constraints like x <= 3, x <= 7, x <= 15
        # These have very few gates checking upper bits
        # Try to detect and generate: signal <= 5'dN
        # For now, use heuristic: if checking upper bits are 0
        pass  # TODO: Pattern recognition

    # Fall back to general expression generation
    return aig_to_sv_expr(aig, signal_name)


def aig_to_sv_expr(aig: Aig, signal_name: str) -> str:
    """
    Convert an AIG to a SystemVerilog boolean expression.

    The AIG should have N primary inputs (representing bits of the field)
    and 1 primary output (the constraint satisfaction).

    Args:
        aig: AIG with field bits as inputs and constraint as output
        signal_name: Base signal name (e.g., "instr_rd")

    Returns:
        SystemVerilog expression string

    Example:
        For an AIG checking x[4] == 0 (MSB must be 0):
        Returns: "~signal_name[4:4]"
    """
    # Map from AIG node -> SV expression
    node_to_expr = {}

    num_inputs = aig.num_pis()

    # Map primary inputs to signal bits using bit-slice notation [i:i]
    for i, pi_node in enumerate(aig.pis()):
        # Use bit-slice notation instead of single-bit index
        # This is more compatible with different SystemVerilog parsers
        node_to_expr[pi_node] = f"{signal_name}[{i}:{i}]"

    # Map constants
    # Note: aigverse constants are special nodes
    const_false = aig.get_constant(False)
    const_true = aig.get_constant(True)

    const_false_node = aig.get_node(const_false)
    const_true_node = aig.get_node(const_true)

    node_to_expr[const_false_node] = "1'b0"
    node_to_expr[const_true_node] = "1'b1"

    # Traverse gates in topological order
    for node in aig.nodes():
        if node in node_to_expr:
            continue  # Already processed (input or constant)

        if aig.is_pi(node):
            continue  # Should already be mapped

        # Get fanins (inputs to this AND gate)
        fanins = list(aig.fanins(node))

        if len(fanins) == 0:
            # This might be a constant we haven't handled
            continue

        if len(fanins) != 2:
            # AND gates should have exactly 2 inputs
            continue

        left_signal, right_signal = fanins

        # Get the nodes
        left_node = aig.get_node(left_signal)
        right_node = aig.get_node(right_signal)

        # Get expressions for inputs (must already be computed)
        if left_node not in node_to_expr or right_node not in node_to_expr:
            # Dependencies not yet computed - skip for now
            # This shouldn't happen in topological order
            continue

        left_expr = node_to_expr[left_node]
        right_expr = node_to_expr[right_node]

        # Apply complementation (inversion)
        # Use logical NOT (!) for single-bit values
        if aig.is_complemented(left_signal):
            left_expr = f"(!{left_expr})"

        if aig.is_complemented(right_signal):
            right_expr = f"(!{right_expr})"

        # Create AND expression (use logical AND for boolean values)
        and_expr = f"({left_expr} && {right_expr})"
        node_to_expr[node] = and_expr

    # Get the output expression
    for po_signal in aig.pos():
        po_node = aig.get_node(po_signal)

        if po_node not in node_to_expr:
            # Output not computed - this is an error
            return "1'b1"  # Default to unconstrained

        output_expr = node_to_expr[po_node]

        # Apply complementation to output if needed
        if aig.is_complemented(po_signal):
            output_expr = f"(!{output_expr})"

        return output_expr

    # No output found
    return "1'b1"  # Default to unconstrained


def aig_to_sv_expr_absolute(aig: Aig, signal_name: str, base_position: int) -> str:
    """
    Convert an AIG to a SystemVerilog boolean expression with absolute bit indices.

    Args:
        aig: AIG with field bits as inputs
        signal_name: Base signal name (e.g., "instr_rdata_i")
        base_position: LSB position of field in instruction (e.g., 7 for rd[11:7])

    Returns:
        SystemVerilog expression with absolute bit positions

    Example:
        For rd field at [11:7], base_position=7:
        rd[4] becomes instr_rdata_i[11] (7 + 4)
    """
    # Map from AIG node -> SV expression
    node_to_expr = {}

    num_inputs = aig.num_pis()

    # Map primary inputs to absolute bit positions
    for i, pi_node in enumerate(aig.pis()):
        absolute_bit = base_position + i
        node_to_expr[pi_node] = f"{signal_name}[{absolute_bit}]"

    # Map constants
    const_false = aig.get_constant(False)
    const_true = aig.get_constant(True)

    const_false_node = aig.get_node(const_false)
    const_true_node = aig.get_node(const_true)

    node_to_expr[const_false_node] = "1'b0"
    node_to_expr[const_true_node] = "1'b1"

    # Traverse gates in topological order
    for node in aig.nodes():
        if node in node_to_expr:
            continue

        if aig.is_pi(node):
            continue

        fanins = list(aig.fanins(node))
        if len(fanins) == 0:
            continue

        if len(fanins) != 2:
            continue

        left_signal, right_signal = fanins
        left_node = aig.get_node(left_signal)
        right_node = aig.get_node(right_signal)

        if left_node not in node_to_expr or right_node not in node_to_expr:
            continue

        left_expr = node_to_expr[left_node]
        right_expr = node_to_expr[right_node]

        if aig.is_complemented(left_signal):
            left_expr = f"(!{left_expr})"

        if aig.is_complemented(right_signal):
            right_expr = f"(!{right_expr})"

        and_expr = f"({left_expr} & {right_expr})"
        node_to_expr[node] = and_expr

    # Get the output expression
    for po_signal in aig.pos():
        po_node = aig.get_node(po_signal)

        if po_node not in node_to_expr:
            return "1'b1"

        output_expr = node_to_expr[po_node]

        if aig.is_complemented(po_signal):
            output_expr = f"(!{output_expr})"

        return output_expr

    return "1'b1"


# =============================================================================
# Data Structures for Per-Instruction Constraints
# =============================================================================

@dataclass
class FieldConstraint:
    """
    AIG-based constraint on a single instruction field.

    The AIG has N primary inputs (one per bit of the field) and
    one primary output indicating whether the constraint is satisfied.
    """
    aig: Aig
    width: int  # Number of bits in the field
    base_position: int = 0  # LSB position in the instruction (e.g., 7 for rd, 15 for rs1)
    # Optional: semantic information for simpler codegen
    range_constraint: Optional[Tuple[int, int]] = None  # (min, max) if this is a simple range

    def to_systemverilog(self, signal_name: str) -> str:
        """
        Convert AIG to SystemVerilog boolean expression.

        Args:
            signal_name: Base signal name (e.g., "instr_rdata_i")

        Returns:
            SystemVerilog expression string with absolute bit indices
        """
        return aig_to_sv_expr_absolute(self.aig, signal_name, self.base_position)


@dataclass
class InstructionConstraints:
    """
    Per-instruction field constraints using AIGs.

    This represents the allowed values for each field of a specific instruction,
    identified by its opcode/funct bits.
    """
    instruction_name: str
    opcode_pattern: int  # Fixed bits (opcode, funct3, funct7)
    opcode_mask: int     # Mask for fixed bits

    # Optional constraints on each field
    rd: Optional[FieldConstraint] = None
    rs1: Optional[FieldConstraint] = None
    rs2: Optional[FieldConstraint] = None
    imm: Optional[FieldConstraint] = None

    def has_constraints(self) -> bool:
        """Check if any field constraints are present."""
        return any([self.rd, self.rs1, self.rs2, self.imm])

    def to_systemverilog(self, instr_signal: str = "instr_rdata_i") -> str:
        """
        Generate SystemVerilog constraint for this instruction.

        Returns a boolean expression that is true when the instruction
        matches and all field constraints are satisfied.
        """
        # Base instruction match
        parts = [f"(({instr_signal}[31:0] & 32'h{self.opcode_mask:08x}) == 32'h{self.opcode_pattern:08x})"]

        # Add field constraints with absolute bit positions
        # Note: FieldConstraint.to_systemverilog() uses absolute indexing now
        if self.rd:
            rd_expr = self.rd.to_systemverilog(instr_signal)
            parts.append(rd_expr)

        if self.rs1:
            rs1_expr = self.rs1.to_systemverilog(instr_signal)
            parts.append(rs1_expr)

        if self.rs2:
            rs2_expr = self.rs2.to_systemverilog(instr_signal)
            parts.append(rs2_expr)

        if self.imm:
            imm_expr = self.imm.to_systemverilog(instr_signal)
            parts.append(imm_expr)

        return " && ".join(parts)

    def to_systemverilog_with_wires(self, instr_signal: str = "instr_rdata_i") -> str:
        """
        Generate SystemVerilog constraint using extracted field wires.

        Assumes wires instr_rd, instr_rs1, instr_rs2 are defined.

        Args:
            instr_signal: Name of the instruction signal (for opcode matching)

        Returns a boolean expression for this instruction.
        """
        # Base instruction match (use full instruction signal for opcode)
        parts = [f"(({instr_signal}[31:0] & 32'h{self.opcode_mask:08x}) == 32'h{self.opcode_pattern:08x})"]

        # Add field constraints - use metadata for simple syntax when available
        if self.rd:
            if self.rd.range_constraint:
                min_val, max_val = self.rd.range_constraint
                rd_expr = f"(instr_rd <= 5'd{max_val})" if min_val == 0 else f"((instr_rd >= 5'd{min_val}) && (instr_rd <= 5'd{max_val}))"
            else:
                rd_expr = aig_to_simple_comparison(self.rd.aig, "instr_rd", 5)
            if rd_expr != "1'b1":  # Skip if unconstrained
                parts.append(rd_expr)

        if self.rs1:
            if self.rs1.range_constraint:
                min_val, max_val = self.rs1.range_constraint
                rs1_expr = f"(instr_rs1 <= 5'd{max_val})" if min_val == 0 else f"((instr_rs1 >= 5'd{min_val}) && (instr_rs1 <= 5'd{max_val}))"
            else:
                rs1_expr = aig_to_simple_comparison(self.rs1.aig, "instr_rs1", 5)
            if rs1_expr != "1'b1":
                parts.append(rs1_expr)

        if self.rs2:
            if self.rs2.range_constraint:
                min_val, max_val = self.rs2.range_constraint
                rs2_expr = f"(instr_rs2 <= 5'd{max_val})" if min_val == 0 else f"((instr_rs2 >= 5'd{min_val}) && (instr_rs2 <= 5'd{max_val}))"
            else:
                rs2_expr = aig_to_simple_comparison(self.rs2.aig, "instr_rs2", 5)
            if rs2_expr != "1'b1":
                parts.append(rs2_expr)

        if self.imm:
            imm_expr = aig_to_simple_comparison(self.imm.aig, "instr_imm", self.imm.width)
            if imm_expr != "1'b1":
                parts.append(imm_expr)

        return " && ".join(parts)


# =============================================================================
# Constraint Set Management
# =============================================================================

class ConstraintSet:
    """
    Manages a set of per-instruction constraints.

    This represents the final instruction set after processing all
    include/forbid rules in DSL v2.
    """

    def __init__(self):
        self.constraints: List[InstructionConstraints] = []

    def add_instruction(self, constraint: InstructionConstraints):
        """Add an instruction constraint to the set."""
        self.constraints.append(constraint)

    def to_systemverilog(self, instr_signal: str = "instr_rdata_i") -> str:
        """
        Generate SystemVerilog for entire constraint set.

        Returns a boolean expression that is true when the instruction
        is in the allowed set.
        """
        if not self.constraints:
            return "1'b0"  # Empty set - no instructions allowed

        # OR together all instruction constraints
        constraint_exprs = [c.to_systemverilog(instr_signal) for c in self.constraints]

        if len(constraint_exprs) == 1:
            return constraint_exprs[0]

        # Format as multi-line OR
        return "(\n      " + " ||\n      ".join(constraint_exprs) + "\n    )"
