#!/usr/bin/env python3
"""
BDD-based operations for PDAT DSL v2.

This module provides utilities to represent instruction constraints as
Binary Decision Diagrams (BDDs) and perform set operations on them.

The BDD represents the characteristic function of a regular language over
32-bit (or 16-bit) instruction encodings.
"""

from dd.autoref import BDD as BDDManager
from typing import Tuple, List, Set, Optional


# Global BDD manager (initialized when needed)
_bdd_manager_32 = None
_bdd_manager_16 = None


def get_bdd_manager(width: int = 32) -> BDDManager:
    """Get or create BDD manager for given bit width."""
    global _bdd_manager_32, _bdd_manager_16

    if width == 32:
        if _bdd_manager_32 is None:
            _bdd_manager_32 = BDDManager()
            # Declare 32 boolean variables for instruction bits
            _bdd_manager_32.declare('b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7',
                                   'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15',
                                   'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23',
                                   'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31')
        return _bdd_manager_32
    elif width == 16:
        if _bdd_manager_16 is None:
            _bdd_manager_16 = BDDManager()
            # Declare 16 boolean variables for compressed instructions
            _bdd_manager_16.declare('b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7',
                                   'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15')
        return _bdd_manager_16
    else:
        raise ValueError(f"Unsupported instruction width: {width}")


def pattern_to_bdd(pattern: int, mask: int, width: int = 32):
    """
    Convert a pattern/mask pair to a BDD.

    Args:
        pattern: Instruction bit pattern
        mask: Mask indicating which bits must match (1 = must match, 0 = don't care)
        width: Instruction width in bits (32 or 16)

    Returns:
        BDD representing all instructions matching the pattern/mask

    Example:
        pattern=0x00000033, mask=0xFE00707F
        Returns BDD where:
        - bits [6:0] = 0x33 (opcode)
        - bits [14:12] = 0x0 (funct3)
        - bits [31:25] = 0x00 (funct7)
        - bits [11:7], [19:15], [24:20] = don't care (rd, rs1, rs2)
    """
    bdd = get_bdd_manager(width)
    result = bdd.true  # Start with TRUE (all instructions)

    # Constrain each bit based on pattern and mask
    for i in range(width):
        bit_name = f'b{i}'
        if mask & (1 << i):  # This bit is constrained
            if pattern & (1 << i):  # Bit must be 1
                result &= bdd.var(bit_name)
            else:  # Bit must be 0
                result &= ~bdd.var(bit_name)

    return result


def field_range_to_bdd(bit_pos: int, width: int, min_val: int, max_val: int, instr_width: int = 32):
    """
    Generate BDD for field ∈ [min_val, max_val].

    Args:
        bit_pos: LSB position of field in instruction
        width: Width of field in bits
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        instr_width: Instruction width (32 or 16)

    Returns:
        BDD representing all instructions where field is in range

    Example:
        field_range_to_bdd(7, 5, 0, 15, 32)  # rd ∈ [0, 15]
        Returns BDD where bits [11:7] represent values 0-15
    """
    bdd = get_bdd_manager(instr_width)
    result = bdd.false  # Start with FALSE (empty set)

    # OR together all values in the range
    for value in range(min_val, max_val + 1):
        # Create BDD for field == value
        field_eq = bdd.true
        for i in range(width):
            bit_name = f'b{bit_pos + i}'
            if value & (1 << i):  # Bit i of value is 1
                field_eq &= bdd.var(bit_name)
            else:  # Bit i of value is 0
                field_eq &= ~bdd.var(bit_name)
        result |= field_eq

    return result


def field_equals_to_bdd(bit_pos: int, width: int, value: int, instr_width: int = 32):
    """
    Generate BDD for field == value.

    Args:
        bit_pos: LSB position of field
        width: Width of field in bits
        value: Exact value
        instr_width: Instruction width (32 or 16)

    Returns:
        BDD representing all instructions where field equals value
    """
    bdd = get_bdd_manager(instr_width)
    result = bdd.true

    for i in range(width):
        bit_name = f'b{bit_pos + i}'
        if value & (1 << i):
            result &= bdd.var(bit_name)
        else:
            result &= ~bdd.var(bit_name)

    return result


def bdd_to_patterns(bdd_expr, width: int = 32, max_patterns: int = 10000) -> List[Tuple[int, int, str]]:
    """
    Convert BDD to list of (pattern, mask, description) tuples for SystemVerilog generation.

    Args:
        bdd_expr: BDD expression
        width: Instruction width (32 or 16)
        max_patterns: Maximum number of patterns to extract

    Returns:
        List of (pattern, mask, description) tuples

    The BDD is converted to disjunctive normal form (DNF) - an OR of AND terms.
    Each AND term becomes a pattern/mask pair.
    """
    bdd = get_bdd_manager(width)

    if bdd_expr == bdd.false:
        return []

    if bdd_expr == bdd.true:
        # Matches all instructions - return pattern with all don't-care mask
        return [(0, 0, "all")]

    # Get satisfying assignments (care set)
    # This returns a generator of models (bit assignments)
    patterns = []

    try:
        # Use BDD.pick_iter to get cube representation
        # Each cube is a partial assignment (only constrained bits)
        for i, cube in enumerate(bdd.pick_iter(bdd_expr)):
            if i >= max_patterns:
                print(f"Warning: BDD has more than {max_patterns} patterns, truncating...")
                break

            # Convert cube (dict of bit assignments) to pattern/mask
            pattern = 0
            mask = 0

            for bit_name, bit_value in cube.items():
                # bit_name is like 'b5', extract bit position
                bit_pos = int(bit_name[1:])

                mask |= (1 << bit_pos)  # This bit is constrained
                if bit_value:
                    pattern |= (1 << bit_pos)  # Bit is 1

            patterns.append((pattern, mask, "bdd_pattern"))

    except Exception as e:
        print(f"Warning: Error extracting patterns from BDD: {e}")
        print(f"BDD size: {bdd.count(bdd_expr)} satisfying assignments")
        # Return empty list on error
        return []

    return patterns


def bdd_size(bdd_expr, width: int = 32) -> int:
    """
    Count the number of satisfying assignments in a BDD.

    Args:
        bdd_expr: BDD expression
        width: Instruction width

    Returns:
        Number of distinct instructions in the language
    """
    bdd = get_bdd_manager(width)
    return bdd.count(bdd_expr, nvars=width)


def bdd_to_systemverilog_expr(bdd_expr, signal_name: str = "instr_rdata_i", width: int = 32) -> str:
    """
    Convert BDD to compact SystemVerilog boolean expression.

    Args:
        bdd_expr: BDD expression
        signal_name: Name of instruction signal in SystemVerilog
        width: Instruction width (32 or 16)

    Returns:
        SystemVerilog boolean expression as string

    Example:
        Input: BDD for "ADD with rd < 16"
        Output: "(b11 == 0) && ((instr & 32'hFE00707F) == 32'h00000033)"
    """
    bdd_mgr = get_bdd_manager(width)

    # Get expression from BDD
    expr_str = bdd_mgr.to_expr(bdd_expr)

    # Convert ite(cond, then, else) to SystemVerilog ternary (cond ? then : else)
    # Convert b0, b1, etc. to instr[0], instr[1], etc.
    # Convert TRUE/FALSE to 1'b1/1'b0

    # Replace variable names
    for i in range(width):
        expr_str = expr_str.replace(f'b{i}', f'{signal_name}[{i}]')

    # Replace ite with ternary operator
    expr_str = expr_str.replace('ite(', '(')
    expr_str = expr_str.replace(', ', ' ? ')
    # This is approximate - need proper parsing

    # Replace TRUE/FALSE
    expr_str = expr_str.replace('TRUE', "1'b1")
    expr_str = expr_str.replace('FALSE', "1'b0")

    # Replace operators
    expr_str = expr_str.replace('~', '!')

    return expr_str


def _convert_bdd_expr_to_sv(expr_str: str, signal_name: str, width: int) -> str:
    """
    Convert BDD expression string to SystemVerilog syntax.

    Handles nested ite() expressions properly.
    """
    # Replace variable names - use word boundaries to avoid replacing digits inside other numbers
    # Replace in reverse order to avoid conflicts (b31 before b3, etc.)
    for i in range(width-1, -1, -1):
        # Use regex-like replacement: match 'b' followed by exact number, not part of larger number
        import re
        expr_str = re.sub(rf'\bb{i}\b', f'{signal_name}[{i}]', expr_str)

    # Replace TRUE/FALSE
    expr_str = expr_str.replace('TRUE', "1'b1")
    expr_str = expr_str.replace('FALSE', "1'b0")

    # Convert ite(cond, then, else) to (cond ? then : else)
    # Use recursive replacement
    while 'ite(' in expr_str:
        # Find innermost ite
        start = expr_str.rfind('ite(')
        if start == -1:
            break

        # Find matching close paren
        depth = 0
        i = start + 4  # Start after 'ite('
        parts = []
        current = ""

        while i < len(expr_str):
            ch = expr_str[i]
            if ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                if depth == 0:
                    # End of this ite
                    parts.append(current)
                    break
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                # Separator between cond/then/else
                parts.append(current)
                current = ""
            else:
                current += ch
            i += 1

        if len(parts) == 3:
            # Convert ite(cond, then, else) to (cond ? then : else)
            cond, then_part, else_part = parts
            replacement = f"({cond.strip()} ? {then_part.strip()} : {else_part.strip()})"
            expr_str = expr_str[:start] + replacement + expr_str[i+1:]
        else:
            break  # Malformed, stop

    # Replace negation operator
    expr_str = expr_str.replace('~ ', '!')
    expr_str = expr_str.replace('~(', '!(')

    return expr_str


def bdd_to_systemverilog_function(bdd_expr, func_name: str = "matches_instruction",
                                   signal_name: str = "instr", width: int = 32) -> str:
    """
    Generate a SystemVerilog function that implements the BDD decision tree.

    Args:
        bdd_expr: BDD expression
        func_name: Name for the generated function
        signal_name: Parameter name for instruction word
        width: Instruction width

    Returns:
        Complete SystemVerilog function definition

    Example output:
        function automatic bit matches_instruction(logic [31:0] instr);
          return (instr[11] ? 1'b0 : ((instr & 32'h...) == 32'h...));
        endfunction
    """
    bdd_mgr = get_bdd_manager(width)

    # Get compact expression from BDD
    expr_str = bdd_mgr.to_expr(bdd_expr)

    # Convert to SystemVerilog syntax
    sv_expr = _convert_bdd_expr_to_sv(expr_str, signal_name, width)

    # Generate function
    code = f"  function automatic bit {func_name}(logic [{width-1}:0] {signal_name});\n"
    code += f"    return {sv_expr};\n"
    code += f"  endfunction\n"

    return code


def reset_bdd_managers():
    """Reset global BDD managers (for testing)."""
    global _bdd_manager_32, _bdd_manager_16
    _bdd_manager_32 = None
    _bdd_manager_16 = None
