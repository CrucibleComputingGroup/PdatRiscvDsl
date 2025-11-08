#!/usr/bin/env python3
"""
V2 DSL rule processor using AIG-based per-instruction constraints.

This module implements the DSL v2 semantics:
    σ₀ = ∅ (empty set)
    include e: σ := σ ∪ ⟦e⟧
    forbid e:  σ := σ ∧ ¬⟦e⟧

But instead of flat pattern/mask lists, we maintain per-instruction field constraints
using AIGs for maximum expressiveness and optimization.
"""

from typing import Dict, Set, Optional, List, Tuple
from dataclasses import dataclass, replace
from aigverse import Aig

from .parser import (
    IncludeRule, ForbidRule, RegisterRangeExpression, InstructionPattern,
    Program
)
from .encodings import (
    get_instruction_encoding, get_extension_instructions, ALL_INSTRUCTIONS,
    InstructionEncoding
)
from .aig_constraints import (
    InstructionConstraints, FieldConstraint,
    range_to_aig, set_to_aig, complement_aig, and_aigs
)


# =============================================================================
# Per-Instruction Constraint Database
# =============================================================================

class InstructionDatabase:
    """
    Maintains per-instruction field constraints using AIGs.

    Key: instruction identifier (opcode + funct bits)
    Value: InstructionConstraints with AIG field constraints
    """

    def __init__(self):
        # Map from instruction name to constraints
        self.constraints: Dict[str, InstructionConstraints] = {}

    def add_instruction(self, instr_name: str, encoding: InstructionEncoding):
        """
        Add an instruction with unconstrained fields (all values allowed).

        Args:
            instr_name: Instruction name (e.g., "ADD", "ADDI")
            encoding: Instruction encoding information
        """
        if instr_name in self.constraints:
            return  # Already exists

        # Create constraint with unconstrained fields (TRUE AIGs)
        # Field positions in RISC-V encoding:
        # rd: [11:7] base=7, rs1: [19:15] base=15, rs2: [24:20] base=20
        constraint = InstructionConstraints(
            instruction_name=instr_name,
            opcode_pattern=encoding.base_pattern,
            opcode_mask=encoding.base_mask,
            rd=self._create_unconstrained_field(5, 7) if 'rd' in encoding.fields else None,
            rs1=self._create_unconstrained_field(5, 15) if 'rs1' in encoding.fields else None,
            rs2=self._create_unconstrained_field(5, 20) if 'rs2' in encoding.fields else None,
            imm=self._create_unconstrained_field_for_imm(encoding) if 'imm' in encoding.fields else None
        )

        self.constraints[instr_name] = constraint

    def remove_instruction(self, instr_name: str):
        """Remove an instruction entirely from the database."""
        if instr_name in self.constraints:
            del self.constraints[instr_name]

    def constrain_field(self, instr_name: str, field_name: str, constraint_aig: Aig, width: int):
        """
        AND a field constraint with existing constraint.

        This is used to progressively narrow down allowed field values.

        Args:
            instr_name: Instruction to constrain
            field_name: Field name ('rd', 'rs1', 'rs2', 'imm')
            constraint_aig: AIG representing additional constraint
            width: Field width in bits
        """
        if instr_name not in self.constraints:
            return  # Instruction not in set

        constraint = self.constraints[instr_name]

        # Get current field constraint
        current = getattr(constraint, field_name)

        if current is None:
            # Field doesn't exist for this instruction
            return

        # AND the constraints together
        combined_aig = and_aigs(current.aig, constraint_aig)
        # Preserve base_position from current constraint
        new_field_constraint = FieldConstraint(combined_aig, width, current.base_position)

        setattr(constraint, field_name, new_field_constraint)

    def constrain_field_with_metadata(self, instr_name: str, field_name: str, constraint_aig: Aig, width: int, range_metadata: Optional[Tuple[int, int]]):
        """
        AND a field constraint with existing constraint, preserving range metadata.

        Args:
            instr_name: Instruction to constrain
            field_name: Field name ('rd', 'rs1', 'rs2', 'imm')
            constraint_aig: AIG representing additional constraint
            width: Field width in bits
            range_metadata: Optional (min, max) tuple for simple range constraints
        """
        if instr_name not in self.constraints:
            return

        constraint = self.constraints[instr_name]
        current = getattr(constraint, field_name)

        if current is None:
            return

        # AND the constraints together
        combined_aig = and_aigs(current.aig, constraint_aig)
        # Create new constraint with metadata
        new_field_constraint = FieldConstraint(
            aig=combined_aig,
            width=width,
            base_position=current.base_position,
            range_constraint=range_metadata  # Track the allowed range
        )

        setattr(constraint, field_name, new_field_constraint)

    def apply_register_range_forbid(self, reg_range: RegisterRangeExpression):
        """
        Forbid a register range across all instructions.

        For each instruction, for each register field, AND with NOT(range constraint).

        Args:
            reg_range: RegisterRangeExpression with set of registers to forbid
        """
        # Create NOT(register in range) constraint
        # First create: register IN range
        min_reg = min(reg_range.registers)
        max_reg = max(reg_range.registers)

        # Check if contiguous for optimization
        if reg_range.is_contiguous():
            # Use optimized range constraint
            in_range_aig = range_to_aig(min_reg, max_reg, 5)
        else:
            # Use set constraint
            in_range_aig = set_to_aig(reg_range.registers, 5)

        # Create NOT(in_range) = allowed range
        # If forbidding x4-x31, allowed is x0-x3
        not_in_range = self._invert_aig_output(in_range_aig)

        # Calculate the allowed range for metadata
        # If forbidding [min_reg, max_reg], allowed is [0, min_reg-1]
        allowed_range = None
        if min_reg > 0 and max_reg == 31:
            # Forbidding [min, 31] -> allow [0, min-1]
            allowed_range = (0, min_reg - 1)

        # Apply to all instructions and all register fields
        for instr_name, constraint in list(self.constraints.items()):
            # Apply to rd
            if constraint.rd is not None:
                self.constrain_field_with_metadata(instr_name, 'rd', not_in_range, 5, allowed_range)

            # Apply to rs1
            if constraint.rs1 is not None:
                self.constrain_field_with_metadata(instr_name, 'rs1', not_in_range, 5, allowed_range)

            # Apply to rs2
            if constraint.rs2 is not None:
                self.constrain_field_with_metadata(instr_name, 'rs2', not_in_range, 5, allowed_range)

    def _create_unconstrained_field(self, width: int, base_position: int) -> FieldConstraint:
        """Create a field constraint that allows all values (TRUE)."""
        aig = Aig()
        # Create inputs but ignore them - always return TRUE
        for _ in range(width):
            aig.create_pi()

        # Output is constant TRUE
        true_signal = aig.get_constant(True)
        aig.create_po(true_signal)

        return FieldConstraint(aig, width, base_position)

    def _create_unconstrained_field_for_imm(self, encoding: InstructionEncoding) -> Optional[FieldConstraint]:
        """Create unconstrained immediate field based on instruction format."""
        if 'imm' not in encoding.fields:
            return None

        pos, width = encoding.fields['imm']
        return self._create_unconstrained_field(width, pos)

    def _invert_aig_output(self, aig: Aig) -> Aig:
        """Create new AIG with inverted output."""
        return complement_aig(aig)


# =============================================================================
# V2 Rule Processing
# =============================================================================

def process_v2_rules_aig(rules: List, has_c_ext: bool = False) -> InstructionDatabase:
    """
    Process v2 rules using AIG-based per-instruction constraints.

    V2 Semantics:
        σ₀ = ∅ (start with empty database)
        include e: σ := σ ∪ ⟦e⟧  (add instructions/constraints)
        forbid e:  σ := σ ∧ ¬⟦e⟧ (remove or constrain)

    Args:
        rules: List of v2 rules (IncludeRule, ForbidRule, etc.)
        has_c_ext: Whether C extension is required

    Returns:
        InstructionDatabase with final constraint set
    """
    db = InstructionDatabase()

    print("Processing v2 rules with AIG-based constraints...")

    for i, rule in enumerate(rules, 1):
        if isinstance(rule, IncludeRule):
            process_include_rule(db, rule, has_c_ext)
            print(f"  {i}. include: σ now contains {len(db.constraints)} instructions")

        elif isinstance(rule, ForbidRule):
            old_count = len(db.constraints)
            process_forbid_rule(db, rule, has_c_ext)
            new_count = len(db.constraints)
            removed = old_count - new_count
            print(f"  {i}. forbid: removed/constrained {removed} instructions, σ now contains {new_count}")

    print(f"Final instruction set: {len(db.constraints)} instructions with field constraints")
    return db


def process_include_rule(db: InstructionDatabase, rule: IncludeRule, has_c_ext: bool):
    """
    Process an include rule: σ := σ ∪ ⟦e⟧

    Args:
        db: Instruction database to update
        rule: Include rule to process
        has_c_ext: Whether C extension is active
    """
    expr = rule.expr

    if isinstance(expr, str):
        # Extension name or instruction name
        instructions = get_extension_instructions(expr)

        if instructions:
            # It's an extension - add all instructions
            for instr_name in instructions:
                encoding = get_instruction_encoding(instr_name)
                if encoding:
                    db.add_instruction(instr_name, encoding)
        else:
            # Single instruction name
            encoding = get_instruction_encoding(expr)
            if encoding:
                db.add_instruction(expr, encoding)
            else:
                print(f"Warning: Unknown extension or instruction '{expr}'")

    elif isinstance(expr, InstructionPattern):
        # Instruction with field constraints
        # TODO: Handle InstructionPattern (e.g., "ADD {rd in x0-x15}")
        print(f"Warning: InstructionPattern not yet implemented in AIG processor")

    elif isinstance(expr, RegisterRangeExpression):
        # Register range - this is tricky for include
        # "include x0-x15" means "include all instructions using ONLY x0-x15"
        # This is equivalent to: include all instructions, then forbid x16-x31
        print(f"Warning: RegisterRangeExpression in include not yet fully implemented")
        # For now, this would need to:
        # 1. Add all instructions
        # 2. Constrain their register fields

    else:
        print(f"Warning: Unknown expression type in include: {type(expr)}")


def process_forbid_rule(db: InstructionDatabase, rule: ForbidRule, has_c_ext: bool):
    """
    Process a forbid rule: σ := σ ∧ ¬⟦e⟧

    Args:
        db: Instruction database to update
        rule: Forbid rule to process
        has_c_ext: Whether C extension is active
    """
    expr = rule.expr

    if isinstance(expr, str):
        # Extension name or instruction name
        instructions = get_extension_instructions(expr)

        if instructions:
            # It's an extension - remove all instructions
            for instr_name in instructions:
                db.remove_instruction(instr_name)
        else:
            # Single instruction name
            db.remove_instruction(expr)

    elif isinstance(expr, InstructionPattern):
        # Instruction with field constraints
        # TODO: Handle InstructionPattern
        print(f"Warning: InstructionPattern not yet implemented in AIG processor")

    elif isinstance(expr, RegisterRangeExpression):
        # Register range - forbid these registers across all instructions
        print(f"  Forbidding register range: x{min(expr.registers)}-x{max(expr.registers)}")
        db.apply_register_range_forbid(expr)

    else:
        print(f"Warning: Unknown expression type in forbid: {type(expr)}")
