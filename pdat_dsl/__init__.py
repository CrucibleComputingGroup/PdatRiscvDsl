"""
PDAT DSL - A Domain-Specific Language for specifying RISC-V ISA subsets.

This package provides tools for:
- Parsing DSL files that specify instruction constraints
- Generating SystemVerilog assertion modules
- Encoding RISC-V instruction patterns

Example DSL syntax:
    # Require RV32I base instruction set
    require RV32I

    # Limit to 16 registers
    require_registers x0-x15

    # Outlaw specific instructions
    instruction MUL
    instruction DIV
"""

__version__ = "0.1.0"

from .parser import (
    parse_dsl,
    Program,
    InstructionRule,
    PatternRule,
    RequireRule,
    RegisterConstraintRule,
    FieldConstraint,
)

from .encodings import (
    InstructionEncoding,
    get_instruction_encoding,
    get_extension_instructions,
    parse_register,
)

from .codegen import (
    instruction_rule_to_pattern,
    generate_inline_assumptions,
)

# VCD analysis tools (for simulation-based signal correspondence)
# These can be imported as modules since they have main() functions
from . import vcd_to_state_json
from . import find_signal_correspondences

__all__ = [
    # Parser
    "parse_dsl",
    "Program",
    "InstructionRule",
    "PatternRule",
    "RequireRule",
    "RegisterConstraintRule",
    "FieldConstraint",
    # Encodings
    "InstructionEncoding",
    "get_instruction_encoding",
    "get_extension_instructions",
    "parse_register",
    # Code generation
    "instruction_rule_to_pattern",
    "generate_inline_assumptions",
]
