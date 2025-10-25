# PDAT DSL - RISC-V ISA Subset Specification Language

A Domain-Specific Language (DSL) for specifying RISC-V ISA subsets and generating SystemVerilog assertions for formal verification and hardware synthesis optimization.

## Overview

PDAT DSL allows you to:
- Specify which RISC-V instruction extensions are required
- Constrain the register file size
- Outlaw specific instructions or instruction patterns
- Generate SystemVerilog assertion modules for formal verification
- Generate inline assumptions for synthesis optimization (e.g., with ABC)

## Installation

### From source

```bash
cd PdatDsl
pip install -e .
```

### Development mode with dev dependencies

```bash
pip install -e ".[dev]"
```

## Usage

### Command-Line Interface

The package can be used via the `pdat-dsl` command or `python -m pdat_dsl`:

```bash
# Parse and validate a DSL file
pdat-dsl parse examples/example_16reg.dsl

# Generate SystemVerilog checker module
pdat-dsl codegen examples/example_16reg.dsl output.sv

# Generate inline SystemVerilog assumptions
pdat-dsl codegen --inline examples/example_16reg.dsl assumptions.sv

# Generate SMT2 constraints
pdat-dsl smt-constraints examples/example_16reg.dsl constraints.smt2

# Generate randomization constraints for VCS testbenches
pdat-dsl random-constraints examples/example_16reg.dsl instr_constraints.sv

# VCD Analysis (for simulation-based signal correspondence)
pdat-dsl find-correspondences simulation.vcd correspondences.json --constants-only
pdat-dsl vcd-to-state reset_state.vcd initial_state.json

# Run built-in tests
pdat-dsl test

# Show version
pdat-dsl version
```

### Python API

```python
from pdat_dsl import parse_dsl, generate_sv_module, instruction_rule_to_pattern

# Parse a DSL file
with open("my_spec.dsl") as f:
    dsl_text = f.read()

ast = parse_dsl(dsl_text)

# Process rules
patterns = []
for rule in ast.rules:
    if isinstance(rule, InstructionRule):
        patterns.extend(instruction_rule_to_pattern(rule))

# Generate SystemVerilog
sv_code = generate_sv_module(patterns)
print(sv_code)
```

## DSL Syntax

### Basic Structure

```
# Comments start with #

# Require specific RISC-V extensions
require RV32I
require RV32M

# Constrain register file (e.g., only use x0-x15)
require_registers x0-x15

# Outlaw specific instructions
instruction MUL
instruction MULH
instruction DIV

# Outlaw instructions with specific field constraints
instruction ADD { rd = x0 }

# Data type constraints (instruction-level)
instruction MUL { dtype = i16 }

# Data type constraints (per-operand)
instruction MULHU { rd_dtype = u16, rs1_dtype = u16, rs2_dtype = u16 }

# Multiple allowed types
instruction DIV { dtype = i8 | i16 }

# Combined register and data type constraints
instruction MUL { rd = x5, rs1_dtype = i16, rs2_dtype = i16 }

# Low-level pattern matching
pattern 0x02000033 mask 0xFE00707F
```

### Grammar

```
program = { rule }
rule = require_rule | register_constraint_rule | instruction_rule | pattern_rule | comment

require_rule = "require" extension_name
register_constraint_rule = "require_registers" register_range
instruction_rule = "instruction" identifier [ field_constraints ]
pattern_rule = "pattern" hex_value "mask" hex_value

register_range = register_name "-" register_name | number "-" number
field_constraints = "{" field_constraint { "," field_constraint } "}"
field_constraint = field_name "=" field_value
field_value = wildcard | number | register_name | data_type_set
data_type_set = data_type { "|" data_type }
data_type = ("i" | "u") ("8" | "16" | "32" | "64")
```

### Keywords

- `require` - Specify required instruction extensions (RV32I, RV32M, RV32C, RV64I, etc.)
- `require_registers` - Limit which registers can be used
- `instruction` - Outlaw a specific instruction by name
- `pattern` / `mask` - Outlaw instructions matching a specific bit pattern

### Compressed Instruction Support (C Extension)

When `require RV32C` or `require RV64C` is specified, the DSL automatically handles compressed instructions:

- **Auto-expansion**: Outlawing an instruction like `ADD` will automatically also outlaw its compressed equivalent `C.ADD` if it exists
- **Compression bit enforcement**: Generated constraints ensure `instr_is_compressed_i` matches the actual encoding bits[1:0]:
  - Compressed instructions have bits[1:0] ≠ 11
  - Uncompressed instructions have bits[1:0] = 11
- **Dual-width pattern matching**: 32-bit patterns check `instr_rdata_i[31:0]`, 16-bit patterns check `instr_rdata_i[15:0]`

You can also explicitly specify compressed instructions:
```
require RV32C

# Auto-expands to outlaw both ADD and C.ADD
instruction ADD

# Explicitly outlaw only the compressed version
instruction C.MV
instruction C.JALR
```

### Data Type Constraints

Data type constraints allow you to specify the width and signedness of operands used by instructions. These constraints follow the DSL's "outlawing" semantics but support negation to express positive requirements.

**Use cases:**
- **Hardware optimization**: Indicate that certain instructions only use narrow data widths (e.g., 8-bit or 16-bit), allowing synthesis tools to optimize away unused bits
- **Formal verification**: Generate assertions that operands fall within specified ranges
- **Power analysis**: Model data switching activity based on actual data widths used
- **Documentation**: Capture C-level data type semantics in hardware specifications

#### Data Type Syntax

Data types specify both width and signedness:

- **Signed types**: `i8`, `i16`, `i32`, `i64` (sign-extended to full register width)
- **Unsigned types**: `u8`, `u16`, `u32`, `u64` (zero-extended to full register width)

#### Negation Operator

The negation operator `~` is a **type expression prefix** (not a general operator) that inverts the constraint:

- **Without `~`**: Outlaw instructions using these types (negative constraint)
- **With `~`**: Outlaw instructions NOT using these types (positive constraint)

**Important**: The `~` can only appear once at the very beginning of a type expression, not nested or repeated.

#### Constraint Styles

**1. Negative constraints** (forbid specific types):
```
instruction MUL { dtype = i16 }           # Forbid MUL when using i16
instruction DIV { dtype = i8 | u8 }       # Forbid DIV when using i8 OR u8
```

**2. Positive constraints** (allow only specific types):
```
instruction MUL { dtype = ~i16 }          # Allow only i16 (forbid all others)
instruction DIV { dtype = ~(i8 | u8) }    # Allow only i8 or u8
```

**3. Per-operand constraints** (fine-grained control):
```
instruction MUL { rd_dtype = ~i16, rs1_dtype = ~i16, rs2_dtype = ~i16 }
instruction MULHU { rd_dtype = ~u32, rs1_dtype = ~u16, rs2_dtype = ~u16 }
```

**4. Combined with register constraints**:
```
instruction MUL { rd = x5, rs1_dtype = ~i8, rs2_dtype = ~(i8 | u8) }
```

#### Data Type Semantics

| Type | Width | Range | Extension |
|------|-------|-------|-----------|
| `i8` | 8-bit | -128 to 127 | Sign-extended |
| `u8` | 8-bit | 0 to 255 | Zero-extended |
| `i16` | 16-bit | -32768 to 32767 | Sign-extended |
| `u16` | 16-bit | 0 to 65535 | Zero-extended |
| `i32` | 32-bit | Full register (RV32) | N/A |
| `u32` | 32-bit | Full register (RV32) | N/A |
| `i64` | 64-bit | Full register (RV64) | N/A |
| `u64` | 64-bit | Full register (RV64) | N/A |

#### Type Expression Examples

```
# Negative constraints (outlaw specific types)
dtype = i8              # Forbid i8
dtype = i8 | u8         # Forbid i8 OR u8
dtype = i8 | i16        # Forbid i8 and i16 (must include all widths up to max)

# Positive constraints (allow only specific types)
dtype = ~i8             # Allow only i8
dtype = ~(i8 | u8)      # Allow only i8 or u8
dtype = ~(u8 | u16)     # Allow only u8 or u16 (must include all widths down to min)

# Invalid syntax (negation must be at start)
dtype = ~~i8            # Error: double negation
dtype = i8 | ~u8        # Error: negation in middle
dtype = ~i8 | ~u8       # Error: multiple negations

# Invalid semantics (creates gaps or ambiguity)
dtype = i8 | i32        # Error: missing i16 (gap)
dtype = ~(u16 | u32)    # Error: missing u8 (can't distinguish u8 from u16 when < 256)
dtype = u8 | u32        # Error: forbidding u8 requires forbidding u16 too
```

#### Validation Rules

**The key insight:** You cannot distinguish u8 from u16 when the value is < 256 using only bit patterns.

For a given signedness (signed or unsigned):

1. **Forbid constraints** (without `~`): When forbidding narrow types, must also forbid all wider types
   - Valid: `u8 | u16` (forbid up to u16, allow u32+)
   - Invalid: `u8 | u32` (missing u16 creates ambiguity)

2. **Allow constraints** (with `~`): When allowing wide types, must also allow all narrower types
   - Valid: `~(u8 | u16)` (allow u8 and u16)
   - Invalid: `~(u16 | u32)` (missing u8 - can't enforce!)

3. **No gaps allowed** within a signedness category
   - Valid: `i8 | i16 | i32`
   - Invalid: `i8 | i32` (missing i16)
```

## Examples

### Example 1: RV32I with 16 registers (RV32E-like)

```
# File: examples/example_16reg.dsl
require RV32I
require_registers x0-x15

instruction MUL
instruction MULH
instruction MULHSU
instruction MULHU
instruction DIV
instruction DIVU
instruction REM
instruction REMU
```

### Example 2: RV32IM without specific instructions

```
require RV32I
require RV32M

# Outlaw multiplication instructions
instruction MUL
instruction MULH
instruction MULHSU
instruction MULHU
```

### Example 2.5: RV32IC with compressed instructions

```
# File: examples/rv32ic.dsl
require RV32I
require RV32C

# Outlaw M extension (multiplication/division)
# Since RV32C is required, this will also outlaw compressed versions if they existed
instruction MUL
instruction MULH
instruction MULHSU
instruction MULHU
instruction DIV
instruction DIVU
instruction REM
instruction REMU
```

### Example 3: Data type constraints for narrow arithmetic

```
require RV32I
require RV32M

# Negative constraint: forbid MUL when using 16-bit signed
instruction MUL { dtype = i16 }

# Positive constraint: allow MUL ONLY with 16-bit signed (forbid all others)
instruction MUL { dtype = ~i16 }

# Allow MULHU only with unsigned 16-bit operands
instruction MULHU { rd_dtype = ~u16, rs1_dtype = ~u16, rs2_dtype = ~u16 }

# Mixed signedness: allow only these specific types
# rd and rs1 must be signed 8-bit, rs2 can be signed or unsigned 8-bit
instruction MULHSU { rd_dtype = ~i8, rs1_dtype = ~i8, rs2_dtype = ~(i8 | u8) }

# Forbid DIV when using 8-bit or 16-bit signed
instruction DIV { dtype = i8 | i16 }

# Allow DIV only when using 8-bit or 16-bit signed
instruction DIV { dtype = ~(i8 | i16) }

# Combine with register constraints
instruction MUL { rd = x5, rs1_dtype = ~i16, rs2_dtype = ~i16 }
```

See `examples/data_type_constraints.dsl` for a comprehensive example with detailed comments.

## Project Structure

```
PdatDsl/
├── pdat_dsl/                           # Main package
│   ├── __init__.py                     # Package exports
│   ├── __main__.py                     # CLI entry point
│   ├── parser.py                       # DSL lexer and parser
│   ├── encodings.py                    # RISC-V instruction encoding database
│   ├── codegen.py                      # SystemVerilog assertion generator
│   ├── random_constraints.py           # SV randomization constraint generator
│   ├── smt_constraints.py              # SMT2 constraint generator
│   ├── find_signal_correspondences.py  # VCD → equivalence classes JSON
│   └── vcd_to_state_json.py           # VCD → initial state JSON
├── examples/           # Example DSL files
│   ├── example_16reg.dsl
│   ├── example_outlawed.dsl
│   ├── rv32im.dsl
│   └── data_type_constraints.dsl
├── tests/                      # Unit tests
├── editors/                    # Editor support
│   ├── vscode/                # VS Code extension
│   │   ├── pdat-dsl.tmLanguage.json
│   │   └── package.json
│   └── README.md              # Editor setup instructions
├── tree-sitter-pdat-dsl/      # Tree-sitter grammar
│   ├── grammar.js
│   ├── queries/highlights.scm
│   └── test/corpus/
├── pyproject.toml              # Package configuration
└── README.md                   # This file
```

## Architecture

1. **Lexer** (`parser.py`): Tokenizes DSL input
2. **Parser** (`parser.py`): Builds Abstract Syntax Tree (AST)
3. **Encodings** (`encodings.py`): RISC-V instruction database
4. **Code Generator** (`codegen.py`): Generates SystemVerilog from AST

## Use Cases

### Formal Verification
Generate assertion modules that can be bound to your RTL design to formally verify that certain instructions never execute.

### Synthesis Optimization
Generate inline assumptions that help synthesis tools (like ABC) optimize away unused hardware (e.g., multiply/divide units for embedded cores).

### Simulation-Based Signal Correspondence (RTL-scorr)
The complete workflow for finding signal equivalences using constrained-random simulation:

1. **DSL → Randomization Constraints** (for VCS testbench)
   ```bash
   pdat-dsl random-constraints spec.dsl instr_constraints.sv
   # Include in testbench: `include "instr_constraints.sv"
   # Use in randomizer module to generate valid RISC-V instructions
   ```

2. **VCS Simulation** (generates VCD with DSL-constrained instructions)
   - Testbench uses `instr_constraints` class to randomize instruction memory
   - Phase 1: NOP cycles to flush pipeline
   - Phase 2: Constrained-random RISC-V instructions
   - VCS produces VCD trace file

3. **VCD → JSON** (extract simulation results)
   ```bash
   pdat-dsl find-correspondences simulation.vcd correspondences.json --constants-only
   pdat-dsl vcd-to-state reset_state.vcd initial_state.json
   ```

4. **Formal Verification** (using Yosys rtl_scorr plugin)
   - Plugin reads both JSON files
   - Proves equivalences with SMT-based k-induction
   - Identifies dead code for optimization

5. **Optional: Generate assumptions for synthesis**
   ```bash
   pdat-dsl codegen --inline spec.dsl assumptions.sv
   # Inject into RTL for synthesis with proven constant assumptions
   ```

### ISA Documentation
Use DSL files as machine-readable documentation of your processor's supported ISA subset.

## Editor Support

Syntax highlighting is available for multiple editors:

- **VS Code**: See `editors/vscode/` - TextMate grammar
- **Neovim/Helix/Emacs**: See `tree-sitter-pdat-dsl/` - Tree-sitter grammar
- **Vim**: See `editors/README.md` for syntax file

Installation instructions are in `editors/README.md`.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

- **Non-commercial use**: Free to use, modify, and share under the ShareAlike terms
- **Commercial use**: Contact Nathan Bleier (nbleier@umich.edu) for commercial licensing options

See the [LICENSE](LICENSE) file for details, or visit https://creativecommons.org/licenses/by-nc-sa/4.0/

## Related Projects

This DSL is part of the PDAT (Processor Design and Test) project ecosystem:
- **PdatScorr** - RTL scorecard and formal equivalence checking
- **PdatDsl** - ISA subset specification (this project)

## References

- [RISC-V ISA Specification](https://riscv.org/technical/specifications/)
- [SystemVerilog Assertions](https://standards.ieee.org/standard/1800-2017.html)
