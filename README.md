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
field_value = wildcard | number | register_name
```

### Keywords

- `require` - Specify required instruction extensions (RV32I, RV32M, RV64I, etc.)
- `require_registers` - Limit which registers can be used
- `instruction` - Outlaw a specific instruction by name
- `pattern` / `mask` - Outlaw instructions matching a specific bit pattern

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
│   └── rv32im.dsl
├── tests/              # Unit tests
├── pyproject.toml      # Package configuration
└── README.md           # This file
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
