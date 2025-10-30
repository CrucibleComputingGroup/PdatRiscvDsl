# Core Configuration Files

This directory contains YAML configuration files for different RISC-V cores. These configs specify core-specific signal names, module hierarchy, and other microarchitecture details.

## Available Configurations

- **`ibex.yaml`**: lowRISC Ibex (RV32IMC) - Default configuration
- **`boom.yaml`**: Berkeley Out-of-Order Machine (RV64GC) - Example config
- **`rocket.yaml`**: Rocket Core (RV64GC) - Example config

## Usage

### Using a specific config:
```bash
pdat-dsl codegen --config configs/ibex.yaml example.dsl output.sv
```

### Using target shortcut:
```bash
pdat-dsl codegen --target ibex example.dsl output.sv
pdat-dsl codegen --target boom example.dsl output.sv
```

### Default behavior (no args):
```bash
pdat-dsl codegen example.dsl output.sv  # Uses ibex.yaml by default
```

## Creating a New Configuration

To add support for a new RISC-V core:

1. Copy `ibex.yaml` as a template
2. Update signal names to match your core's signals
3. Update binding configuration for your hierarchy
4. Save as `configs/<corename>.yaml`

### Configuration Structure

```yaml
core_name: "your_core"
architecture: "rv32"  # or rv64

signals:
  instruction_data: "signal_name"
  instruction_valid: "signal_name"
  instruction_compressed: "signal_name"
  pc: "signal_name"

  operands:
    alu:
      rs1: "signal_name"
      rs2: "signal_name"
    multdiv:
      rs1: "signal_name"
      rs2: "signal_name"

binding:
  target_module: "path.to.module"
  instance_name: "checker_inst"

vcd:
  testbench_prefix: "tb.dut"
```

## Finding Signal Names

To discover signal names for your core:

1. **Use VCD waveforms**: Run a simulation and inspect signals in GTKWave/Verdi
2. **Read RTL source**: Look at module ports and signal declarations
3. **Check documentation**: Core documentation often includes signal descriptions

### Key signals to find:

- **Instruction data**: The instruction word being executed
- **PC**: Program counter at instruction fetch or decode
- **Operands**: Data values from register file (rs1, rs2)
  - Usually at EX (execute) stage
  - May differ for ALU vs multiply/divide units

## Notes

- **BOOM and Rocket configs are examples** - You must verify signal names match your specific core version and configuration
- **Signal timing matters**: Operand signals should be sampled where data type checks are meaningful (typically EX stage)
- **Out-of-order cores**: May need special handling for speculation/rollback
