#!/usr/bin/env python3
"""
Core configuration management for PDAT DSL.

This module handles loading and validating YAML configuration files that
specify core-specific signal names, hierarchy, and other microarchitecture
details. This allows the DSL to target different RISC-V cores (Ibex, BOOM,
Rocket, CVA6, etc.) without code changes.
"""

import yaml
import jsonschema
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# JSON Schema for validating core configuration YAML files
CORE_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "PDAT Core Configuration",
    "description": "Configuration for targeting different RISC-V cores",
    "type": "object",
    "required": ["core_name", "architecture", "signals"],
    "properties": {
        "core_name": {
            "type": "string",
            "description": "Name of the RISC-V core (e.g., ibex, boom, rocket)"
        },
        "architecture": {
            "type": "string",
            "enum": ["rv32", "rv64"],
            "description": "RISC-V architecture variant"
        },
        "signals": {
            "type": "object",
            "required": ["instruction_data", "pc"],
            "properties": {
                "instruction_data": {
                    "type": "string",
                    "description": "Signal name for instruction word"
                },
                "pc": {
                    "type": "string",
                    "description": "Signal name for program counter"
                },
                "operands": {
                    "type": "object",
                    "description": "Operand signal names by execution unit",
                    "properties": {
                        "alu": {
                            "type": "object",
                            "required": ["rs1", "rs2"],
                            "properties": {
                                "rs1": {"type": "string", "description": "ALU source register 1"},
                                "rs2": {"type": "string", "description": "ALU source register 2"}
                            },
                            "additionalProperties": False
                        },
                        "multdiv": {
                            "type": "object",
                            "required": ["rs1", "rs2"],
                            "properties": {
                                "rs1": {"type": "string", "description": "MUL/DIV source register 1"},
                                "rs2": {"type": "string", "description": "MUL/DIV source register 2"}
                            },
                            "additionalProperties": False
                        }
                    },
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        },
        "vcd": {
            "type": "object",
            "properties": {
                "testbench_prefix": {
                    "type": "string",
                    "description": "VCD signal path prefix to strip"
                }
            },
            "additionalProperties": False
        }
    },
    "additionalProperties": False
}


@dataclass
class SignalConfig:
    """Signal name configuration for a specific core."""
    # Instruction and PC signals
    instruction_data: str = "instr_rdata_i"
    pc: str = "pc_if_o"

    # Operand signals by execution unit
    alu_rs1: str = "alu_operand_a_ex_i"
    alu_rs2: str = "alu_operand_b_ex_i"
    multdiv_rs1: str = "multdiv_operand_a_ex_i"
    multdiv_rs2: str = "multdiv_operand_b_ex_i"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SignalConfig':
        """Create SignalConfig from dictionary (loaded from YAML)."""
        signals = cls()

        if 'instruction_data' in d:
            signals.instruction_data = d['instruction_data']
        if 'pc' in d:
            signals.pc = d['pc']

        # Handle operand signals (can be flat or nested)
        if 'operands' in d:
            operands = d['operands']
            if 'alu' in operands:
                if 'rs1' in operands['alu']:
                    signals.alu_rs1 = operands['alu']['rs1']
                if 'rs2' in operands['alu']:
                    signals.alu_rs2 = operands['alu']['rs2']
            if 'multdiv' in operands:
                if 'rs1' in operands['multdiv']:
                    signals.multdiv_rs1 = operands['multdiv']['rs1']
                if 'rs2' in operands['multdiv']:
                    signals.multdiv_rs2 = operands['multdiv']['rs2']

        return signals


@dataclass
class VcdConfig:
    """VCD analysis configuration."""
    testbench_prefix: str = "tb_ibex_random.dut"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VcdConfig':
        """Create VcdConfig from dictionary."""
        vcd = cls()
        if 'testbench_prefix' in d:
            vcd.testbench_prefix = d['testbench_prefix']
        return vcd


@dataclass
class CoreConfig:
    """Complete configuration for a RISC-V core target."""
    core_name: str = "ibex"
    architecture: str = "rv32"  # rv32 or rv64
    signals: SignalConfig = field(default_factory=SignalConfig)
    vcd: VcdConfig = field(default_factory=VcdConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'CoreConfig':
        """Load and validate configuration from YAML file.

        Raises:
            jsonschema.ValidationError: If config doesn't match schema
            yaml.YAMLError: If YAML is malformed
            FileNotFoundError: If file doesn't exist
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Validate against JSON schema
        try:
            jsonschema.validate(instance=data, schema=CORE_CONFIG_SCHEMA)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid config file {yaml_path}: {e.message}") from e

        config = cls()

        if 'core_name' in data:
            config.core_name = data['core_name']
        if 'architecture' in data:
            config.architecture = data['architecture']

        if 'signals' in data:
            config.signals = SignalConfig.from_dict(data['signals'])
        if 'vcd' in data:
            config.vcd = VcdConfig.from_dict(data['vcd'])

        return config

    @classmethod
    def default_ibex(cls) -> 'CoreConfig':
        """Return default Ibex configuration (for backward compatibility)."""
        return cls(
            core_name="ibex",
            architecture="rv32",
            signals=SignalConfig(),
            vcd=VcdConfig()
        )

    def get_operand_signal(self, instruction: str, operand: str) -> Optional[str]:
        """
        Get the signal name for a specific instruction's operand.

        Args:
            instruction: Instruction name (e.g., "MUL", "ADD")
            operand: Operand field name ("rs1", "rs2", "rd")

        Returns:
            Signal name or None if not found
        """
        # Classify instruction type
        mul_div_instrs = {'MUL', 'MULH', 'MULHSU', 'MULHU', 'DIV', 'DIVU', 'REM', 'REMU'}

        instr_upper = instruction.upper()

        if instr_upper in mul_div_instrs:
            # Multiply/Divide unit
            if operand == 'rs1':
                return self.signals.multdiv_rs1
            elif operand == 'rs2':
                return self.signals.multdiv_rs2
        else:
            # Regular ALU instruction
            if operand == 'rs1':
                return self.signals.alu_rs1
            elif operand == 'rs2':
                return self.signals.alu_rs2

        return None

    @property
    def data_width(self) -> int:
        """Return data width in bits (32 for RV32, 64 for RV64)."""
        if self.architecture.lower().startswith('rv64'):
            return 64
        return 32


def load_config(config_path: Optional[Path] = None, target: Optional[str] = None) -> CoreConfig:
    """
    Load core configuration from file or use builtin config.

    Args:
        config_path: Path to YAML config file
        target: Shortcut name for builtin configs ('ibex', 'boom', 'rocket')

    Returns:
        CoreConfig object

    Priority:
        1. config_path if provided
        2. builtin config matching target name
        3. default Ibex config
    """
    if config_path and config_path.exists():
        return CoreConfig.from_yaml(config_path)

    if target:
        # Try to load builtin config
        builtin_path = Path(__file__).parent.parent / "configs" / f"{target}.yaml"
        if builtin_path.exists():
            return CoreConfig.from_yaml(builtin_path)

    # Default to Ibex
    return CoreConfig.default_ibex()
