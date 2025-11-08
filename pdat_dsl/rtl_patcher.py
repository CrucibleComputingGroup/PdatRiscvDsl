#!/usr/bin/env python3
"""
RTL Patching System for Hierarchical Synthesis

This module patches SystemVerilog RTL to add instruction/PC input ports
to submodules, enabling hierarchical synthesis where operand signals become
primary inputs with data type constraints.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModulePort:
    """Represents a SystemVerilog module port."""
    direction: str  # 'input', 'output', 'inout'
    port_type: str  # e.g., 'logic', 'logic [31:0]', 'custom_type'
    name: str
    comment: str = ""


def parse_module_header(rtl_content: str, module_name: str) -> Tuple[int, int, List[ModulePort]]:
    """
    Parse a SystemVerilog module header to find port list.

    Args:
        rtl_content: Full RTL file content
        module_name: Name of the module to find

    Returns:
        Tuple of (start_line, end_line, port_list)
        start_line: Line number where module declaration starts
        end_line: Line number where port list ends (before closing );)
        port_list: List of ModulePort objects

    Raises:
        ValueError: If module not found or cannot parse
    """
    lines = rtl_content.split('\n')

    # Find module declaration
    module_pattern = rf'^\s*module\s+{re.escape(module_name)}\s*[#(]'
    module_start = None
    for i, line in enumerate(lines):
        if re.search(module_pattern, line):
            module_start = i
            break

    if module_start is None:
        raise ValueError(f"Module '{module_name}' not found in RTL")

    # Find the start of port list (first line with 'input' or 'output' after module keyword)
    port_list_start = None
    for i in range(module_start, len(lines)):
        if re.search(r'^\s*(input|output|inout)\s', lines[i]):
            port_list_start = i
            break

    if port_list_start is None:
        raise ValueError(f"Could not find port list for module '{module_name}'")

    # Find end of port list (line with );)
    port_list_end = None
    for i in range(port_list_start, len(lines)):
        if re.search(r'\);', lines[i]):
            port_list_end = i
            break

    if port_list_end is None:
        raise ValueError(f"Could not find end of port list for module '{module_name}'")

    # Parse individual ports (simplified - handles most common cases)
    ports = []
    # This is a simplified parser - a full parser would handle multi-line ports better

    return (module_start, port_list_end, ports)


def add_ports_to_module(rtl_path: Path, module_name: str,
                        ports_to_add: Dict[str, str],
                        output_path: Path) -> None:
    """
    Add new input ports to a SystemVerilog module.

    Args:
        rtl_path: Path to original RTL file
        module_name: Name of module to patch
        ports_to_add: Dict mapping port names to types (e.g., {'instr_rdata_i': 'logic [31:0]'})
        output_path: Where to write patched RTL

    Strategy:
        - Find the last input port in the module
        - Insert new ports after it (before outputs)
        - Maintain formatting and comments
    """
    with open(rtl_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')

    # Find module declaration
    module_pattern = rf'^\s*module\s+{re.escape(module_name)}\s*'
    module_start = None
    for i, line in enumerate(lines):
        if re.search(module_pattern, line):
            module_start = i
            break

    if module_start is None:
        raise ValueError(f"Module '{module_name}' not found in {rtl_path}")

    # Find the last input port declaration
    # Strategy: Find all lines with "input" between module start and first "output"
    last_input_line = None
    for i in range(module_start, len(lines)):
        line = lines[i]
        if re.search(r'^\s*input\s', line):
            last_input_line = i
        elif re.search(r'^\s*output\s', line):
            # Stop when we hit outputs
            break

    if last_input_line is None:
        raise ValueError(f"Could not find input ports in module '{module_name}'")

    # Build new port declarations
    # Match the indentation of existing ports
    existing_line = lines[last_input_line]
    indent_match = re.match(r'^(\s*)', existing_line)
    indent = indent_match.group(1) if indent_match else '  '

    new_port_lines = []
    new_port_lines.append("")
    new_port_lines.append(f"{indent}// === Added for hierarchical synthesis ===")

    for port_name, port_type in ports_to_add.items():
        new_port_lines.append(f"{indent}input  {port_type:<20}  {port_name},")

    new_port_lines.append(f"{indent}// === End added ports ===")

    # Insert new ports after last input
    patched_lines = (
        lines[:last_input_line + 1] +
        new_port_lines +
        lines[last_input_line + 1:]
    )

    # Write patched file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(patched_lines))

    print(f"Patched module '{module_name}': added {len(ports_to_add)} ports")
    print(f"  Output: {output_path}")


def patch_parent_module(parent_rtl_path: Path, child_instance_name: str,
                        signals_to_connect: Dict[str, str],
                        output_path: Path) -> None:
    """
    Patch parent module to wire new signals to child instance.

    Args:
        parent_rtl_path: Path to parent module RTL (e.g., ibex_core.sv)
        child_instance_name: Instance name of child (e.g., 'ex_block_i')
        signals_to_connect: Dict mapping child port names to parent signal names
        output_path: Where to write patched parent RTL

    Strategy:
        - Find child module instantiation
        - Find end of port connections (before closing );)
        - Add new port connections
    """
    with open(parent_rtl_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')

    # Find child instance
    instance_pattern = rf'{re.escape(child_instance_name)}\s*\('
    instance_start = None
    for i, line in enumerate(lines):
        if re.search(instance_pattern, line):
            instance_start = i
            break

    if instance_start is None:
        raise ValueError(f"Instance '{child_instance_name}' not found in {parent_rtl_path}")

    # Find end of port list (look for );)
    instance_end = None
    for i in range(instance_start, len(lines)):
        if re.search(r'\);', lines[i]):
            instance_end = i
            break

    if instance_end is None:
        raise ValueError(f"Could not find end of instance '{child_instance_name}'")

    # Get indentation from existing port connections
    # Look at a few lines before the end to find a port connection
    indent = '    '  # default
    for i in range(max(instance_start, instance_end - 10), instance_end):
        port_match = re.match(r'^(\s*)\.\w+', lines[i])
        if port_match:
            indent = port_match.group(1)
            break

    # Find the last port connection line and add comma if missing
    last_port_line = None
    for i in range(instance_end - 1, instance_start, -1):
        if re.search(r'\.\w+\s*\(', lines[i]):
            last_port_line = i
            break

    # Add comma to last port if it doesn't have one
    if last_port_line is not None:
        if not lines[last_port_line].rstrip().endswith(','):
            lines[last_port_line] = lines[last_port_line].rstrip() + ','

    # Build new port connections (last one without comma)
    new_connections = []
    new_connections.append("")
    new_connections.append(f"{indent}// === Added for hierarchical synthesis ===")

    port_items = list(signals_to_connect.items())
    for idx, (child_port, parent_signal) in enumerate(port_items):
        is_last = (idx == len(port_items) - 1)
        comma = "" if is_last else ","
        new_connections.append(f"{indent}.{child_port:<30}({parent_signal}){comma}")

    new_connections.append(f"{indent}// === End added connections ===")

    # Insert before the closing );
    patched_lines = (
        lines[:instance_end] +
        new_connections +
        [lines[instance_end]]  # The closing );
        + lines[instance_end + 1:]
    )

    # Write patched file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(patched_lines))

    print(f"Patched parent module: wired {len(signals_to_connect)} signals to '{child_instance_name}'")
    print(f"  Output: {output_path}")


def inject_code_into_module(rtl_path: Path, module_name: str,
                            code_to_inject: str, output_path: Path) -> None:
    """
    Inject code (e.g., assumptions) into a SystemVerilog module before endmodule.

    Args:
        rtl_path: Path to RTL file
        module_name: Name of module to inject into
        code_to_inject: Code to inject (should be properly indented)
        output_path: Where to write patched RTL
    """
    with open(rtl_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')

    # Find the module and its endmodule
    module_pattern = rf'^\s*module\s+{re.escape(module_name)}\s*'
    module_start = None
    for i, line in enumerate(lines):
        if re.search(module_pattern, line):
            module_start = i
            break

    if module_start is None:
        raise ValueError(f"Module '{module_name}' not found in {rtl_path}")

    # Find endmodule
    endmodule_line = None
    for i in range(module_start, len(lines)):
        if re.match(r'^\s*endmodule', lines[i]):
            endmodule_line = i
            break

    if endmodule_line is None:
        raise ValueError(f"Could not find endmodule for '{module_name}'")

    # Get indentation from a line near the end
    indent = '  '  # default

    # Insert code before endmodule
    injected_lines = code_to_inject.split('\n')
    patched_lines = (
        lines[:endmodule_line] +
        [''] +
        injected_lines +
        [''] +
        lines[endmodule_line:]
    )

    # Write patched file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(patched_lines))

    print(f"Injected {len(injected_lines)} lines of code into module '{module_name}'")
    print(f"  Output: {output_path}")


def generate_patched_hierarchy(module_config, parent_signals: Dict[str, str],
                                core_rtl_dir: Path, output_dir: Path) -> Tuple[Path, Path]:
    """
    Generate complete patched hierarchy for a module.

    Args:
        module_config: ModuleConfig object
        parent_signals: Dict mapping added port names to parent module signal names
        core_rtl_dir: Directory containing core RTL files
        output_dir: Output directory for patched files

    Returns:
        Tuple of (patched_module_path, patched_parent_path)
    """
    # 1. Patch the child module to add ports
    original_module_path = core_rtl_dir / Path(module_config.rtl_path).name
    patched_module_path = output_dir / f"{module_config.name}_patched.sv"

    # Convert added_ports config to port declarations
    port_declarations = {}
    for port_key, port_name in module_config.added_ports.items():
        if port_key == 'instruction_data':
            port_declarations[port_name] = 'logic [31:0]'
        elif port_key == 'instruction_compressed':
            port_declarations[port_name] = 'logic'
        elif port_key == 'pc':
            port_declarations[port_name] = 'logic [31:0]'

    # Extract module name from path (e.g., ibex_ex_block.sv -> ibex_ex_block)
    module_name = Path(module_config.rtl_path).stem

    add_ports_to_module(
        original_module_path,
        module_name,
        port_declarations,
        patched_module_path
    )

    # 2. Patch the parent module to wire signals
    # For now, we'll skip this and do it manually or in a later phase
    # This is complex because we need to know the parent module and instance name

    return (patched_module_path, None)
