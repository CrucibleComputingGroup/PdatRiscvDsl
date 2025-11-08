#!/usr/bin/env python3
"""
Hierarchical Synthesis Orchestrator

Orchestrates multi-stage hierarchical synthesis where submodules (like ibex_ex_block)
are synthesized separately with data type constraints on their inputs, then composed
back into the full design.
"""

import sys
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from .config import CoreConfig, ModuleConfig
from .parser import parse_dsl
from .codegen import generate_module_dtype_assertions
from .rtl_patcher import add_ports_to_module, inject_code_into_module, patch_parent_module


def parse_original_module_interface(rtl_path: Path, module_name: str) -> List[str]:
    """
    Extract the original port list from a module.

    Args:
        rtl_path: Path to original RTL file
        module_name: Name of module

    Returns:
        List of port declaration lines
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
        return []

    # Collect port lines until we hit endmodule or first begin/always
    port_lines = []
    in_ports = False
    for i in range(module_start, len(lines)):
        line = lines[i]

        # Start collecting after seeing first input/output
        if re.search(r'^\s*(input|output)', line):
            in_ports = True

        # Stop at endmodule or first procedural block
        if re.search(r'^\s*(endmodule|always|initial|assign)\s', line):
            break

        if in_ports:
            # Skip our added ports
            if 'Added for hierarchical synthesis' in line:
                # Skip until end marker
                for j in range(i, len(lines)):
                    if 'End added ports' in lines[j]:
                        i = j
                        break
                continue
            port_lines.append(line)

    return port_lines


def convert_aiger_to_rtlil(aiger_path: Path,
                           output_rtlil: Path,
                           module_name: str) -> bool:
    """
    Convert ABC-optimized AIGER to RTLIL format.

    Args:
        aiger_path: Path to optimized AIGER file
        output_rtlil: Where to write RTLIL file
        module_name: Module name for the AIGER

    Returns:
        True if conversion succeeded
    """
    # Create Yosys script
    ys_script = output_rtlil.parent / "aiger_to_rtlil.ys"
    ys_content = f"""# Convert ABC-optimized AIGER to RTLIL
read_aiger -module_name {module_name} -clk_name clk_i {aiger_path}
write_rtlil {output_rtlil}
"""

    with open(ys_script, 'w') as f:
        f.write(ys_content)

    try:
        result = subprocess.run(
            ['yosys', '-s', str(ys_script)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"  ERROR: AIGER to RTLIL conversion failed")
            print(f"  {result.stderr}")
            return False

        print(f"  Converted to RTLIL: {output_rtlil}")
        return True

    except Exception as e:
        print(f"  ERROR: Conversion failed: {e}")
        return False


def convert_aiger_to_structural_verilog(aiger_path: Path,
                                        output_verilog: Path,
                                        original_module_name: str,
                                        original_rtl_path: Optional[Path] = None) -> bool:
    """
    Convert optimized AIGER to structural Verilog with original module interface.

    Args:
        aiger_path: Path to optimized AIGER file from ABC
        output_verilog: Where to write structural Verilog
        original_rtl_path: Path to original RTL (for port interface)
        original_module_name: Original module name

    Returns:
        True if conversion succeeded

    Strategy:
        1. Use ABC write_verilog to get structural Verilog (AND/NOT gates)
        2. Parse ABC output to extract gate logic
        3. Parse original RTL to get exact port interface
        4. Generate new module with original interface + ABC's gate logic
        5. Added instruction ports become internal wires (tied off or optimized away)
    """
    # Check for vmap file (verbose map with better signal preservation)
    map_file = Path(str(aiger_path) + ".vmap")
    if not map_file.exists():
        # Fall back to regular .map file
        map_file = Path(str(aiger_path) + ".map")
        if not map_file.exists():
            print(f"  ERROR: Map file not found: {map_file}")
            print(f"  AIGER was exported without -map/-vmap flag")
            return False

    # Extract original module parameters if provided
    param_declaration = ""
    if original_rtl_path and original_rtl_path.exists():
        with open(original_rtl_path, 'r') as f:
            rtl_content = f.read()
        # Extract parameter declaration from original module
        # Look for "module name #(" and extract until ") ("
        param_match = re.search(
            rf'module\s+{re.escape(original_module_name)}\s*#\s*\((.*?)\)\s*\(',
            rtl_content,
            re.DOTALL
        )
        if param_match:
            params = param_match.group(1).strip()
            # These parameters were used during elaboration but are now constants in gates
            # Add them as unused parameters for interface compatibility
            param_declaration = f"#(\n{params}\n) "
            print(f"  Extracted {params.count('parameter')} parameters from original module")

    # Create Yosys script to read AIGER with map and write Verilog
    ys_script = output_verilog.parent / "aiger_to_verilog.ys"
    ys_content = f"""# Convert AIGER to structural Verilog with signal names restored
# Use map file to restore original signal names
read_aiger -map {map_file} -module_name {original_module_name} -clk_name clk_i -wideports {aiger_path}

# Write Verilog with signal names
write_verilog -noattr -noexpr {output_verilog}
"""

    with open(ys_script, 'w') as f:
        f.write(ys_content)

    try:
        result = subprocess.run(
            ['yosys', '-s', str(ys_script)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"  ERROR: Yosys AIGER conversion failed")
            # Print last 30 lines of error
            error_lines = result.stderr.split('\n')[-30:]
            for line in error_lines:
                if line.strip():
                    print(f"    {line}")
            return False

    except Exception as e:
        print(f"  ERROR: Yosys conversion failed: {e}")
        return False

    # Post-process: Add parameter declaration if needed
    if param_declaration:
        with open(output_verilog, 'r') as f:
            verilog_content = f.read()

        # Replace "module name(" with "module name #(...) ("
        updated_content = verilog_content.replace(
            f"module {original_module_name}(",
            f"module {original_module_name} {param_declaration}(",
            1  # Only replace first occurrence
        )

        with open(output_verilog, 'w') as f:
            f.write(updated_content)

        print(f"  Added parameter declarations for interface compatibility")

    print(f"  Converted AIGER to structural Verilog: {output_verilog}")
    print(f"  Module name: {original_module_name}")
    print(f"  Signal names restored from map file")

    return True


def generate_module_yosys_script(module_config: ModuleConfig,
                                  patched_module: Path,
                                  core_rtl_dir: Path,
                                  output_dir: Path,
                                  data_width: int = 32) -> Path:
    """
    Generate Yosys synthesis script for a single module.

    Args:
        module_config: Module configuration
        patched_module: Path to patched module RTL (with ports and assumptions injected)
        core_rtl_dir: Directory containing core RTL files
        output_dir: Output directory for synthesis products
        data_width: Data width (32 for RV32, 64 for RV64)

    Returns:
        Path to generated Yosys script
    """
    script_path = output_dir / f"{module_config.name}_synth.ys"
    aig_output = output_dir / f"{module_config.name}_optimized.aig"

    # Module name from RTL path
    module_name = Path(module_config.rtl_path).stem

    # Compute additional include paths for vendor code
    vendor_dir = core_rtl_dir.parent / "vendor/lowrisc_ip"
    prim_rtl = vendor_dir / "ip/prim/rtl"

    script_content = f"""# Synlig/Yosys synthesis script for {module_config.name}
# Hierarchical synthesis mode - module has operands as PRIMARY INPUTS
# Data type assumptions are injected directly into the module

# Read all SystemVerilog files in ONE command so packages are available
# Note: Synlig must be in PATH as 'synlig'
# Include package, patched module (with assumptions), and dependencies
read_systemverilog \\
  -I{core_rtl_dir} \\
  -I{prim_rtl} \\
  {prim_rtl}/prim_assert.sv \\
  {core_rtl_dir}/ibex_pkg.sv \\
  {core_rtl_dir}/ibex_alu.sv \\
  {core_rtl_dir}/ibex_multdiv_slow.sv \\
  {core_rtl_dir}/ibex_multdiv_fast.sv \\
  {patched_module}

# Set top module
hierarchy -check -top {module_name}

# Flatten the design to remove hierarchy
flatten

# Do basic optimization
proc
opt

# Map memory and synthesize
memory
techmap
opt

# Convert to AIG with constraints
# Step 1: Handle resets
async2sync
simplemap
dfflegalize -cell $_DFF_P_ 01 -mince 99999
clean

# Step 2: Replace undefined values
setundef -zero

# Step 3: Convert to AIG
aigmap
clean

# Export to AIGER with constraints and signal name mapping
# Data type assumptions will be constraint outputs
# -map writes port/latch mapping for signal name reconstruction
write_aiger -zinit -map {aig_output}.map {aig_output}

# Statistics
stat
"""

    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, 'w') as f:
        f.write(script_content)

    print(f"Generated Yosys script: {script_path}")
    return script_path


def synthesize_module(module_config: ModuleConfig,
                      dsl_rules: List,
                      core_config: CoreConfig,
                      core_rtl_dir: Path,
                      output_dir: Path) -> Path:
    """
    Synthesize a single module with data type constraints.

    Args:
        module_config: Module to synthesize
        dsl_rules: Parsed DSL rules
        core_config: Core configuration
        core_rtl_dir: Directory with core RTL
        output_dir: Output directory

    Returns:
        Path to optimized AIGER file
    """
    print(f"\n{'='*60}")
    print(f"Synthesizing module: {module_config.name}")
    print(f"{'='*60}")

    # Step 1: Patch RTL to add instruction ports
    print(f"\n[1/6] Patching RTL...")
    original_rtl = core_rtl_dir / Path(module_config.rtl_path).name
    patched_rtl = output_dir / f"{module_config.name}_patched.sv"

    # Convert added_ports config to port declarations
    port_declarations = {}
    for port_key, port_name in module_config.added_ports.items():
        if port_key == 'instruction_data':
            port_declarations[port_name] = 'logic [31:0]'
        elif port_key == 'instruction_compressed':
            port_declarations[port_name] = 'logic'
        elif port_key == 'pc':
            port_declarations[port_name] = 'logic [31:0]'

    module_name = Path(module_config.rtl_path).stem
    add_ports_to_module(original_rtl, module_name, port_declarations, patched_rtl)

    # Step 2: Generate and inject data type assumptions directly into module
    print(f"\n[2/6] Generating and injecting data type assumptions...")
    assumptions_code = generate_module_dtype_assertions(
        dsl_rules,
        module_config,
        data_width=core_config.data_width
    )

    if assumptions_code:
        # Add header comment
        injection_code = f"""  // ======================================================================
  // INJECTED: Data type constraint assumptions for hierarchical synthesis
  // These constrain PRIMARY INPUTS to this module
  // ======================================================================
{assumptions_code}"""

        # Inject into the patched module
        patched_with_assumptions = output_dir / f"{module_config.name}_patched_with_assumptions.sv"
        inject_code_into_module(
            patched_rtl,
            module_name,
            injection_code,
            patched_with_assumptions
        )

        # Update patched_rtl to point to version with assumptions
        patched_rtl = patched_with_assumptions
        print(f"  Injected {assumptions_code.count('assume')} assumptions into module")
    else:
        print(f"  No data type constraints to inject")

    # Step 3: Generate Yosys synthesis script
    print(f"\n[3/6] Generating Yosys synthesis script...")
    yosys_script = generate_module_yosys_script(
        module_config,
        patched_rtl,  # Use the patched module with assumptions
        core_rtl_dir,
        output_dir,
        data_width=core_config.data_width
    )

    # Step 4: Run Synlig/Yosys synthesis
    print(f"\n[4/6] Running Synlig/Yosys synthesis...")
    import subprocess
    aig_output = output_dir / f"{module_config.name}_optimized.aig"
    yosys_log = output_dir / f"{module_config.name}_yosys.log"

    try:
        # Use synlig (which includes Yosys with SystemVerilog support)
        result = subprocess.run(
            ['synlig', '-s', str(yosys_script)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        with open(yosys_log, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

        if result.returncode != 0:
            print(f"  ERROR: Synlig failed with return code {result.returncode}")
            print(f"  See log: {yosys_log}")
            return None

        print(f"  Success! Generated: {aig_output}")

    except subprocess.TimeoutExpired:
        print(f"  ERROR: Synlig timed out after 5 minutes")
        return None
    except FileNotFoundError:
        print(f"  ERROR: Synlig not found. Is it installed?")
        print(f"  Install: https://github.com/chipsalliance/synlig")
        return None

    # Step 5: Run ABC optimization with scorr
    print(f"\n[5/6] Running ABC optimization with scorr...")
    abc_output = output_dir / f"{module_config.name}_post_abc.aig"
    abc_log = output_dir / f"{module_config.name}_abc.log"

    abc_commands = f"read_aiger {aig_output}; strash; scorr -k 2; dc2; dretime; write_aiger {abc_output}"

    try:
        result = subprocess.run(
            ['abc', '-c', abc_commands],
            capture_output=True,
            text=True,
            timeout=300
        )

        with open(abc_log, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

        # Check for "init state does not satisfy constraints" error
        if "init state does not satisfy" in result.stdout.lower():
            print(f"  WARNING: ABC scorr reported init state constraint violation!")
            print(f"  This means data type constraints are being removed by ABC.")
            print(f"  See log: {abc_log}")
        elif result.returncode == 0:
            print(f"  Success! ABC scorr completed without init state errors")
            print(f"  Optimized AIGER: {abc_output}")
        else:
            print(f"  ERROR: ABC failed with return code {result.returncode}")
            print(f"  See log: {abc_log}")

    except subprocess.TimeoutExpired:
        print(f"  ERROR: ABC timed out after 5 minutes")
        return None
    except FileNotFoundError:
        print(f"  ERROR: ABC not found. Is it installed?")
        return None

    # Step 6: Convert ABC-optimized AIGER to RTLIL for composition
    print(f"\n[6/6] Converting optimized AIGER to RTLIL...")
    rtlil_output = output_dir / f"{module_config.name}_optimized.il"

    success = convert_aiger_to_rtlil(
        abc_output,
        rtlil_output,
        module_name
    )

    if not success:
        print(f"  ERROR: Failed to convert AIGER to RTLIL")
        return None

    print(f"\n✓ Module optimization complete")
    print(f"  Optimized AIGER: {abc_output}")
    print(f"  RTLIL for composition: {rtlil_output}")

    return rtlil_output


def synthesize_full_core(core_config: CoreConfig,
                         optimized_modules: Dict[str, Path],
                         dsl_rules: List,
                         core_rtl_dir: Path,
                         output_dir: Path) -> Optional[Path]:
    """
    Synthesize the full core with optimized submodules integrated.

    Args:
        core_config: Core configuration
        optimized_modules: Dict mapping module names to optimized RTLIL paths
        dsl_rules: Parsed DSL rules (for PI-level constraints)
        core_rtl_dir: Directory with core RTL
        output_dir: Output directory

    Returns:
        Path to final optimized AIGER, or None if failed
    """
    print(f"\n{'='*70}")
    print(f"SYNTHESIZING FULL CORE")
    print(f"{'='*70}")

    # Step 1: Patch ibex_core.sv to wire instruction signals to optimized modules
    print(f"\n[1/4] Patching ibex_core to wire instruction signals...")

    original_core = core_rtl_dir.parent / "rtl" / "ibex_core.sv"
    patched_core = output_dir / "ibex_core_patched.sv"

    # For ex_block, wire instruction signals
    if 'ex_block' in optimized_modules:
        signals_to_wire = {
            'instr_rdata_i': 'instr_rdata_id',
            'instr_is_compressed_i': '(instr_rdata_id[1:0] != 2\'b11)'
        }

        patch_parent_module(
            original_core,
            'ex_block_i',
            signals_to_wire,
            patched_core
        )
    else:
        # Just copy if no modules to patch
        import shutil
        shutil.copy(original_core, patched_core)

    # Step 2: Generate PI-level constraints (instruction patterns, PC, registers only)
    print(f"\n[2/4] Generating PI-level constraints...")
    # TODO: Implement generate_core_level_constraints()
    # For now, use existing generate_inline_assumptions() without instruction_rules
    from .codegen import generate_inline_assumptions
    from .parser import RequireRule, RegisterConstraintRule, PcConstraintRule, InstructionRule, PatternRule

    # Separate rules into PI-level only (no dtype constraints)
    required_extensions = set()
    register_constraint = None
    pc_constraint = None
    patterns = []

    for rule in dsl_rules:
        if isinstance(rule, RequireRule):
            required_extensions.add(rule.extension)
        elif isinstance(rule, RegisterConstraintRule):
            register_constraint = rule
        elif isinstance(rule, PcConstraintRule):
            pc_constraint = rule
        elif isinstance(rule, InstructionRule):
            # Skip instruction rules with dtype constraints - those are already in ex_block gates
            has_dtype = any(c.field_name == 'dtype' or c.field_name.endswith('_dtype')
                          for c in rule.constraints)
            if not has_dtype:
                # Include instruction constraints without dtype
                from .codegen import instruction_rule_to_pattern, has_c_extension_required
                has_c_ext = has_c_extension_required(dsl_rules)
                rule_patterns = instruction_rule_to_pattern(rule, has_c_ext)
                patterns.extend(rule_patterns)
        elif isinstance(rule, PatternRule):
            desc = rule.description if rule.description else f"Pattern"
            patterns.append((rule.pattern, rule.mask, desc, False))

    pc_bits = pc_constraint.pc_bits if pc_constraint else None

    core_constraints = generate_inline_assumptions(
        patterns, required_extensions, register_constraint, pc_bits, core_config, None)

    constraints_file = output_dir / "core_pi_constraints.sv"
    with open(constraints_file, 'w') as f:
        f.write(core_constraints)

    print(f"  Generated PI-level constraints: {constraints_file}")

    # Step 3: Generate Synlig script for full core
    print(f"\n[3/4] Generating synthesis script for full core...")

    # Use RTL configuration from config file
    rtl_root = Path(core_config.rtl.root_dir)
    if not rtl_root.is_absolute():
        # Make relative paths relative to config file location
        rtl_root = core_rtl_dir.parent

    # Build include directory list
    include_dirs = []
    for inc_dir in core_config.rtl.include_dirs:
        full_path = rtl_root / inc_dir
        include_dirs.append(str(full_path))

    # Collect all RTL files (EXCLUDING optimized modules)
    all_rtl_files = []

    # Add support files (packages, primitives)
    for support_file in core_config.rtl.support_files:
        full_path = rtl_root / support_file
        all_rtl_files.append(str(full_path))

    # Add ALL core files (including modules to be replaced)
    # We'll delete and replace them AFTER hierarchy is established
    for core_file in core_config.rtl.core_files:
        full_path = rtl_root / core_file
        all_rtl_files.append(str(full_path))

    # Add patched core (replaces original ibex_core.sv)
    all_rtl_files.append(str(patched_core))

    # Generate synthesis script
    synth_script = output_dir / "full_core_synth.ys"
    aig_output = output_dir / "ibex_core_full.aig"

    rtl_list = " \\\n  ".join(all_rtl_files)
    include_flags = " \\\n  ".join([f"-I{d}" for d in include_dirs])

    # Build delete+read_rtlil commands for optimized modules
    rtlil_composition = ""
    for module_name, rtlil_path in optimized_modules.items():
        rtlil_composition += f"""
# Replace original {module_name} with optimized RTLIL version
delete ibex_{module_name}
read_rtlil -nooverwrite {rtlil_path}
"""

    script_content = f"""# Full core synthesis with RTLIL-based optimized submodules
# Strategy: Industry-standard hierarchical synthesis approach
# 1. Read all RTL and synthesize
# 2. Delete original submodules, replace with optimized RTLIL
# 3. Flatten and export to AIGER with PI constraints

# Step 1: Read all RTL (including original versions of modules to be replaced)
read_systemverilog \\
  {include_flags} \\
  {rtl_list}

# Step 2: Set hierarchy and parameters
hierarchy -check -top ibex_core
chparam -set WritebackStage 0 ibex_core
chparam -set BranchTargetALU 0 ibex_core

# Step 3: Synthesize to gate level (before composition)
proc
opt
memory
techmap
opt
aigmap
clean

# Step 4: Replace original modules with optimized RTLIL versions
{rtlil_composition}

# Step 5: Flatten entire design (merges optimized modules + rest)
flatten

# Basic optimization
proc
opt

# Map and synthesize
memory
techmap
opt

# Convert to AIG
async2sync
simplemap
dfflegalize -cell $_DFF_P_ 01 -mince 99999
clean
setundef -zero
aigmap
clean

# Export to AIGER
write_aiger -zinit -map {aig_output}.map {aig_output}

stat
"""

    with open(synth_script, 'w') as f:
        f.write(script_content)

    print(f"  Generated synthesis script: {synth_script}")

    # Step 4: Run full core synthesis
    print(f"\n[4/4] Running full core synthesis...")
    yosys_log = output_dir / "full_core_yosys.log"

    try:
        result = subprocess.run(
            ['synlig', '-s', str(synth_script)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes for full core
        )

        with open(yosys_log, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

        if result.returncode != 0:
            print(f"  ERROR: Full core synthesis failed")
            print(f"  See log: {yosys_log}")
            return None

        print(f"  Success! Generated full core AIGER: {aig_output}")

    except subprocess.TimeoutExpired:
        print(f"  ERROR: Synthesis timed out")
        return None
    except FileNotFoundError:
        print(f"  ERROR: Synlig not found")
        return None

    # Step 5: Run ABC on full core
    print(f"\nRunning ABC scorr on full core...")
    abc_output = output_dir / "ibex_core_full_post_abc.aig"
    abc_log = output_dir / "full_core_abc.log"

    abc_commands = f"read_aiger {aig_output}; strash; scorr -k 2; dc2; dretime; write_aiger {abc_output}"

    try:
        result = subprocess.run(
            ['abc', '-c', abc_commands],
            capture_output=True,
            text=True,
            timeout=600
        )

        with open(abc_log, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

        if result.returncode == 0:
            print(f"  Success! Optimized full core: {abc_output}")
        else:
            print(f"  ERROR: ABC failed on full core")
            print(f"  See log: {abc_log}")
            return None

    except Exception as e:
        print(f"  ERROR: ABC failed: {e}")
        return None

    return abc_output


def hierarchical_synthesis(dsl_path: Path,
                           config_path: Path,
                           output_dir: Path,
                           core_rtl_dir: Optional[Path] = None) -> bool:
    """
    Main hierarchical synthesis orchestrator.

    Args:
        dsl_path: Path to DSL file with constraints
        config_path: Path to core config YAML with modules section
        output_dir: Output directory for all synthesis products
        core_rtl_dir: Directory containing core RTL (auto-detected if None)

    Returns:
        True if synthesis succeeded, False otherwise
    """
    print(f"{'='*70}")
    print(f"HIERARCHICAL SYNTHESIS")
    print(f"{'='*70}")
    print(f"DSL file:    {dsl_path}")
    print(f"Config:      {config_path}")
    print(f"Output dir:  {output_dir}")

    # Load configuration
    config = CoreConfig.from_yaml(config_path)
    print(f"\nCore: {config.core_name} ({config.architecture})")
    print(f"Modules for separate synthesis: {len([m for m in config.modules if m.synthesize_separately])}")

    # Parse DSL
    print(f"\nParsing DSL...")
    with open(dsl_path, 'r') as f:
        dsl_content = f.read()

    try:
        ast = parse_dsl(dsl_content)
    except SyntaxError as e:
        print(f"ERROR: DSL syntax error: {e}")
        return False

    print(f"  Parsed {len(ast.rules)} rules")

    # Auto-detect core RTL directory if not provided
    if core_rtl_dir is None:
        # Try to find PdatCoreSim relative to config location
        config_dir = config_path.parent
        potential_paths = [
            config_dir / "../PdatCoreSim/cores" / config.core_name,
            config_dir / "../../PdatCoreSim/cores" / config.core_name,
            Path(f"../PdatCoreSim/cores/{config.core_name}"),
        ]
        for path in potential_paths:
            rtl_path = path / "rtl"
            if rtl_path.exists():
                core_rtl_dir = rtl_path
                print(f"  Auto-detected RTL dir: {core_rtl_dir}")
                break

        if core_rtl_dir is None:
            print(f"ERROR: Could not auto-detect core RTL directory")
            print(f"  Please specify --rtl-dir")
            return False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Synthesize each module separately
    optimized_modules = {}
    for module_config in config.modules:
        if not module_config.synthesize_separately:
            continue

        result = synthesize_module(
            module_config,
            ast.rules,
            config,
            core_rtl_dir,
            output_dir / module_config.name
        )

        if result is None:
            print(f"\nERROR: Failed to synthesize {module_config.name}")
            return False

        optimized_modules[module_config.name] = result
        print(f"\n✓ Module '{module_config.name}' optimized: {result}")

    # Phase 2: Synthesize full core with optimized modules
    if optimized_modules:
        full_core_result = synthesize_full_core(
            config,
            optimized_modules,
            ast.rules,
            core_rtl_dir,
            output_dir / "full_core"
        )

        if full_core_result is None:
            print(f"\nERROR: Failed to synthesize full core")
            return False

        print(f"\n{'='*70}")
        print(f"HIERARCHICAL SYNTHESIS COMPLETE")
        print(f"{'='*70}")
        print(f"\nFinal optimized core: {full_core_result}")
        print(f"Submodules optimized: {', '.join(optimized_modules.keys())}")

        return True
    else:
        print(f"\n{'='*70}")
        print(f"NO MODULES TO OPTIMIZE")
        print(f"{'='*70}")
        return False


def main():
    """CLI entry point for hierarchical synthesis."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Hierarchical synthesis with data type constraints'
    )
    parser.add_argument('dsl_file', type=Path,
                        help='DSL file with constraints')
    parser.add_argument('--config', type=Path, required=True,
                        help='Core configuration YAML with modules section')
    parser.add_argument('--output', type=Path, default=Path('output/hierarchical'),
                        help='Output directory (default: output/hierarchical)')
    parser.add_argument('--rtl-dir', type=Path,
                        help='Core RTL directory (auto-detected if not specified)')

    args = parser.parse_args()

    success = hierarchical_synthesis(
        args.dsl_file,
        args.config,
        args.output,
        args.rtl_dir
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
