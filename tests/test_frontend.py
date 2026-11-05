"""Acceptance and rejection tests for the legacy-preserving typed frontend."""
import json
from pathlib import Path

import pytest

from pdat_dsl.config import CoreConfig
from pdat_dsl.frontend import FrontendError, elaborate_frontend
from pdat_dsl.parser import DataTypeSet, InstructionRule

BASE = """crisp frontend 1
context decode = isa
context ex = alu
context mul = multdiv
context bus = memory
"""


def compile_body(body, config=None):
    return elaborate_frontend(BASE + body, config or CoreConfig.default_ibex())


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("expression,target", [
    ("fits_u(value, 8)", "~u8"),
    ("value < 256", "~u8"),
    ("(value & bv(255, W)) == value", "~u8"),
    ("fits_s(value, 8)", "~i8"),
    ("fits_s(value, 16) or fits_u(value, 16)", "~(i8 | u8 | i16 | u16)"),
    ("not fits_u(value, 8)", "u8"),
    ("fits_u(value + bv(0, W), 8)", "~u8"),
])
def test_exact_domains_and_bitvector_expressions(width, expression, target):
    config = CoreConfig.default_ibex()
    config.architecture = f"rv{width}"
    result = compile_body(f"operand ADD.rs1 at ex satisfies {expression.replace('W', str(width))} lowers {target}\n", config)
    rule = result.program.rules[0]
    assert isinstance(rule, InstructionRule)
    assert isinstance(rule.constraints[0].field_value, DataTypeSet)
    proof = result.manifest["exactness_checks"][0]
    assert proof["source_minus_legacy"] == proof["legacy_minus_source"] == "unsat"
    assert proof["carrier_width"] == width
    json.dumps(result.manifest)


@pytest.mark.parametrize("expression,direction,number", [("value <= 256", "lost", 256), ("value < 255", "extra", 255)])
def test_both_exactness_directions_have_concrete_counterexamples(expression, direction, number):
    with pytest.raises(FrontendError, match=f"{direction} value") as error:
        compile_body(f"operand ADD.rs1 at ex satisfies {expression} lowers ~u8\n")
    assert error.value.witness["direction"] == direction
    assert error.value.witness["unsigned"] == number
    assert error.value.witness["source_accepts"] != error.value.witness["legacy_accepts"]


def test_generic_nested_predicates_and_bundles_preserve_rule_order_and_roles():
    result = compile_body("""predicate within(x: bits[W], n: nat) = fits_u(x,n)
predicate both(x: bits[W], n: nat) = within(x,n) or fits_s(x,n)
constraint one(op: instruction, n: nat) at c: alu:
  operand $op.inputs at $c satisfies both(value,$n) lowers ~(i8|u8|i$n|u$n)
end
constraint two(n: nat) at c: alu:
  use one(ADD,$n) at $c
  use one(SUB,$n) at $c
end
at decode: require RV32I
use two(16) at ex
at bus: data_miss_latency 4
""")
    assert result.legacy_text == "version 1\nrequire RV32I\ninstruction ADD { dtype = ~(i8 | u8 | i16 | u16) }\ninstruction SUB { dtype = ~(i8 | u8 | i16 | u16) }\ndata_miss_latency 4\n"
    rules = result.manifest["rules"]
    assert [r["role"] for r in rules] == ["instruction_assumption", "operand_assumption", "operand_assumption", "timing_assumption"]
    assert len(rules[1]["expansion"]) == 2
    assert result.manifest["exactness_checks"][0]["operands"] == ["rs1", "rs2"]


def test_context_records_config_without_adding_a_sampling_guard():
    config = CoreConfig.default_ibex()
    config.signals.multdiv_rs1 = "custom_multiplier_input"
    result = compile_body("operand MUL.rs1 at mul satisfies fits_u(value,8) lowers ~u8\n", config)
    assert result.manifest["exactness_checks"][0]["bindings"] == {"rs1": "custom_multiplier_input"}
    assert result.manifest["contexts"]["mul"]["sampling"] == "legacy combinational assumptions"
    assert result.legacy_text == "version 1\ninstruction MUL { rs1_dtype = ~u8 }\n"


@pytest.mark.parametrize("body,diagnostic", [
    ("operand ADD.rs1 at mul satisfies fits_u(value,8) lowers ~u8", "requires alu"),
    ("operand MUL.rs1 at ex satisfies fits_u(value,8) lowers ~u8", "requires multdiv"),
    ("operand ADD.rd at ex satisfies fits_u(value,8) lowers ~u8", "output operand"),
    ("operand LUI.inputs at ex satisfies fits_u(value,8) lowers ~u8", "no requested input"),
    ("operand ADDI.rs2 at ex satisfies fits_u(value,8) lowers ~u8", "no requested input"),
    ("operand ADD.rs1 at missing satisfies fits_u(value,8) lowers ~u8", "unknown context"),
    ("operand ADD.rs1 at ex satisfies value lowers ~u8", "return bool"),
    ("operand ADD.rs1 at ex satisfies fits_u(value,33) lowers ~u8", "within the carrier width"),
    ("operand ADD.rs1 at ex satisfies value == bv(1,8) lowers ~u8", "width mismatch"),
    ("operand ADD.rs1 at ex satisfies fits_u(value,16) lowers ~u16", "requires allowing all narrower"),
    ("operand ADD.rs1 at ex satisfies fits_u(value,32) lowers ~(u8|u16|u32)", "strictly below carrier"),
    ("operand ADD.rs1 at ex satisfies fits_u(value,8) lowers ~u8, rd=x0", "exactly one legacy"),
    ("operand ADD.rs1 at ex satisfies fits_u(value,8) lowers u8", "inexact lowering"),
    ("backend 2\noperand ADD.rs1 at ex satisfies fits_u(value,8) lowers ~u8", "backend 2 drops dtype"),
    ("at decode: instruction ADD { dtype = ~u8 }", "must use 'operand"),
    ("at decode: instruction ADD { rd_dtype = ~u8 }", "must use 'operand"),
    ("at ex: require RV32I", "requires isa"),
    ("at decode: data_miss_latency 4", "requires memory"),
    ("at bus: locality_bits 12", "ignores locality_bits"),
    ("at bus: data_miss_latency 8", "3 bits"),
    ("at bus: data_hit_latency 2\nat bus: data_hit_latency 1", "overwrite an earlier"),
    ("at decode: require_registers x0-x40", "out of range"),
    ("at decode: require_pc_bits 33", "out of range"),
    ("at decode: instruction ADD { rs1 = x32 }", "outside 5-bit"),
    ("at decode: instruction SLLI { shamt = 6'b000001 }", "width 5"),
    ("at decode: allow instruction ADD", "unsupported"),
    ("at decode: pattern 0 mask 0", "source-line-dependent"),
    ("backend 2\nat decode: require_registers x0-x15", "ignores require_registers"),
    ("backend 2\nat decode: include ADD {rs1=x0}", "constrained inclusion/exclusion"),
    ("backend 2\nat decode: forbid ADD {rs1=x0,rs2=x0}", "constrained inclusion/exclusion"),
    ("backend 2\nat decode: include ADD\nat decode: forbid ADD\nat decode: include ADD", "repeated inclusion"),
    ("context late = alu on posedge clk_i when valid", "custom clocks/guards/outputs"),
    ("event ex = execute", "custom clocks/guards/outputs"),
])
def test_rejected_unsupported_or_inexact_source(body, diagnostic):
    with pytest.raises(FrontendError, match=diagnostic):
        compile_body(body + "\n")


@pytest.mark.parametrize("definitions,call,diagnostic", [
    ("predicate narrow(x: bits[8]) = fits_u(x,8)", "narrow(value)", "inconsistent width"),
    ("predicate narrow(x: bool) = x", "narrow(value)", "expects bool"),
    ("predicate narrow(x: bits[W], n: nat) = fits_u(x,n)", "narrow(value,value)", "nonnegative elaboration"),
    ("predicate narrow(x: bits[W]) = fits_u(value,8)", "narrow(value)", "free name"),
    ("predicate narrow(x: bits[W]) = narrow(x)", "narrow(value)", "recursive predicate"),
    ("predicate narrow(x: bits[W]) = True", "narrow(value)", "inexact lowering"),
    ("predicate narrow(x: bits[W]) = fits_u(x,8)", "narrow(value,8)", "expects 1"),
])
def test_predicates_are_hygienic_and_checked_at_instantiation(definitions, call, diagnostic):
    with pytest.raises(FrontendError, match=diagnostic):
        compile_body(definitions + f"\noperand ADD.rs1 at ex satisfies {call} lowers ~u8\n")


@pytest.mark.parametrize("body,diagnostic", [
    ("constraint a(op: instruction) at c: alu:\n  operand $op.rs1 at ex satisfies fits_u(value,8) lowers ~u8\nend", "explicit context parameter"),
    ("constraint a(op: instruction) at c: alu:\n  operand $missing.rs1 at $c satisfies fits_u(value,8) lowers ~u8\nend", "unbound constraint parameter"),
    ("constraint a() at c: alu:\n  use a() at $c\nend", "recursive constraint"),
    ("constraint a(op: instruction) at c: alu:\n  operand $op.rs1 at $c satisfies fits_u(value,8) lowers ~u8\nend\nuse a(8) at ex", "unknown instruction"),
    ("constraint a(n: nat) at c: memory:\n  at $c: data_miss_latency $n\nend\nuse a(ADD) at bus", "expects nat"),
    ("constraint a(r: registers) at c: isa:\n  at $c: require_registers $r\nend\nuse a(x0-x99) at decode", "expects a register range"),
    ("constraint a(n: nat) at c: memory:\n  at $c: data_miss_latency $n\nend\nuse a(4) at ex", "requires memory"),
])
def test_bundle_parameters_and_context_cannot_be_captured_or_confused(body, diagnostic):
    with pytest.raises(FrontendError, match=diagnostic):
        compile_body(body + "\n")


def test_missing_config_binding_rejected():
    config = CoreConfig.default_ibex()
    config.signals.alu_rs1 = ""
    with pytest.raises(FrontendError, match="configured signal alu_rs1"):
        compile_body("", config)


def test_imports_are_definition_only_deduplicated_and_recorded(tmp_path):
    library = tmp_path / "narrow.crisplib"
    library.write_text("""crisp library 1
predicate narrow(x: bits[W], n: nat) = fits_u(x,n)
constraint narrow_one(op: instruction) at c: alu:
  operand $op.rs1 at $c satisfies narrow(value,8) lowers ~u8
end
""")
    result = elaborate_frontend(BASE + 'import "narrow.crisplib"\nimport "narrow.crisplib"\nuse narrow_one(ADD) at ex\n', CoreConfig.default_ibex(), tmp_path)
    assert len(result.manifest["imports"]) == 1
    assert len(result.manifest["imports"][0]["sha256"]) == 64
    assert result.manifest["rules"][0]["source"] == str(library)
    library.write_text("crisp library 1\ncontext hidden = alu\n")
    with pytest.raises(FrontendError, match="definitions and imports only"):
        elaborate_frontend(BASE + 'import "narrow.crisplib"\n', CoreConfig.default_ibex(), tmp_path)


def test_import_cycle_rejected(tmp_path):
    (tmp_path / "a").write_text('crisp library 1\nimport "b"\n')
    (tmp_path / "b").write_text('crisp library 1\nimport "a"\n')
    with pytest.raises(FrontendError, match="cyclic library import"):
        elaborate_frontend(BASE + 'import "a"\n', CoreConfig.default_ibex(), tmp_path)


def test_solver_unknown_fails_closed(monkeypatch):
    import z3
    class Inconclusive:
        def set(self, **kwargs): pass
        def add(self, *args): pass
        def check(self): return z3.unknown
        def reason_unknown(self): return "test timeout"
    monkeypatch.setattr(z3, "Solver", Inconclusive)
    with pytest.raises(FrontendError, match="inconclusive.*test timeout"):
        compile_body("operand ADD.rs1 at ex satisfies fits_u(value,8) lowers ~u8\n")


def test_all_paired_examples_elaborate():
    directory = Path(__file__).parents[1] / "examples" / "frontend"
    for path in directory.glob("*.crisp"):
        result = elaborate_frontend(path.read_text(), CoreConfig.default_ibex(), path.parent)
        assert path.with_suffix(".dsl").is_file()
        assert result.manifest["rules"]


@pytest.mark.parametrize("body,diagnostic", [
    ("at decode: require_registers x3-x15", "ignores register lower bounds"),
    ("at decode: require_pc_bits 32", "strictly below"),
    ("at decode: instruction ADD {rs1 in x0-x3}", "field-range lowering"),
    ("at decode: instruction ADD {opcode=0}", "overlaps fixed"),
    ("at decode: require RV32I\nat decode: instruction SLLI {shamt=5'b1xxxx}", "broadens field exclusions"),
    ("backend 2\nat decode: include RV32I\nat decode: forbid SLLI {shamt=5'b1xxxx}", "ignores all constrained"),
    ("backend 2\nat decode: include RV32I\nat decode: forbid x8-x15", "one suffix"),
    ("backend 2\nat decode: include RV32I\nat decode: forbid x8-x31\nat decode: forbid x16-x31", "overwrite an earlier"),
    ("backend 2\nat decode: include RV32C", "bypasses compressed"),
    ("backend 2\nat decode: include ADD\nat decode: forbid ADD", "empty instruction set"),
])
def test_audited_backend_coverage_failures_are_rejected(body, diagnostic):
    with pytest.raises(FrontendError, match=diagnostic):
        compile_body(body + "\n")


@pytest.mark.parametrize("require", [False, True])
def test_dtype_legacy_instruction_exclusion_is_exposed_not_claimed_fixed(require):
    from pdat_dsl.codegen import generate_inline_assumptions, instruction_rule_to_pattern
    config = CoreConfig.default_ibex()
    source = ("at decode: require RV32I\n" if require else "")
    result = compile_body(source + "operand ADD.rs1 at ex satisfies fits_u(value,8) lowers ~u8\n", config)
    rule = result.program.rules[-1]
    patterns = instruction_rule_to_pattern(rule)
    code = generate_inline_assumptions(patterns, {"RV32I"} if require else set(), config=config, instruction_rules=[rule])
    assert patterns[0][:2] == (0x00000033, 0xfe00707f)
    assert "// ADD rs1: allow only ~u8" in code
    if require:
        assert "// Excluding outlawed: ADD" in code
        assert "// ADD (RV32I)" not in code
        assert "// ADDI (RV32I)" in code
    else:
        assert "// 32-bit instruction outlawed patterns" in code
        assert "Pattern=0x00000033, Mask=0xfe00707f" in code
    assert result.manifest["legacy_effects"][0]["kind"] == "dtype_instruction_exclusion"
    assert "also outlaws instruction ADD" in result.manifest["warnings"][0]
    assert "local dtype predicate only" in result.manifest["exactness_checks"][0]["scope"]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("target", ["~u8", "~i8", "~(i8|u8|i16|u16)", "u8", "i8", "i8|u8|i16|u16"])
def test_actual_legacy_dtype_sv_has_exact_mathematical_domain(width, target, tmp_path):
    """Yosys proves the emitted local SV equality for every carrier value."""
    import shutil
    import subprocess
    from pdat_dsl.dtype_codegen import generate_dtype_set_check_expr
    from pdat_dsl.parser import parse_dsl
    if shutil.which("yosys") is None:
        pytest.skip("Yosys is needed for the independent generated-SV proof")
    dtype = parse_dsl(f"instruction ADD {{ rs1_dtype = {target} }}\n").rules[0].constraints[0].field_value
    emitted = generate_dtype_set_check_expr(dtype, "x", width)
    if not dtype.negated:
        emitted = f"!({emitted})"
    intervals = []
    for member in dtype.types:
        bound = 1 << (member.width - int(member.signed))
        interval = f"(x < {width}'d{bound})"
        if member.signed:
            interval = f"({interval} || (x >= {width}'d{(1 << width) - bound}))"
        intervals.append(interval)
    reference = "(" + " || ".join(intervals) + ")"
    if not dtype.negated:
        reference = "!" + reference
    (tmp_path / "m.v").write_text(f"module m(input [{width-1}:0] x, output ok);\nassign ok = ({emitted}) == ({reference});\nendmodule\n")
    command = ["yosys", "-Q", "-T", "-p", "read_verilog m.v; prep -top m; sat -verify -prove ok 1 -show x"]
    proof = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=20)
    assert proof.returncode == 0, proof.stdout + proof.stderr
    assert "SUCCESS" in proof.stdout
