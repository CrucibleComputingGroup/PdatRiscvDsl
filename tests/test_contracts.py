"""Semantic tests for CRISP 3, including independently specified HDL miters.

These tests validate the language/front end. They do not prove full processor
equivalence, infer application contracts, or reproduce historical PPA results.
"""
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import z3

from pdat_dsl.typed_core import (ContractError, Elaborator, Predicate, SVEmitter,
                                signal, evaluate, to_smt)
from pdat_dsl.typed_isa import ISARegistry
from pdat_dsl.contracts import (parse_contract, emit_contract, audit_trace,
                                local_witnesses, compare_observations,
                                validate_rtl_bindings)

ROOT = Path(__file__).resolve().parents[1]
HEADER = """crisp 3
clock core = clk_i reset rst_ni active_low
event ex on core scope execute when valid
signal valid : bool @ ex = valid_i
signal x : bits[32] @ ex = x_i
signal y : bits[32] @ ex = y_i
"""


def prove_equal(left, right):
    solver = z3.Solver()
    solver.set(timeout=10000)
    solver.add(left != right)
    assert solver.check() == z3.unsat, solver.model() if solver.check() == z3.sat else "unknown"


def run(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


@pytest.mark.parametrize("source,message", [
    ("assume a at ex: x == bv(0, 16)", "width mismatch"),
    ("assume a at ex: x == 4294967296", "outside"),
    ("assume a at ex: x and valid", "require bool"),
    ("assume a at ex: x", "proposition"),
    ("assume a at ex: fits_u(x, 33)", "refinement width"),
    ("assume a at ex: slice(x, 32, 0) == 0", "invalid slice"),
    ("assume a at ex: zext(x, 16) == 0", "nondecreasing"),
    ("assume a at ex: missing == 0", "unknown name"),
    ("assume a at ex: x.value == 0", "unsupported"),
    ("assume a at ex: __import__('os')", "unknown predicate"),
    ("assume a at ex: fits_u(x, x)", "elaboration-time"),
    ("predicate p(a: bits[W]) = x == 0\nassume a at ex: p(x)", "free name"),
    ("predicate p(a: bits[W]) = q(a)\npredicate q(a: bits[W]) = p(a)", "recursive"),
    ("predicate p(a: bits[W], b: bits[W]) = a == b\nassume a at ex: p(x, bv(0, 16))", "inconsistent width"),
    ("assume response r at ex: valid -> valid within 3..1", "response interval"),
    ("assume response r at ex: valid -> valid within 1..65", "response interval"),
    ("candidate cut at ex: x -> bv(0, 16)", "replacement"),
    ("candidate cut at ex: x -> bv(0, 32)", "observation interface"),
])
def test_rejects_invalid_contracts(source, message):
    with pytest.raises(ContractError, match=message): parse_contract(HEADER+source+"\n")


def test_event_scope_cannot_be_lifted_or_mixed():
    source = HEADER + """event retirement on core scope retire when committed
signal committed : bool @ retirement = retired_i
signal retired_x : bits[32] @ retirement = retired_x_i
"""
    with pytest.raises(ContractError, match="not implicitly lifted"):
        parse_contract(source + "assume partition at ex: retired_x == 0\n")
    with pytest.raises(ContractError, match="cannot combine sampling events"):
        parse_contract(source + "assert mapping at ex: retired_x == x\n")


@pytest.mark.parametrize("statement", [
    "include ADD {absent = 0}", "include ADD {rd = x32}",
    "include SLLI {shamt = 6'b000000}", "include ADD {rd = x0, rd = x1}",
    "include RV32C", "include UNKNOWN", "forbid UNKNOWN",
    "include ADD {dtype = i8}", "include ADD {imm = 1}",
    "require_pc_bits 16", "instr_hit_latency 1",
])
def test_isa_rejects_unsupported_or_malformed_constructs(statement):
    with pytest.raises(ContractError):
        ISARegistry().compile("version 2\n"+statement+"\n", signal("instr", 32, "ex"))


@pytest.mark.parametrize("policy,golden", [
    ("include ADD {rd = x0}", lambda w: z3.And(w & 0xfe00707f == 0x33, z3.Extract(11, 7, w) == 0)),
    ("include ADD\nforbid x16-x31\ninclude ADD", lambda w: w & 0xfe00707f == 0x33),
    ("include ADD\nforbid ADD {rd = x3, rs1 = x7}", lambda w: z3.And(w & 0xfe00707f == 0x33, z3.Not(z3.And(z3.Extract(11, 7, w) == 3, z3.Extract(19, 15, w) == 7)))),
    ("include ADD {rd in x3-x11}", lambda w: z3.And(w & 0xfe00707f == 0x33, z3.UGE(z3.Extract(11, 7, w), 3), z3.ULE(z3.Extract(11, 7, w), 11))),
    ("include SLLI {shamt = 5'b000xx}", lambda w: w & 0xffc0707f == 0x1013),
    ("include ADD\nforbid ADD", lambda w: z3.BoolVal(False)),
    ("include ADD {rd = x0}\ninclude ADD {rd = x1}", lambda w: z3.And(w & 0xfe00707f == 0x33, z3.ULE(z3.Extract(11, 7, w), 1))),
])
def test_exact_isa_semantics_over_all_32_bit_words(policy, golden):
    expr = ISARegistry().compile("version 2\n"+policy+"\n", signal("instr", 32, "ex"))
    prove_equal(to_smt(expr), golden(z3.BitVec("instr", 32)))


def test_all_seven_existing_v2_example_isa_fragments():
    # Configuration directives are migrated explicitly, not silently discarded
    # by the compiler. This test isolates the ISA fragment for its own check.
    paths = sorted((ROOT / "examples").glob("v2_*.dsl"))
    assert len(paths) == 7
    for path in paths:
        source = "\n".join(line for line in path.read_text().splitlines()
                           if not line.startswith(("instr_hit_latency", "instr_miss_latency", "data_hit_latency", "data_miss_latency")))
        expr = ISARegistry().compile(source, signal("instr", 32, "ex"))
        solver = z3.Solver(); solver.add(to_smt(expr)); solver.set(timeout=3000)
        assert solver.check() == z3.sat, path


def test_library_extension_and_refinement_subtyping():
    predicates = {
        "aligned": Predicate((("x", "bits[W]"), ("n", "nat")), "slice(x, n-1, 0) == 0"),
        "region": Predicate((("x", "bits[W]"),), "slice(x, 21, 16) == 0"),
    }
    elab = Elaborator({"x": signal("x", 32, "ex")}, predicates)
    x = z3.BitVec("x", 32)
    prove_equal(to_smt(elab.parse("aligned(x, 4)")), x & 15 == 0)
    prove_equal(to_smt(elab.parse("region(x)")), z3.Extract(21, 16, x) == 0)
    prove_equal(to_smt(elab.parse("fits_u(x, 8) or fits_u(x, 16)")), z3.ULT(x, 65536))
    prove_equal(to_smt(elab.parse("fits_s(x, 16)")), z3.And(x >= -32768, x <= 32767))
    prove_equal(to_smt(elab.parse("fits_s(x, 32)")), z3.BoolVal(True))


def test_exhaustive_small_bitvector_interpreter():
    elab = Elaborator({"x": signal("x", 4, "ex"), "y": signal("y", 4, "ex")})
    exprs = [elab.parse(e) for e in ("x+y", "x-y", "x*y", "slt(x,y)", "x<y", "fits_s(x,2)", "fits_u(x,2)")]
    signed = lambda v: v if v < 8 else v-16
    for x, y in itertools.product(range(16), repeat=2):
        assert [evaluate(e, {"x": x, "y": y}) for e in exprs] == [
            (x+y) % 16, (x-y) % 16, (x*y) % 16, signed(x)<signed(y), x<y, -2 <= signed(x) <= 1, x < 4]


@pytest.mark.parametrize("expression,golden", [
    ("fits_s(x, 16)", "x[31:16] == {16{x[15]}}"),
    ("fits_u(x, 8)", "x[31:8] == 24'b0"),
    ("slice(x + y, 7, 0) == 0", "((x+y) & 32'hff) == 0"),
    ("slt(x,y)", "$signed(x) < $signed(y)"),
    ("sext(slice(x,7,0),32) == y", "{{24{x[7]}},x[7:0]} == y"),
    ("ite(x == 0, x, y) == y", "((x == 0) ? x : y) == y"),
    ("slice(bv(128, 8), 7, 0) == 128", "1'b1"),
    ("sext(bv(128, 8), 32) == x", "x == 32'hffffff80"),
])
def test_yosys_sat_checks_generated_bitvector_logic(tmp_path, expression, golden):
    assert shutil.which("yosys"), "Yosys is required for HDL validation"
    expr = Elaborator({"x": signal("x", 32, "ex"), "y": signal("y", 32, "ex")}).parse(expression)
    emitter = SVEmitter({"x": "x", "y": "y"})
    result = emitter.emit(expr)
    source = "module miter(input [31:0] x, y, output ok);\n" + "\n".join(emitter.lines)
    source += f"\nassign ok = ({result} == ({golden}));\nendmodule\n"
    (tmp_path / "miter.sv").write_text(source)
    run(["yosys", "-Q", "-T", "-q", "-p", "read_verilog -sv miter.sv; prep -top miter; sat -verify -prove ok 1 -show-inputs"], tmp_path)


def test_yosys_sat_checks_generated_cross_field_set(tmp_path):
    expr = ISARegistry().compile("version 2\ninclude ADD\nforbid ADD {rd = x3, rs1 = x7}\n", signal("instr", 32, "ex"))
    emitter = SVEmitter({"instr": "instr"}); result = emitter.emit(expr)
    source = "module miter(input [31:0] instr, output ok);\n" + "\n".join(emitter.lines)
    source += f"\nwire expected = ((instr & 32'hfe00707f) == 32'h33) && !(instr[11:7] == 5'd3 && instr[19:15] == 5'd7);\nassign ok = {result} == expected;\nendmodule\n"
    (tmp_path / "miter.sv").write_text(source)
    run(["yosys", "-Q", "-T", "-q", "-p", "read_verilog -sv miter.sv; prep -top miter; sat -verify -prove ok 1 -show-inputs"], tmp_path)


def response_contract(low, high):
    return parse_contract(f"""crisp 3
clock core = clk_i reset rst_ni active_low
event bus on core scope interface when enabled
signal enabled : bool @ bus = enabled_i
signal request : bool @ bus = req_i
signal response : bool @ bus = rsp_i
assert response latency at bus: request -> response within {low}..{high}
""")


@pytest.mark.parametrize("low,high", [(0, 0), (0, 1), (1, 2), (2, 2)])
def test_generated_temporal_monitors_against_direct_trace_semantics(tmp_path, low, high):
    assert shutil.which("iverilog"), "Icarus Verilog is required for temporal validation"
    contract = response_contract(low, high)
    source, _ = emit_contract(contract)
    (tmp_path / "monitor.sv").write_text(source)
    # All 4-cycle Boolean request/response traces, plus reset and event gaps.
    traces = [[{"reset": False, "enabled": True, "request": bool(v & 1), "response": bool(v & 2)} for v in row]
              for row in itertools.product(range(4), repeat=4)]
    traces.extend([
        [{"reset": t == 1, "enabled": True, "request": t == 0, "response": False} for t in range(4)],
        [{"reset": False, "enabled": t != 1, "request": t == 0, "response": t == 1} for t in range(4)],
    ])
    tb = ["module tb; reg clk=0, rst=0, enabled=0, request=0, response=0; wire ok; always #5 clk=~clk;",
          "crisp_contract dut(.crisp_core_clk(clk), .crisp_core_rst(rst), .crisp_in_enabled(enabled), .crisp_in_request(request), .crisp_in_response(response), .crisp_check_latency(ok));",
          "initial begin"]
    expected = {}
    for i, samples in enumerate(traces):
        tb.append("@(negedge clk); rst=0; enabled=0; request=0; response=0;")
        failures = {v["cycle"] for v in audit_trace(contract, samples, "core")["violations"]}
        for t, sample in enumerate(samples):
            tb.append(f"@(negedge clk); rst={int(not sample['reset'])}; enabled={int(sample['enabled'])}; request={int(sample['request'])}; response={int(sample['response'])}; #1; $display(\"SAMPLE {i} {t} %b\", ok);")
            expected[i, t] = int(t not in failures)
    tb.append("$finish; end endmodule")
    (tmp_path / "tb.sv").write_text("\n".join(tb))
    run(["iverilog", "-g2012", "-s", "tb", "-o", "sim", "monitor.sv", "tb.sv"], tmp_path)
    output = run(["vvp", "sim"], tmp_path)
    actual = {(int(i), int(t)): int(value) for i, t, value in re.findall(r"SAMPLE (\d+) (\d+) ([01])", output)}
    assert actual == expected


def test_pending_deadlines_not_misreported_as_proofs():
    c = response_contract(1, 2)
    result = audit_trace(c, [{"reset": False, "enabled": True, "request": True, "response": False}], "core")
    assert not result["violations"] and len(result["pending"]) == 1
    assert result["evidence"] == "finite trace only"


def test_observation_validity_and_invalid_payloads():
    c = parse_contract(HEADER+"observe result at ex: x\n")
    a = [{"reset": False, "valid": True, "x": 7}]
    b = [{"reset": False, "valid": False, "x": 7}]
    assert compare_observations(c, a, b, "core")
    a[0]["valid"] = False
    b[0]["x"] = 123
    assert not compare_observations(c, a, b, "core")


def test_local_unsatisfiability_does_not_claim_reachability():
    c = parse_contract(HEADER+"assume a at ex: x == 0\nassume b at ex: x == 1\n")
    result = local_witnesses(c)["ex"]
    assert result["status"] == "unsat"
    assert "excludes" in result["scope"]


def test_real_elaboration_binding_validation(tmp_path):
    source = "module target(input clk_i, rst_ni, valid_i, input [31:0] x_i,y_i, output [31:0] o); assign o=x_i+y_i; endmodule"
    (tmp_path / "target.sv").write_text(source)
    run(["yosys", "-Q", "-T", "-q", "-p", "read_verilog target.sv; write_json target.json"], tmp_path)
    design = json.loads((tmp_path / "target.json").read_text())
    c = parse_contract(HEADER)
    assert validate_rtl_bindings(c, design, "target")["status"] == "names and widths checked"
    with pytest.raises(ContractError, match="declared width"):
        validate_rtl_bindings(parse_contract(HEADER.replace("x : bits[32]", "x : bits[16]")), design, "target")
    with pytest.raises(ContractError, match="not found"):
        validate_rtl_bindings(parse_contract(HEADER.replace("x_i", "missing_i")), design, "target")


@pytest.mark.parametrize("name", ["isa_sets", "remainder", "memory_regions", "memory_response"])
def test_examples_compile_through_yosys(tmp_path, name):
    path = ROOT / "examples" / "contracts" / f"{name}.crisp"
    c = parse_contract(path.read_text(), path.parent)
    code, manifest = emit_contract(c)
    (tmp_path / "contract.sv").write_text(code)
    run(["yosys", "-Q", "-T", "-q", "-p", "read_verilog -formal -sv contract.sv; prep -top crisp_contract; check -assert"], tmp_path)
    assert "no RTL equivalence" in manifest["guarantee"]
    assert all(not x["authorized"] for x in manifest["candidates"])


def test_library_change_invalidates_fingerprint(tmp_path):
    source = HEADER+'import "lib.crisp"\nassume a at ex: p(x)\n'
    library = tmp_path / "lib.crisp"
    library.write_text("library crisp 3\npredicate p(x: bits[W]) = fits_u(x, 8)\n")
    a = parse_contract(source, tmp_path).fingerprint()
    library.write_text("library crisp 3\npredicate p(x: bits[W]) = fits_u(x, 16)\n")
    assert a != parse_contract(source, tmp_path).fingerprint()


def test_cli_dispatch(tmp_path):
    source = ROOT / "examples" / "contracts" / "memory_regions.crisp"
    run(["python3", "-m", "pdat_dsl", "contract", str(source), "--output", str(tmp_path / "monitor.sv"), "--witnesses"], ROOT)
    manifest = json.loads((tmp_path / "monitor.json").read_text())
    assert len(manifest["clauses"]) == 2
    assert manifest["binding_validation"]["status"].startswith("unchecked")


def test_hdl_reserved_signal_names_and_obligation_roles(tmp_path):
    # A logical name must not break emitted Verilog or turn an obligation into
    # an assumption. The local consistency query deliberately excludes asserts.
    c = parse_contract(HEADER+"signal wire : bits[32] @ ex = keyword_i\nassert impossible at ex: False\nassume domain at ex: wire == x\n")
    code, manifest = emit_contract(c)
    assert manifest["bindings"]["crisp_in_wire"] == "keyword_i"
    assert local_witnesses(c)["ex"]["status"] == "sat"
    (tmp_path / "monitor.sv").write_text(code)
    run(["yosys", "-Q", "-T", "-q", "-p", "read_verilog -formal -sv monitor.sv; prep -top crisp_contract; write_json monitor.json"], tmp_path)
    cells = json.loads((tmp_path / "monitor.json").read_text())["modules"]["crisp_contract"]["cells"].values()
    assert sorted(c["type"] for c in cells if c["type"] in ("$assert", "$assume")) == ["$assert", "$assume"]
    with pytest.raises(ContractError, match="crisp_"):
        emit_contract(c, "module")


def test_new_isa_signature_needs_no_backend_changes():
    from pdat_dsl.encodings import InstructionEncoding
    registry = ISARegistry()
    registry.register_extension("CUSTOM", {
        "FOO": InstructionEncoding(name="FOO", base_pattern=0x0b, base_mask=0x7f, format="R", fields={"rd": (7, 5), "rs1": (15, 5)})
    })
    expr = registry.compile("version 2\ninclude CUSTOM\nforbid FOO {rd = x3, rs1 = x7}\n", signal("instr", 32, "ex"))
    w = z3.BitVec("instr", 32)
    prove_equal(to_smt(expr), z3.And(w & 0x7f == 0x0b, z3.Not(z3.And(z3.Extract(11, 7, w) == 3, z3.Extract(19, 15, w) == 7))))
    with pytest.raises(ContractError, match="duplicate instruction"):
        registry.register_extension("BAD", {"bar": registry.instructions["FOO"], "BAR": registry.instructions["FOO"]})
    assert "BAR" not in registry.instructions
