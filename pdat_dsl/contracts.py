"""CRISP 3: reusable predicates and contracts over explicitly sampled events.

The compiler emits verification harnesses and a provenance manifest. It never
authorizes an RTL cut: candidate transformations remain relational obligations.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .typed_core import (ContractError, Expr, Predicate, Elaborator, SVEmitter,
                         boolean, signal, logical, evaluate, to_smt)
from .typed_isa import ISARegistry

IDENT = r"[A-Za-z][A-Za-z0-9_]*"
RTL = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*"
BUILTINS = {"bv", "slice", "zext", "sext", "concat", "ite", "slt", "sle", "fits_u", "fits_s", "matches"}


@dataclass
class Clock:
    name: str
    binding: str
    reset: str
    active_low: bool


@dataclass
class Event:
    name: str
    clock: str
    scope: str
    guard_text: str
    guard: Expr = None


@dataclass
class Clause:
    kind: str
    name: str
    event: str
    predicate: Expr
    line: int


@dataclass
class Response:
    kind: str
    name: str
    event: str
    request: Expr
    response: Expr
    low: int
    high: int
    line: int


@dataclass
class Candidate:
    name: str
    event: str
    target: str
    replacement: Expr
    line: int


@dataclass
class Contract:
    source: str
    clocks: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)
    bindings: dict = field(default_factory=dict)
    predicates: dict = field(default_factory=dict)
    clauses: list = field(default_factory=list)
    responses: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    dependencies: dict = field(default_factory=dict)

    def fingerprint(self):
        # Includes library contents; HDL and manifest hashes bind actual elaboration.
        data = json.dumps({"source": self.source, "libraries": self.dependencies}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


def _predicate(line):
    m = re.fullmatch(rf"predicate ({IDENT})\((.*?)\)\s*=\s*(.+)", line)
    if not m: raise ContractError("expected predicate name(parameters) = expression")
    params = []
    for raw in m[2].split(",") if m[2].strip() else []:
        p = re.fullmatch(rf"\s*({IDENT})\s*:\s*(bool|nat|bits\[(?:\d+|[A-Z]\w*)\])\s*", raw)
        if not p: raise ContractError(f"invalid predicate parameter {raw!r}")
        if p[1] in {x[0] for x in params}: raise ContractError(f"duplicate parameter {p[1]}")
        params.append((p[1], p[2]))
    return m[1], Predicate(tuple(params), m[3])


def _lines(source):
    for n, raw in enumerate(source.splitlines(), 1):
        # Comments are only full lines; '#' inside a literal is preserved.
        line = raw.strip()
        if line and not line.startswith("#"):
            yield n, line


def _validate_predicates(predicates):
    graph = {}
    for name, pred in predicates.items():
        try: tree = ast.parse(pred.body, mode="eval")
        except SyntaxError as e: raise ContractError(f"predicate {name}: {e.msg}") from e
        params = {p[0] for p in pred.parameters}
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.keywords:
                    raise ContractError(f"predicate {name}: only positional DSL calls are allowed")
                calls.add(node.func.id)
            if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.NamedExpr,
                                 ast.ListComp, ast.DictComp, ast.GeneratorExp)):
                raise ContractError(f"predicate {name}: unsupported {type(node).__name__}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in params | BUILTINS | set(predicates):
                raise ContractError(f"predicate {name}: free name '{node.id}'; pass signals as parameters")
        if not calls <= BUILTINS | set(predicates):
            raise ContractError(f"predicate {name}: unknown call")
        graph[name] = calls & set(predicates)
    def visit(name, stack):
        if name in stack: raise ContractError(f"recursive predicate library: {' -> '.join(stack+(name,))}")
        for callee in graph[name]: visit(callee, stack+(name,))
    for name in graph: visit(name, ())


def parse_contract(source: str, base_dir=None, registry=None) -> Contract:
    registry = registry or ISARegistry()
    contract = Contract(source)
    lines = list(_lines(source))
    if not lines or lines[0][1] != "crisp 3": raise ContractError("a typed contract must start with 'crisp 3'")
    names, pending, policy = set(), [], None

    def reserve(name):
        if name in names or name.startswith("crisp_") or name in BUILTINS:
            raise ContractError(f"duplicate or reserved name '{name}'")
        names.add(name)

    for number, line in lines[1:]:
        try:
            if policy is not None:
                if line == "end":
                    pending.append((policy[0], "policy", policy[1], policy[2], policy[3], "\n".join(policy[4])))
                    policy = None
                else: policy[4].append(line)
                continue
            if line.startswith("import "):
                path_text = ast.literal_eval(line[7:])
                if not isinstance(path_text, str) or base_dir is None: raise ContractError("imports require a path and source directory")
                path = (Path(base_dir) / path_text).resolve()
                text = path.read_text()
                lib = list(_lines(text))
                if not lib or lib[0][1] != "library crisp 3": raise ContractError("imports may contain only a CRISP predicate library")
                contract.dependencies[path_text] = hashlib.sha256(text.encode()).hexdigest()
                for _, decl in lib[1:]:
                    name, pred = _predicate(decl)
                    reserve(name)
                    contract.predicates[name] = pred
                continue
            if line.startswith("predicate "):
                name, pred = _predicate(line)
                reserve(name)
                contract.predicates[name] = pred
                continue
            m = re.fullmatch(rf"clock ({IDENT}) = ({RTL}) reset ({RTL}) (active_low|active_high)", line)
            if m:
                reserve(m[1]); contract.clocks[m[1]] = Clock(m[1], m[2], m[3], m[4] == "active_low"); continue
            m = re.fullmatch(rf"event ({IDENT}) on ({IDENT}) scope (interface|decode|execute|retire) when (.+)", line)
            if m:
                reserve(m[1]); contract.events[m[1]] = Event(m[1], m[2], m[3], m[4]); continue
            m = re.fullmatch(rf"signal ({IDENT})\s*:\s*(bool|bits\[(\d+)\])\s*@\s*({IDENT}) = ({RTL})", line)
            if m:
                reserve(m[1]); contract.signals[m[1]] = signal(m[1], int(m[3]) if m[3] else None, m[4]); contract.bindings[m[1]] = m[5]; continue
            m = re.fullmatch(rf"policy ({IDENT}) at ({IDENT}) on ({IDENT})", line)
            if m:
                reserve(m[1]); policy = [number, m[1], m[2], m[3], ["version 2"]]; continue
            m = re.fullmatch(rf"(assume|assert) response ({IDENT}) at ({IDENT}): (.+) -> (.+) within (\d+)\.\.(\d+)", line)
            if m:
                reserve(m[2]); pending.append((number, "response", *m.groups())); continue
            m = re.fullmatch(rf"(assume|assert|observe) ({IDENT}) at ({IDENT}): (.+)", line)
            if m:
                reserve(m[2]); pending.append((number, "clause", *m.groups())); continue
            m = re.fullmatch(rf"candidate ({IDENT}) at ({IDENT}): ({IDENT}) -> (.+)", line)
            if m:
                reserve(m[1]); pending.append((number, "candidate", *m.groups())); continue
            raise ContractError("unknown declaration")
        except (ContractError, OSError, ValueError, SyntaxError) as e:
            raise ContractError(f"line {number}: {e}") from e
    if policy is not None: raise ContractError(f"line {policy[0]}: policy requires 'end'")
    if not contract.events: raise ContractError("a contract needs at least one sampling event")
    _validate_predicates(contract.predicates)
    for sig in contract.signals.values():
        if sig.type.event not in contract.events: raise ContractError(f"{sig.value}: unknown event {sig.type.event}")
    elaborator = Elaborator(contract.signals, contract.predicates, registry.matches)

    def at(text, event, bool_required=True):
        if event not in contract.events: raise ContractError(f"unknown event '{event}'")
        expression = elaborator.parse(text)
        if expression.type.event not in (None, event):
            raise ContractError(f"{expression.type} cannot be used at {event}; scopes are not implicitly lifted")
        if bool_required and expression.type.width is not None: raise ContractError("a proposition must have type bool")
        return expression

    for event in contract.events.values():
        if event.clock not in contract.clocks: raise ContractError(f"{event.name}: unknown clock {event.clock}")
        event.guard = at(event.guard_text, event.name)
    for number, form, *data in pending:
        try:
            if form == "clause":
                kind, name, event, body = data
                contract.clauses.append(Clause(kind, name, event, at(body, event, kind != "observe"), number))
            elif form == "policy":
                name, event, word, body = data
                expr = at(word, event, False)
                contract.clauses.append(Clause("assume", name, event, registry.compile(body, expr), number))
            elif form == "response":
                kind, name, event, req, rsp, low, high = data
                low, high = int(low), int(high)
                if not 0 <= low <= high <= 64: raise ContractError("response interval must satisfy 0 <= low <= high <= 64")
                contract.responses.append(Response(kind, name, event, at(req, event), at(rsp, event), low, high, number))
            else:
                name, event, target, replacement = data
                if target not in contract.signals: raise ContractError(f"unknown cut target {target}")
                before, after = at(target, event, False), at(replacement, event, False)
                if before.type.width != after.type.width: raise ContractError("replacement must have the target's width and kind")
                contract.candidates.append(Candidate(name, event, target, after, number))
        except ContractError as e:
            raise ContractError(f"line {number}: {e}") from e
    if contract.candidates and not any(c.kind == "observe" for c in contract.clauses):
        raise ContractError("candidate cuts require an explicit observation interface")
    return contract


def emit_contract(contract: Contract, module="crisp_contract"):
    """Emit a standalone sampled monitor and its binding/proof manifest.

    Input ports prefix logical signal names. The manifest maps them to RTL names;
    wiring, scope meanings, and actual RTL widths must be checked by integration.
    Response deadlines count clock edges, not just enabled event occurrences.
    Reset cancels pending response obligations and clears ghost history.
    """
    if not re.fullmatch(IDENT, module) or not module.startswith("crisp_"):
        raise ContractError("generated module names must be identifiers starting with crisp_")
    ports, mapping = [], {}
    for clock in contract.clocks.values():
        for suffix, binding in (("clk", clock.binding), ("rst", clock.reset)):
            port = f"crisp_{clock.name}_{suffix}"
            ports.append(f"input wire {port}"); mapping[port] = binding
    for name, sig in contract.signals.items():
        width = "" if sig.type.width is None else f"[{sig.type.width-1}:0] "
        port = f"crisp_in_{name}"
        ports.append(f"input wire {width}{port}"); mapping[port] = contract.bindings[name]
    emitter = SVEmitter({n: f"crisp_in_{n}" for n in contract.signals})
    logic, records = [], []
    def timing(event_name):
        event = contract.events[event_name]
        clock = contract.clocks[event.clock]
        reset = f"crisp_{clock.name}_rst"
        if clock.active_low: reset = f"!{reset}"
        return f"crisp_{clock.name}_clk", reset, emitter.emit(event.guard)
    for clause in contract.clauses:
        clk, reset, guard = timing(clause.event)
        value = emitter.emit(clause.predicate)
        record = {"kind": clause.kind, "name": clause.name, "event": clause.event,
                  "scope": contract.events[clause.event].scope, "source_line": clause.line,
                  "predicate_wire": value, "guard_wire": guard}
        if clause.kind == "observe":
            # Validity is an observation too; suppressing results is not equivalence.
            ports.extend([f"output wire crisp_obs_{clause.name}_valid", f"output wire {'' if clause.predicate.type.width is None else '['+str(clause.predicate.type.width-1)+':0] '}crisp_obs_{clause.name}_value"])
            logic.append(f"assign crisp_obs_{clause.name}_valid = !({reset}) && {guard};")
            logic.append(f"assign crisp_obs_{clause.name}_value = {value};")
            record["comparison"] = "equal validity every sample; equal value when valid"
        else:
            check = f"crisp_check_{clause.name}"
            ports.append(f"output wire {check}")
            logic.append(f"assign {check} = ({reset}) || !{guard} || {value};")
            logic.append(f"always @(posedge {clk}) {clause.kind} ({check}); // {clause.name}")
            record["check_port"] = check
            record["status"] = "environment assumption" if clause.kind == "assume" else "unproved obligation"
        records.append(record)
    for response in contract.responses:
        clk, reset, guard = timing(response.event)
        req = emitter.emit(logical("and", contract.events[response.event].guard, response.request))
        rsp = emitter.emit(logical("and", contract.events[response.event].guard, response.response))
        prefix, high = f"crisp_history_{response.name}", response.high
        if high:
            logic.extend([f"reg [{high-1}:0] {prefix}_req = {high}'b0;", f"reg [{high-1}:0] {prefix}_rsp = {high}'b0;"])
            req_shift = req if high == 1 else "{"+f"{prefix}_req[{high-2}:0], {req}"+"}"
            rsp_shift = rsp if high == 1 else "{"+f"{prefix}_rsp[{high-2}:0], {rsp}"+"}"
            logic.extend([f"always @(posedge {clk}) begin", f"  if ({reset}) begin {prefix}_req <= 0; {prefix}_rsp <= 0; end",
                          f"  else begin {prefix}_req <= {req_shift}; {prefix}_rsp <= {rsp_shift}; end", "end"])
        alternatives = [rsp if high-j == 0 else f"{prefix}_rsp[{high-j-1}]" for j in range(response.low, high+1)]
        due = f"{prefix}_req[{high-1}]" if high else req
        condition = f"!{due} || (" + " || ".join(alternatives) + ")"
        ports.append(f"output wire crisp_check_{response.name}")
        logic.append(f"assign crisp_check_{response.name} = ({reset}) || ({condition});")
        logic.append(f"always @(posedge {clk}) {response.kind} (crisp_check_{response.name}); // {response.name}")
        records.append({"kind": response.kind, "name": response.name, "event": response.event,
                        "scope": contract.events[response.event].scope, "source_line": response.line,
                        "response_window": [response.low, high], "status": "environment assumption" if response.kind == "assume" else "unproved obligation"})
    candidates = []
    for candidate in contract.candidates:
        replacement = emitter.emit(candidate.replacement)
        candidates.append({"name": candidate.name, "event": candidate.event,
                           "target": candidate.target, "replacement_wire": replacement,
                           "status": "pending relational preservation proof", "authorized": False})
    text = "// Generated CRISP 3 monitor. This file is not a proof certificate.\n"
    text += f"module {module}(\n  " + ",\n  ".join(ports) + "\n);\n"
    text += "\n".join(emitter.lines + logic) + "\nendmodule\n"
    manifest = {"crisp_version": 3, "source_sha256": contract.fingerprint(),
                "monitor_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "bindings": mapping, "clocks": {k: vars(v) for k, v in contract.clocks.items()},
                "signal_types": {k: str(v.type) for k, v in contract.signals.items()},
                "events": {k: {"clock": v.clock, "scope": v.scope, "guard": v.guard_text} for k, v in contract.events.items()},
                "clauses": records, "candidates": candidates,
                "trust_boundary": ["RTL binding and event scope meanings", "instruction encoding registry", "monitor compiler", "downstream proof engine"],
                "guarantee": "typed specification and generated obligations; no RTL equivalence result",
                "response_semantics": "sampled bounded response; reset cancels pending obligations; a response may satisfy multiple requests"}
    return text, manifest


def audit_trace(contract: Contract, samples: list, clock: str):
    """Direct finite-prefix semantics; return violations and open deadlines.

    samples are successive edges of one clock, with Boolean 'reset' indicating
    asserted reset. A clean finite prefix is not an inductive proof.
    """
    if clock not in contract.clocks: raise ContractError(f"unknown clock {clock}")
    failures, pending = [], []
    for t, sample in enumerate(samples):
        if type(sample.get("reset")) is not bool: raise ContractError("each trace sample needs Boolean reset")
        if sample["reset"]: continue
        for clause in contract.clauses:
            event = contract.events[clause.event]
            if event.clock != clock or clause.kind == "observe": continue
            if evaluate(event.guard, sample) and not evaluate(clause.predicate, sample):
                failures.append({"cycle": t, "name": clause.name, "kind": clause.kind})
        for response in contract.responses:
            event = contract.events[response.event]
            if event.clock != clock: continue
            if not (evaluate(event.guard, sample) and evaluate(response.request, sample)): continue
            deadline = t+response.high
            window = samples[t:min(deadline+1, len(samples))]
            if any(s["reset"] for s in window): continue
            found = any(evaluate(event.guard, samples[j]) and evaluate(response.response, samples[j])
                        for j in range(t+response.low, min(deadline+1, len(samples))))
            if not found:
                item = {"cycle": deadline, "name": response.name, "kind": response.kind}
                (failures if deadline < len(samples) else pending).append(item)
    return {"violations": failures, "pending": pending, "evidence": "finite trace only"}


def local_witnesses(contract: Contract, timeout_ms=2000):
    """Per-event combinational feasibility, NOT RTL reachability/realizability."""
    import z3
    results = {}
    for name, event in contract.events.items():
        names, solver = {}, z3.Solver()
        solver.set(timeout=timeout_ms)
        solver.add(to_smt(event.guard, names))
        for clause in contract.clauses:
            if clause.event == name and clause.kind == "assume": solver.add(to_smt(clause.predicate, names))
        status = solver.check()
        results[name] = {"status": str(status), "scope": "local predicates only; excludes temporal obligations and RTL transitions"}
    return results


def validate_rtl_bindings(contract: Contract, design: dict, top: str):
    """Check declared widths against a Yosys JSON elaboration of the binding scope.

    Flatten hierarchical designs before export. This establishes name/width
    agreement, not transaction identity, environment validity, or clock behavior.
    """
    try: nets = design["modules"][top]["netnames"]
    except KeyError as e: raise ContractError(f"no netnames for elaborated module {top}") from e
    requirements = [(contract.bindings[n], s.type.width or 1) for n, s in contract.signals.items()]
    requirements += [(c.binding, 1) for c in contract.clocks.values()]
    requirements += [(c.reset, 1) for c in contract.clocks.values()]
    for binding, width in requirements:
        if binding not in nets:
            raise ContractError(f"RTL binding {binding!r} not found in {top}; flatten hierarchy and expose whole signals")
        actual = len(nets[binding]["bits"])
        if width != actual: raise ContractError(f"RTL binding {binding}: declared width {width}, elaborated width {actual}")
    return {"status": "names and widths checked", "top": top,
            "design_sha256": hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()}


def compare_observations(contract: Contract, left: list, right: list, clock: str):
    """Finite-trace comparison of guard AND guarded value; no stuttering quotient."""
    if len(left) != len(right): raise ContractError("cycle-sensitive observations require equal trace lengths")
    if clock not in contract.clocks: raise ContractError(f"unknown clock {clock}")
    failures = []
    for t, (a, b) in enumerate(zip(left, right)):
        if type(a.get("reset")) is not bool or type(b.get("reset")) is not bool:
            raise ContractError("observation traces require Boolean reset")
        if a["reset"] != b["reset"]: raise ContractError("observation traces must share reset")
        if a["reset"]: continue
        for clause in contract.clauses:
            event = contract.events[clause.event]
            if clause.kind != "observe" or event.clock != clock: continue
            ga, gb = evaluate(event.guard, a), evaluate(event.guard, b)
            if ga != gb or (ga and evaluate(clause.predicate, a) != evaluate(clause.predicate, b)):
                failures.append({"cycle": t, "observation": clause.name})
    return failures


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Check and compile CRISP 3 contracts")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--output", type=Path, help="write SystemVerilog monitor and adjacent JSON manifest")
    parser.add_argument("--witnesses", action="store_true", help="check local predicate consistency with Z3")
    parser.add_argument("--rtl-json", type=Path, help="Yosys JSON for checking binding widths and names")
    parser.add_argument("--top", help="module scope in --rtl-json (flatten hierarchy first)")
    args = parser.parse_args(argv)
    try:
        contract = parse_contract(args.input_file.read_text(), args.input_file.parent)
        text, manifest = emit_contract(contract)
        if bool(args.rtl_json) != bool(args.top): raise ContractError("--rtl-json and --top must be supplied together")
        manifest["binding_validation"] = validate_rtl_bindings(contract, json.loads(args.rtl_json.read_text()), args.top) if args.rtl_json else {"status": "unchecked; declared binding signature is trusted"}
        if args.witnesses: manifest["local_witnesses"] = local_witnesses(contract)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
            args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n")
        print(json.dumps({"checked": str(args.input_file), "events": len(contract.events),
                          "clauses": len(contract.clauses)+len(contract.responses), "candidates": len(contract.candidates),
                          "equivalence_proved": False, "local_witnesses": manifest.get("local_witnesses", {})}, indent=2))
        return 0
    except (ContractError, OSError) as e:
        parser.exit(2, f"CRISP error: {e}\n")


if __name__ == "__main__":
    raise SystemExit(main())
