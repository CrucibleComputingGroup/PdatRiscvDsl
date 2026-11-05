"""Typed, exact-domain elaboration into the existing CRISP DSL.

This module does not emit SVA. Contexts name existing configured generator
routes; they do not add clocks, validity guards, or pipeline correspondence.
Operand expressions are accepted only when two solver queries establish exact
agreement with an explicitly stated legacy dtype domain. Other accepted rules
retain the legacy meaning and order, with unsupported legacy cases rejected.
"""
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .config import CoreConfig
from .encodings import get_instruction_encoding, get_extension_instructions
from .parser import (BitPattern, DataTypeSet, ForbidRule, IncludeRule,
                     InstructionPattern, InstructionRule, PatternRule,
                     PcConstraintRule, Program, RangeValue, RegisterConstraintRule,
                     RegisterRangeExpression, RequireRule, TimingConstraintRule, parse_dsl)
from .typed_core import ContractError, Elaborator, Predicate, signal, to_smt

IDENT = r"[A-Za-z_][A-Za-z_0-9]*"
KINDS = {"isa", "alu", "multdiv", "memory"}
BUILTINS = {"bv", "slice", "zext", "sext", "concat", "ite", "slt", "sle", "fits_u", "fits_s"}
MULTDIV = {"MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU"}


class FrontendError(ContractError):
    """A rejected frontend specification, including an optional exactness witness."""
    def __init__(self, message, witness=None):
        super().__init__(message)
        self.witness = witness


@dataclass
class FrontendResult:
    program: Program
    legacy_text: str
    manifest: dict


@dataclass(frozen=True)
class _Line:
    text: str
    number: int
    source: str

    def location(self):
        return {"source": self.source, "line": self.number}


@dataclass(frozen=True)
class _Bundle:
    parameters: tuple
    context_name: str
    context_kind: str
    body: tuple


def _parameters(text, predicate=False):
    params = []
    annotation = r"bool|nat|bits\[(?:\d+|[A-Z]\w*)\]" if predicate else r"instruction|nat|registers"
    for raw in text.split(",") if text.strip() else []:
        match = re.fullmatch(rf"\s*({IDENT})\s*:\s*({annotation})\s*", raw)
        if not match:
            raise FrontendError(f"invalid parameter {raw!r}")
        if match[1] in {p[0] for p in params}:
            raise FrontendError(f"duplicate parameter {match[1]}")
        params.append((match[1], match[2]))
    return tuple(params)


def _validate_predicates(predicates):
    graph = {}
    for name, pred in predicates.items():
        try:
            tree = ast.parse(pred.body, mode="eval")
        except SyntaxError as error:
            raise FrontendError(f"predicate {name}: {error.msg}") from error
        calls = set()
        params = {p[0] for p in pred.parameters}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.keywords:
                    raise FrontendError(f"predicate {name}: only positional DSL calls are supported")
                calls.add(node.func.id)
            if isinstance(node, ast.Name) and node.id not in params | BUILTINS | set(predicates):
                raise FrontendError(f"predicate {name}: free name '{node.id}'; use a parameter")
            if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.NamedExpr,
                                 ast.ListComp, ast.DictComp, ast.GeneratorExp)):
                raise FrontendError(f"predicate {name}: unsupported {type(node).__name__}")
        graph[name] = calls & set(predicates)
    _acyclic(graph, "predicate")


def _acyclic(graph, what):
    finished = set()
    def visit(name, stack):
        if name in stack:
            raise FrontendError(f"recursive {what}: {' -> '.join(stack + (name,))}")
        if name in finished:
            return
        for callee in graph[name]:
            visit(callee, stack + (name,))
        finished.add(name)
    for name in graph:
        visit(name, ())


def _context(kind, config):
    fields = {
        "isa": ("instruction_data", "pc", "rst_n"),
        "alu": ("instruction_data", "alu_rs1", "alu_rs2", "rst_n"),
        "multdiv": ("instruction_data", "multdiv_rs1", "multdiv_rs2", "rst_n"),
        "memory": ("clk", "rst_n", "instr_req", "instr_gnt", "data_req", "data_gnt", "data_rvalid"),
    }[kind]
    bindings = {field: getattr(config.signals, field, None) for field in fields}
    for field, value in bindings.items():
        if not isinstance(value, str) or not value.strip():
            raise FrontendError(f"{kind} context requires configured signal {field}")
    if not config.injection.module_path:
        raise FrontendError("contexts require a configured injection module")
    return {"kind": kind, "bindings": bindings, "data_width": config.data_width,
            "injection_module": config.injection.module_path,
            "sampling": "legacy counter state and combinational assumptions" if kind == "memory"
                        else "legacy combinational assumptions",
            "trust_boundary": "configuration is trusted; no RTL correspondence proof"}


def _exact_domain(expression, dtype, width, timeout_ms):
    try:
        import z3
    except ImportError as error:
        raise FrontendError("operand exactness checking requires z3-solver; install pdat-dsl[contracts]") from error
    value = z3.BitVec("value", width)
    source = to_smt(expression, {"value": value})
    domains = []
    for item in sorted(dtype.types, key=lambda t: (t.width, not t.signed)):
        if not 0 < item.width < width:
            raise FrontendError(f"legacy dtype {item} requires a width strictly below carrier bits[{width}]")
        if item.signed:
            limit = 1 << (item.width - 1)
            domains.append(z3.Or(z3.ULT(value, limit), z3.UGE(value, (1 << width) - limit)))
        else:
            domains.append(z3.ULT(value, 1 << item.width))
    target = z3.Or(*domains)
    if not dtype.negated:
        target = z3.Not(target)
    for direction, query in (("lost", z3.And(source, z3.Not(target))),
                             ("extra", z3.And(target, z3.Not(source)))):
        solver = z3.Solver()
        solver.set(timeout=timeout_ms)
        solver.add(query)
        result = solver.check()
        if result == z3.sat:
            number = solver.model().eval(value, model_completion=True).as_long()
            witness = {"direction": direction, "unsigned": number,
                       "hex": f"0x{number:0{(width + 3)//4}x}", "width": width,
                       "source_accepts": direction == "lost", "legacy_accepts": direction == "extra"}
            raise FrontendError(f"inexact lowering to {dtype}: {direction} value {witness['hex']} ({number})", witness)
        if result != z3.unsat:
            raise FrontendError(f"exactness check inconclusive ({direction}): {solver.reason_unknown()}")
    return {"source_minus_legacy": "unsat", "legacy_minus_source": "unsat", "carrier_width": width}


def _instruction(name):
    encoding = get_instruction_encoding(name)
    if encoding is None:
        raise FrontendError(f"unknown instruction '{name}'")
    return encoding


def _check_fields(name, constraints):
    encoding = _instruction(name)
    if encoding.is_compressed and constraints:
        raise FrontendError("compressed field constraints are outside the checked legacy subset")
    names = set()
    for item in constraints:
        if item.field_name in names:
            raise FrontendError(f"duplicate field {name}.{item.field_name}")
        names.add(item.field_name)
        if isinstance(item.field_value, DataTypeSet) or item.field_name.endswith("dtype"):
            raise FrontendError("dtype rules must use 'operand ... satisfies ... lowers ...' for exactness checking")
        if item.field_name not in encoding.fields:
            raise FrontendError(f"unknown field {name}.{item.field_name}")
        position, width = encoding.fields[item.field_name]
        if encoding.base_mask & (((1 << width) - 1) << position):
            raise FrontendError(f"field {name}.{item.field_name} overlaps fixed instruction bits")
        value = item.field_value
        if isinstance(value, BitPattern) and value.width != width:
            raise FrontendError(f"{name}.{item.field_name} has width {width}, not {value.width}")
        if isinstance(value, RangeValue):
            raise FrontendError("legacy v1 field-range lowering is unsupported; use an exact bit pattern")
        elif isinstance(value, int):
            bounds = (value, value)
        elif isinstance(value, str) and re.fullmatch(r"x\d+", value):
            bounds = (int(value[1:]), int(value[1:]))
        elif value == "*" or isinstance(value, BitPattern):
            continue
        else:
            raise FrontendError(f"unsupported value for {name}.{item.field_name}")
        if not 0 <= bounds[0] <= bounds[1] < (1 << width):
            raise FrontendError(f"value outside {width}-bit field {name}.{item.field_name}")


def elaborate_frontend(source: str, config: CoreConfig, base_dir=None, *, solver_timeout_ms=5000) -> FrontendResult:
    """Check and erase a frontend specification to legacy DSL and its original AST.

    Libraries start with ``crisp library 1`` and contain only predicates,
    constraint definitions, and imports. A program starts ``crisp frontend 1``.
    ``lowers`` names an existing dtype set, not an alternative output emitter.
    No hardware correctness or general temporal equivalence claim is made.
    """
    if config.architecture not in ("rv32", "rv64"):
        raise FrontendError("frontend requires configured rv32 or rv64 architecture")
    if type(solver_timeout_ms) is not int or solver_timeout_ms <= 0:
        raise FrontendError("solver_timeout_ms must be a positive integer")
    predicates, bundles, contexts, actions, imports = {}, {}, {}, [], []
    names, loaded = set(), set()
    backend = 1
    backend_seen = False

    def reserve(name):
        if name in names or name in BUILTINS or name == "value":
            raise FrontendError(f"duplicate or reserved name '{name}'")
        names.add(name)

    def read(text, directory, origin, library=False, stack=()):
        nonlocal backend, backend_seen
        lines = [_Line(raw.strip(), n, origin) for n, raw in enumerate(text.splitlines(), 1)
                 if raw.strip() and not raw.lstrip().startswith("#")]
        header = "crisp library 1" if library else "crisp frontend 1"
        if not lines or lines[0].text != header:
            raise FrontendError(f"{origin}: expected '{header}'")
        index = 1
        while index < len(lines):
            line = lines[index]
            index += 1
            try:
                if line.text.startswith("import "):
                    try:
                        path_string = ast.literal_eval(line.text[7:])
                    except (ValueError, SyntaxError) as error:
                        raise FrontendError("import requires a quoted path") from error
                    if not isinstance(path_string, str) or directory is None:
                        raise FrontendError("imports require a quoted path and base_dir")
                    path = (Path(directory) / path_string).resolve()
                    if path in stack:
                        raise FrontendError("cyclic library import: " + " -> ".join(map(str, stack + (path,))))
                    if path not in loaded:
                        content = path.read_text()
                        imports.append({"path": str(path), "sha256": hashlib.sha256(content.encode()).hexdigest()})
                        read(content, path.parent, str(path), True, stack + (path,))
                        loaded.add(path)
                    continue
                match = re.fullmatch(rf"predicate ({IDENT})\((.*?)\)\s*=\s*(.+)", line.text)
                if match:
                    reserve(match[1])
                    predicates[match[1]] = Predicate(_parameters(match[2], True), match[3])
                    continue
                match = re.fullmatch(rf"constraint ({IDENT})\((.*?)\) at ({IDENT}): (isa|alu|multdiv|memory):", line.text)
                if match:
                    reserve(match[1])
                    params = _parameters(match[2])
                    if match[3] in {p[0] for p in params}:
                        raise FrontendError("context parameter duplicates a value parameter")
                    body = []
                    while index < len(lines) and lines[index].text != "end":
                        body.append(lines[index])
                        index += 1
                    if index == len(lines):
                        raise FrontendError("constraint missing 'end'")
                    index += 1
                    if not body:
                        raise FrontendError("constraint body cannot be empty")
                    allowed = {p[0] for p in params} | {match[3]}
                    for entry in body:
                        free = set(re.findall(rf"\$({IDENT})", entry.text)) - allowed
                        if free:
                            raise FrontendError(f"unbound constraint parameter ${sorted(free)[0]}")
                        route = re.search(rf"\bat (\$?{IDENT})", entry.text)
                        if not route or route[1] != "$" + match[3]:
                            raise FrontendError("constraint bodies must use their explicit context parameter")
                    bundles[match[1]] = _Bundle(params, match[3], match[4], tuple(body))
                    continue
                if library:
                    raise FrontendError("libraries contain definitions and imports only")
                match = re.fullmatch(r"backend ([12])", line.text)
                if match:
                    if backend_seen:
                        raise FrontendError("duplicate backend declaration")
                    backend, backend_seen = int(match[1]), True
                    continue
                match = re.fullmatch(rf"context ({IDENT}) = (isa|alu|multdiv|memory)", line.text)
                if match:
                    reserve(match[1])
                    contexts[match[1]] = _context(match[2], config)
                    continue
                actions.append(line)
            except (ContractError, SyntaxError, ValueError, OSError) as error:
                raise FrontendError(f"{origin}:{line.number}: {error}", getattr(error, "witness", None)) from error

    read(source, base_dir, "<source>")
    _validate_predicates(predicates)
    graph = {}
    for name, bundle in bundles.items():
        graph[name] = set()
        for line in bundle.body:
            if line.text.startswith("use "):
                match = re.match(rf"use ({IDENT})\(", line.text)
                if not match or match[1] not in bundles:
                    raise FrontendError(f"constraint {name}: unknown constraint call")
                graph[name].add(match[1])
    _acyclic(graph, "constraint")
    statements, provenance, proofs, legacy_effects, warnings = [], [], [], [], []
    singleton_rules, v2_includes, v2_active = set(), set(), set()
    has_v1_require = False
    has_v1_fields = False

    def route(name, required=None):
        if name not in contexts:
            raise FrontendError(f"unknown context '{name}'")
        context = contexts[name]
        if required is not None and context["kind"] != required:
            raise FrontendError(f"context '{name}' has kind {context['kind']}; requires {required}")
        return context

    def append(statement, line, stack, context, role):
        statements.append(statement)
        provenance.append({**line.location(), "expansion": list(stack), "context": context,
                           "role": role, "legacy_rule": len(statements), "legacy_text": statement})

    def expand(line, stack=()):
        nonlocal has_v1_require, has_v1_fields
        if len(stack) > 64 or len(statements) > 10000:
            raise FrontendError("constraint expansion exceeds resource limit")
        text = line.text
        match = re.fullmatch(rf"use ({IDENT})\((.*?)\) at ({IDENT})", text)
        if match:
            if match[1] not in bundles:
                raise FrontendError(f"unknown constraint '{match[1]}'")
            bundle = bundles[match[1]]
            route(match[3], bundle.context_kind)
            args = [a.strip() for a in match[2].split(",")] if match[2].strip() else []
            if len(args) != len(bundle.parameters):
                raise FrontendError(f"constraint {match[1]} expects {len(bundle.parameters)} arguments")
            values = {bundle.context_name: match[3]}
            for (param, annotation), arg in zip(bundle.parameters, args):
                if annotation == "nat":
                    if not arg.isascii() or not arg.isdigit() or not 0 <= int(arg) <= 4096:
                        raise FrontendError(f"{match[1]}.{param} expects nat in 0..4096")
                    arg = str(int(arg))
                elif annotation == "instruction":
                    _instruction(arg)
                else:
                    regs = re.fullmatch(r"x(\d+)-x(\d+)", arg)
                    if not regs or not 0 <= int(regs[1]) <= int(regs[2]) <= 31:
                        raise FrontendError(f"{match[1]}.{param} expects a register range x0-x31")
                values[param] = arg
            for entry in bundle.body:
                body = re.sub(rf"\$({IDENT})", lambda m: values[m[1]], entry.text)
                expand(_Line(body, entry.number, entry.source), stack + (f"{match[1]}@{line.source}:{line.number}",))
            return
        match = re.fullmatch(rf"operand ({IDENT})\.(rs1|rs2|inputs|rd) at ({IDENT}) satisfies (.+) lowers (.+)", text)
        if match:
            instruction, operand, context_name, source_expr, target_text = match.groups()
            if backend != 1:
                raise FrontendError("legacy backend 2 drops dtype rules; operand predicates require backend 1")
            encoding = _instruction(instruction)
            if operand == "rd":
                raise FrontendError("legacy backend has no output operand route; rd constraints are unsupported")
            operands = [o for o in ("rs1", "rs2") if o in encoding.fields] if operand == "inputs" else [operand]
            if not operands or any(o not in encoding.fields for o in operands):
                raise FrontendError(f"{instruction} has no requested input operand {operand}")
            if encoding.is_compressed:
                raise FrontendError("compressed operand dtype lowering is unsupported by this frontend")
            context = route(context_name, "multdiv" if instruction.upper() in MULTDIV else "alu")
            field = "dtype" if operand == "inputs" else operand + "_dtype"
            statement = f"instruction {instruction} {{ {field} = {target_text} }}"
            parsed = parse_dsl(statement)
            if len(parsed.rules) != 1 or len(parsed.rules[0].constraints) != 1:
                raise FrontendError("lowers must contain exactly one legacy dtype set")
            dtype = parsed.rules[0].constraints[0].field_value
            if not isinstance(dtype, DataTypeSet):
                raise FrontendError("lowers requires a legacy dtype set")
            typed = Elaborator({"value": signal("value", config.data_width, context_name)}, predicates).parse(source_expr)
            if typed.type.width is not None:
                raise FrontendError("operand predicate must return bool")
            proof = _exact_domain(typed, dtype, config.data_width, solver_timeout_ms)
            statement = f"instruction {instruction} {{ {field} = {dtype} }}"
            proofs.append({**line.location(), "context": context_name, "instruction": instruction,
                           "operands": operands, "bindings": {o: config.get_operand_signal(instruction, o) for o in operands},
                           "expression": source_expr, "legacy_domain": str(dtype),
                           "scope": "local dtype predicate only; not whole execution admission", **proof})
            warning = (f"The standalone legacy generator also outlaws instruction {instruction} for this dtype rule; "
                       "the exactness proof covers only its local operand predicate, not execution coverage.")
            legacy_effects.append({**line.location(), "kind": "dtype_instruction_exclusion",
                                   "instruction": instruction, "message": warning,
                                   "emission_path": "pdat_dsl.codegen.main (not the filtered hierarchical flow)",
                                   "cause": "instruction_rule_to_pattern retains the base instruction pattern after skipping dtype fields"})
            warnings.append(warning)
            append(statement, line, stack, context_name, "operand_assumption")
            return
        match = re.fullmatch(rf"at ({IDENT}): (.+)", text)
        if not match:
            raise FrontendError("expected context-bound constraint, operand predicate, or constraint use; custom clocks/guards/outputs are unsupported")
        context_name, statement = match.groups()
        parsed = parse_dsl(statement + "\n")
        if parsed.version is not None or len(parsed.rules) != 1:
            raise FrontendError("each context-bound statement must contain one legacy rule")
        rule = parsed.rules[0]
        if isinstance(rule, TimingConstraintRule):
            route(context_name, "memory")
            if rule.param_name == "locality_bits":
                raise FrontendError("legacy backend ignores locality_bits; cannot preserve its intended constraint")
            if not 0 <= rule.value <= 7:
                raise FrontendError("legacy timing counters are 3 bits; latency must be in 0..7")
            key, role = rule.param_name, "timing_assumption"
        else:
            route(context_name, "isa")
            key, role = None, "instruction_assumption"
            if isinstance(rule, RequireRule):
                if backend != 1:
                    raise FrontendError("require is supported only by backend 1")
                if not get_extension_instructions(rule.extension):
                    raise FrontendError(f"unknown or empty extension {rule.extension}")
                has_v1_require = True
            elif isinstance(rule, RegisterConstraintRule):
                if backend != 1:
                    raise FrontendError("backend 2 ignores require_registers; use forbid register ranges")
                if not 0 <= rule.min_reg <= rule.max_reg <= 31:
                    raise FrontendError("register constraint must be within x0-x31")
                if rule.min_reg != 0:
                    raise FrontendError("legacy backend ignores register lower bounds; range must start at x0")
                key = "registers"
            elif isinstance(rule, PcConstraintRule):
                if not 1 <= rule.pc_bits < min(config.data_width, 32 if backend == 2 else config.data_width):
                    raise FrontendError("PC bit count must be strictly below the legacy PC carrier width")
                key = "pc"
            elif isinstance(rule, InstructionRule):
                if backend != 1 or rule.allow:
                    raise FrontendError("legacy allow/instruction combination is unsupported; use backend-appropriate forbid rules")
                _check_fields(rule.name, rule.constraints)
                has_v1_fields = has_v1_fields or bool(rule.constraints)
            elif isinstance(rule, (IncludeRule, ForbidRule)):
                if backend != 2:
                    raise FrontendError("include/forbid require backend 2")
                expr = rule.expr
                if isinstance(expr, InstructionPattern):
                    _check_fields(expr.name, expr.constraints)
                    raise FrontendError("legacy backend 2 ignores all constrained inclusion/exclusion patterns")
                elif isinstance(expr, RegisterRangeExpression):
                    if isinstance(rule, IncludeRule):
                        raise FrontendError("legacy register inclusion cannot restore an exact instruction set")
                    if not expr.registers or min(expr.registers) < 0 or max(expr.registers) > 31:
                        raise FrontendError("register set must be within x0-x31")
                    if not expr.is_contiguous() or min(expr.registers) == 0 or max(expr.registers) != 31:
                        raise FrontendError("checked backend 2 register exclusion requires one suffix xN-x31 with N > 0")
                    key = "registers"
                    affected = set()
                else:
                    affected = set(get_extension_instructions(expr) or ())
                    if not affected:
                        _instruction(expr)
                        affected = {expr}
                if isinstance(rule, IncludeRule):
                    if any(_instruction(name).is_compressed for name in affected):
                        raise FrontendError("legacy backend 2 bypasses compressed instructions")
                    if affected & v2_includes:
                        raise FrontendError("repeated inclusion may lose cases in legacy backend; use a single initial inclusion")
                    v2_includes.update(affected)
                    v2_active.update(affected)
                elif not isinstance(expr, RegisterRangeExpression):
                    v2_active.difference_update(affected)
            elif isinstance(rule, PatternRule):
                raise FrontendError("raw patterns have source-line-dependent legacy output; use named instruction constraints")
            else:
                raise FrontendError(f"unsupported legacy rule {type(rule).__name__}")
        if key is not None:
            if key in singleton_rules:
                raise FrontendError(f"duplicate {key} constraint would overwrite an earlier rule in legacy backend")
            singleton_rules.add(key)
        append(statement, line, stack, context_name, role)

    for action in actions:
        try:
            expand(action)
        except (ContractError, SyntaxError, ValueError) as error:
            raise FrontendError(f"{action.source}:{action.number}: {error}", getattr(error, "witness", None)) from error
    if has_v1_require and has_v1_fields:
        raise FrontendError("legacy backend 1 broadens field exclusions to whole instructions when combined with require")
    if backend == 2 and not v2_active:
        raise FrontendError("legacy backend 2 cannot emit a valid empty instruction set")
    if any(r["role"] == "timing_assumption" for r in provenance):
        warning = ("Legacy timing counter blocks contain literal signal placeholders; output is preserved "
                   "for compatibility and is not certified as a valid temporal monitor.")
        legacy_effects.append({"kind": "timing_literal_placeholders", "message": warning})
        warnings.append(warning)
    legacy_text = f"version {backend}\n" + "\n".join(statements) + ("\n" if statements else "")
    manifest = {"frontend_version": 1, "backend_version": backend, "core": config.core_name,
                "architecture": config.architecture, "contexts": contexts,
                "imports": imports, "predicates": sorted(predicates), "constraints": sorted(bundles),
                "rules": provenance, "exactness_checks": proofs,
                "legacy_effects": legacy_effects, "warnings": warnings,
                "legacy_sha256": hashlib.sha256(legacy_text.encode()).hexdigest(),
                "guarantees": ["typed expansion preserves rule order", "local operand domains equal declared legacy dtype domains"],
                "not_proved": ["configuration matches RTL", "instruction and operand transaction correspondence",
                               "whole-program execution coverage or legacy generator correctness",
                               "legacy timing semantics satisfy a separate temporal specification", "hardware removal correctness"]}
    return FrontendResult(parse_dsl(legacy_text), legacy_text, manifest)
