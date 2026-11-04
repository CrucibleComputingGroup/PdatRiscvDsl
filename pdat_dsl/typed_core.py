"""Small, exact, event-indexed Boolean/bit-vector core for CRISP contracts.

Expressions have one sampling event (or are constants). This is a specification
language: checking a predicate's type does not establish its truth in RTL.
No host-language eval, truncating pattern enumeration, or implicit casts occur.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping, Optional


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class Type:
    width: Optional[int] = None  # None means Boolean
    event: Optional[str] = None

    def __str__(self):
        base = "bool" if self.width is None else f"bits[{self.width}]"
        return base + (f"@{self.event}" if self.event else "")


@dataclass(frozen=True)
class Expr:
    op: str
    type: Type
    args: tuple = ()
    value: object = None


def boolean(value: bool) -> Expr:
    return Expr("const", Type(), value=bool(value))


def bits(value: int, width: int) -> Expr:
    if type(width) is not int or not 1 <= width <= 4096:
        raise ContractError("bit width must be between 1 and 4096")
    if type(value) is not int or not 0 <= value < (1 << width):
        raise ContractError(f"{value!r} is outside unsigned bits[{width}]; use explicit two's-complement bits")
    return Expr("const", Type(width), value=value)


def signal(name: str, width: Optional[int], event: str) -> Expr:
    if width is not None:
        bits(0, width)
    return Expr("signal", Type(width, event), value=name)


def event_of(*exprs: Expr) -> Optional[str]:
    events = {e.type.event for e in exprs if e.type.event is not None}
    if len(events) > 1:
        raise ContractError(f"cannot combine sampling events {', '.join(sorted(events))}; an explicit correspondence proof is required")
    return next(iter(events), None)


def logical(op: str, *args: Expr) -> Expr:
    if op not in ("and", "or", "not") or (op == "not" and len(args) != 1):
        raise ContractError(f"invalid Boolean operator {op}")
    if any(a.type.width is not None for a in args):
        raise ContractError("Boolean operators require bool, not bit vectors")
    event = event_of(*args)
    if not args:
        return boolean(op == "and")
    if op != "not" and len(args) == 1:
        return args[0]
    return Expr(op, Type(event=event), tuple(args))


def binary(op: str, left: Expr, right: Expr) -> Expr:
    event = event_of(left, right)
    if left.type.width != right.type.width:
        raise ContractError(f"width mismatch: {left.type} and {right.type}")
    if op not in ("eq", "ne", "and_bits", "or_bits", "xor", "add", "sub", "mul", "ult", "ule", "slt", "sle"):
        raise ContractError(f"unknown operator {op}")
    if op not in ("eq", "ne") and left.type.width is None:
        raise ContractError(f"{op} requires bit vectors")
    width = None if op in ("eq", "ne", "ult", "ule", "slt", "sle") else left.type.width
    return Expr(op, Type(width, event), (left, right))


def extract(expr: Expr, hi: int, lo: int) -> Expr:
    if expr.type.width is None or not 0 <= lo <= hi < expr.type.width:
        raise ContractError(f"invalid slice [{hi}:{lo}] of {expr.type}")
    return Expr("slice", Type(hi-lo+1, expr.type.event), (expr,), (hi, lo))


def extend(expr: Expr, width: int, signed: bool = False) -> Expr:
    if expr.type.width is None or width < expr.type.width:
        raise ContractError("extension needs a bit vector and a nondecreasing width")
    bits(0, width)
    return Expr("sext" if signed else "zext", Type(width, expr.type.event), (expr,))


def concat(*args: Expr) -> Expr:
    if not args or any(a.type.width is None for a in args):
        raise ContractError("concat requires bit vectors")
    width = sum(a.type.width for a in args)
    bits(0, width)
    return Expr("concat", Type(width, event_of(*args)), tuple(args))


def choose(cond: Expr, yes: Expr, no: Expr) -> Expr:
    if cond.type.width is not None or yes.type.width != no.type.width:
        raise ContractError("if_then_else requires a Boolean condition and equal branch types")
    return Expr("ite", Type(yes.type.width, event_of(cond, yes, no)), (cond, yes, no))


def evaluate(expr: Expr, values: Mapping[str, object]):
    """Reference value semantics; bit-vector arithmetic wraps at its width."""
    if expr.op == "const":
        return expr.value
    if expr.op == "signal":
        value = values[expr.value]
        if expr.type.width is None:
            if type(value) is not bool:
                raise ContractError(f"{expr.value} requires a Boolean sample")
        else:
            bits(value, expr.type.width)
        return value
    a = [evaluate(x, values) for x in expr.args]
    if expr.op == "not": return not a[0]
    if expr.op == "and": return all(a)
    if expr.op == "or": return any(a)
    if expr.op == "eq": return a[0] == a[1]
    if expr.op == "ne": return a[0] != a[1]
    if expr.op == "ult": return a[0] < a[1]
    if expr.op == "ule": return a[0] <= a[1]
    if expr.op in ("slt", "sle"):
        width = expr.args[0].type.width
        a = [v - (1 << width) if v & (1 << (width-1)) else v for v in a]
        return a[0] < a[1] if expr.op == "slt" else a[0] <= a[1]
    if expr.op == "slice": return (a[0] >> expr.value[1]) & ((1 << expr.type.width)-1)
    if expr.op == "ite": return a[1] if a[0] else a[2]
    if expr.op == "concat":
        result = 0
        for child, val in zip(expr.args, a):
            result = (result << child.type.width) | val
        return result
    if expr.op in ("sext", "zext"):
        old = expr.args[0].type.width
        value = a[0]
        if expr.op == "sext" and value & (1 << (old-1)):
            value -= 1 << old
    else:
        operations = {"and_bits": lambda: a[0] & a[1], "or_bits": lambda: a[0] | a[1],
                      "xor": lambda: a[0] ^ a[1], "add": lambda: a[0]+a[1],
                      "sub": lambda: a[0]-a[1], "mul": lambda: a[0]*a[1]}
        value = operations[expr.op]()
    return value & ((1 << expr.type.width)-1)


class SVEmitter:
    """DAG lowering to explicitly sized wires; safe for slices of expressions."""
    def __init__(self, names: Mapping[str, str], prefix="crisp"):
        self.names, self.prefix = names, prefix
        self.memo, self.lines = {}, []

    def emit(self, expr: Expr) -> str:
        if expr in self.memo:
            return self.memo[expr]
        if expr.op == "signal": return self.names[expr.value]
        if expr.op == "const":
            # Materialize constants too: Verilog cannot portably slice a literal
            # directly (e.g. 8'h80[7]). The same DAG path handles all operands.
            literal = ("1'b1" if expr.value else "1'b0") if expr.type.width is None else f"{expr.type.width}'h{expr.value:x}"
            name = f"{self.prefix}_{len(self.memo)}"
            width = "" if expr.type.width is None else f"[{expr.type.width-1}:0] "
            self.lines.append(f"wire {width}{name} = {literal};")
            self.memo[expr] = name
            return name
        a = [self.emit(x) for x in expr.args]
        operators = {"eq": "==", "ne": "!=", "and_bits": "&", "or_bits": "|",
                     "xor": "^", "add": "+", "sub": "-", "mul": "*", "ult": "<", "ule": "<="}
        if expr.op in operators: body = f"({a[0]} {operators[expr.op]} {a[1]})"
        elif expr.op in ("slt", "sle"): body = f"($signed({a[0]}) {'<' if expr.op == 'slt' else '<='} $signed({a[1]}))"
        elif expr.op in ("and", "or"): body = "(" + (" && " if expr.op == "and" else " || ").join(a) + ")"
        elif expr.op == "not": body = f"(!{a[0]})"
        elif expr.op == "slice": body = f"{a[0]}[{expr.value[0]}:{expr.value[1]}]"
        elif expr.op == "concat": body = "{" + ", ".join(a) + "}"
        elif expr.op == "ite": body = f"({a[0]} ? {a[1]} : {a[2]})"
        elif expr.op in ("sext", "zext"):
            old = expr.args[0].type.width
            extra = expr.type.width - old
            fill = f"{a[0]}[{old-1}]" if expr.op == "sext" else "1'b0"
            body = a[0] if not extra else "{{" + str(extra) + "{" + fill + "}}, " + a[0] + "}"
        else: raise ContractError(f"no SystemVerilog lowering for {expr.op}")
        name = f"{self.prefix}_{len(self.memo)}"
        width = "" if expr.type.width is None else f"[{expr.type.width-1}:0] "
        self.lines.append(f"wire {width}{name} = {body};")
        self.memo[expr] = name
        return name


def to_smt(expr: Expr, names=None):
    """Independent solver encoding of the core (optional z3 dependency)."""
    import z3
    if names is None: names = {}
    if expr.op == "signal":
        if expr.value not in names:
            names[expr.value] = z3.Bool(expr.value) if expr.type.width is None else z3.BitVec(expr.value, expr.type.width)
        return names[expr.value]
    if expr.op == "const":
        return z3.BoolVal(expr.value) if expr.type.width is None else z3.BitVecVal(expr.value, expr.type.width)
    a = [to_smt(x, names) for x in expr.args]
    if expr.op == "not": return z3.Not(a[0])
    if expr.op == "and": return z3.And(*a)
    if expr.op == "or": return z3.Or(*a)
    if expr.op == "eq": return a[0] == a[1]
    if expr.op == "ne": return a[0] != a[1]
    if expr.op == "ult": return z3.ULT(*a)
    if expr.op == "ule": return z3.ULE(*a)
    if expr.op == "slt": return a[0] < a[1]
    if expr.op == "sle": return a[0] <= a[1]
    if expr.op == "slice": return z3.Extract(*expr.value, a[0])
    if expr.op == "concat": return a[0] if len(a) == 1 else z3.Concat(*a)
    if expr.op == "ite": return z3.If(*a)
    if expr.op in ("sext", "zext"):
        return (z3.SignExt if expr.op == "sext" else z3.ZeroExt)(expr.type.width-expr.args[0].type.width, a[0])
    return {"and_bits": lambda: a[0] & a[1], "or_bits": lambda: a[0] | a[1],
            "xor": lambda: a[0] ^ a[1], "add": lambda: a[0]+a[1],
            "sub": lambda: a[0]-a[1], "mul": lambda: a[0]*a[1]}[expr.op]()


@dataclass(frozen=True)
class Predicate:
    parameters: tuple  # (name, annotation), e.g. ('x', 'bits[W]')
    body: str


class Elaborator:
    """Bidirectional literal checking and hygienic predicate instantiation.

    Python's parser supplies expression syntax only. AST nodes are explicitly
    interpreted; attributes, indexing, comprehensions, and host calls are absent.
    Natural parameters are elaboration-time integers, never RTL values.
    """
    def __init__(self, signals: Mapping[str, Expr], predicates=None, match=None):
        self.signals = signals
        self.predicates = predicates or {}
        self.match = match

    def parse(self, text: str, expected=None) -> Expr:
        try:
            result = self._visit(ast.parse(text, mode="eval").body, {}, (), expected)
        except SyntaxError as e:
            raise ContractError(f"invalid expression: {e.msg}") from e
        if not isinstance(result, Expr):
            raise ContractError("a contract requires a typed expression, not a natural or string")
        return result

    def _visit(self, node, locals_, stack, expected=None):
        visit = lambda n, ex=None: self._visit(n, locals_, stack, ex)
        if isinstance(node, ast.Constant):
            if type(node.value) is bool: return boolean(node.value)
            if type(node.value) is int:
                if expected is not None and expected.width is not None:
                    return bits(node.value, expected.width)
                return node.value
            if type(node.value) is str: return node.value
        if isinstance(node, ast.Name):
            if node.id in locals_: return locals_[node.id]
            if node.id in self.signals: return self.signals[node.id]
            raise ContractError(f"unknown name '{node.id}'")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return logical("not", self._expr(visit(node.operand)))
        if isinstance(node, ast.BoolOp):
            return logical("and" if isinstance(node.op, ast.And) else "or", *(self._expr(visit(v)) for v in node.values))
        if isinstance(node, (ast.Compare, ast.BinOp)):
            if isinstance(node, ast.Compare):
                if len(node.ops) != 1: raise ContractError("write chained comparisons as explicit conjunctions")
                op = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "ult", ast.LtE: "ule"}.get(type(node.ops[0]))
                right_node = node.comparators[0]
            else:
                op = {ast.BitAnd: "and_bits", ast.BitOr: "or_bits", ast.BitXor: "xor", ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}.get(type(node.op))
                right_node = node.right
            if op is None: raise ContractError("unsupported operator; signed comparisons use slt/sle")
            left = visit(node.left)
            right = visit(right_node, left.type if isinstance(left, Expr) else expected)
            if type(left) is int and type(right) is int and op in ("add", "sub", "mul"):
                result = {"add": lambda: left+right, "sub": lambda: left-right, "mul": lambda: left*right}[op]()
                if not 0 <= result <= 4096:
                    raise ContractError("natural parameter arithmetic must remain between 0 and 4096")
                return result
            if type(left) is int and isinstance(right, Expr): left = bits(left, right.type.width) if right.type.width else left
            if type(right) is int and isinstance(left, Expr): right = bits(right, left.type.width) if left.type.width else right
            return binary(op, self._expr(left), self._expr(right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            args = [visit(a) for a in node.args]
            if name in self.predicates:
                if name in stack: raise ContractError(f"recursive predicate expansion: {' -> '.join(stack + (name,))}")
                if len(stack) >= 64: raise ContractError("predicate expansion depth exceeds 64")
                pred = self.predicates[name]
                if len(args) != len(pred.parameters): raise ContractError(f"{name} expects {len(pred.parameters)} arguments")
                import re
                widths, bindings = {}, {}
                for (param, ann), value in zip(pred.parameters, args):
                    if ann == "nat": self._nat(value)
                    elif ann == "bool":
                        if self._expr(value).type.width is not None: raise ContractError(f"{name}.{param} expects bool")
                    else:
                        match = re.fullmatch(r"bits\[(\d+|[A-Z]\w*)\]", ann)
                        if not match: raise ContractError(f"invalid parameter type {ann}")
                        width = self._expr(value).type.width
                        key = match[1]
                        required = int(key) if key.isdigit() else widths.setdefault(key, width)
                        if width is None or width != required: raise ContractError(f"{name}.{param}: inconsistent width parameter {key}")
                    bindings[param] = value
                # No globals inside templates: all sampled dependencies are parameters.
                nested = Elaborator({}, self.predicates, self.match)
                result = nested._visit(ast.parse(pred.body, mode="eval").body, bindings, stack+(name,))
                if self._expr(result).type.width is not None: raise ContractError(f"predicate {name} must return bool")
                return result
            if name == "bv" and len(args) == 2: return bits(self._nat(args[0]), self._nat(args[1]))
            if name == "slice" and len(args) == 3: return extract(self._expr(args[0]), self._nat(args[1]), self._nat(args[2]))
            if name in ("zext", "sext") and len(args) == 2: return extend(self._expr(args[0]), self._nat(args[1]), name == "sext")
            if name == "concat": return concat(*(self._expr(a) for a in args))
            if name == "ite" and len(args) == 3: return choose(*(self._expr(a) for a in args))
            if name in ("slt", "sle") and len(args) == 2: return binary(name, *(self._expr(a) for a in args))
            if name in ("fits_u", "fits_s") and len(args) == 2:
                value, n = self._expr(args[0]), self._nat(args[1])
                if value.type.width is None or not 1 <= n <= value.type.width: raise ContractError("refinement width must be within the carrier width")
                return binary("eq", extend(extract(value, n-1, 0), value.type.width, name == "fits_s"), value)
            if name == "matches" and len(args) == 2 and self.match:
                if not isinstance(args[1], str): raise ContractError("matches requires a literal instruction/set name")
                return self.match(self._expr(args[0]), args[1])
            raise ContractError(f"unknown predicate or wrong arity: {name}")
        raise ContractError(f"unsupported expression form: {type(node).__name__}")

    @staticmethod
    def _expr(value):
        if not isinstance(value, Expr): raise ContractError("expected a typed expression; size literals with bv(value, width)")
        return value

    @staticmethod
    def _nat(value):
        if type(value) is not int or value < 0: raise ContractError("expected a nonnegative elaboration-time integer")
        return value
