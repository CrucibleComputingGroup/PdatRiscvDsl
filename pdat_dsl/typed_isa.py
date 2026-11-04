"""Exact encoding-set interpretation of CRISP include/forbid programs.

The checked profile is RV32IM. Encoding signatures are explicit and extensible;
unsupported profiles are errors, never empty predicates. Operand values belong
in sampled contracts, not in a predicate over the instruction word alone.
"""
from . import encodings
from .parser import (parse_dsl, IncludeRule, ForbidRule, InstructionPattern,
                     RegisterRangeExpression, VersionDirective, RangeValue,
                     BitPattern, DataTypeSet)
from .typed_core import (ContractError, Expr, boolean, bits, binary, logical,
                         extract)


class ISARegistry:
    def __init__(self):
        self.instructions = {}
        self.extensions = {}
        self.register_extension("RV32I", encodings.RV32I_INSTRUCTIONS)
        self.register_extension("RV32M", encodings.RV32M_INSTRUCTIONS)

    def register_extension(self, name, instructions):
        """Add a set of 32-bit encoding signatures without changing the backend.

        This validates structural layout, not faithfulness to an ISA manual.
        The registry is part of the trusted binding boundary.
        """
        name = name.upper()
        if name in self.extensions or name in self.instructions:
            raise ContractError(f"duplicate ISA name {name}")
        pending = {}
        for key, enc in instructions.items():
            key = key.upper()
            if key in pending or key in self.instructions or key in self.extensions or key == name:
                raise ContractError(f"duplicate instruction {key}")
            if enc.is_compressed:
                raise ContractError("compressed instruction signatures need a separate checked profile")
            bits(enc.base_pattern, 32)
            bits(enc.base_mask, 32)
            if enc.base_pattern & ~enc.base_mask:
                raise ContractError(f"{key}: base pattern sets unconstrained bits")
            for field, (pos, width) in enc.fields.items():
                if not 0 <= pos < 32 or not 1 <= width <= 32-pos:
                    raise ContractError(f"{key}.{field}: invalid field layout")
            pending[key] = enc
        self.instructions.update(pending)
        self.extensions[name] = tuple(pending)

    @staticmethod
    def _word(word):
        if word.type.width != 32:
            raise ContractError("RV32IM instruction encodings require bits[32]")

    def matches(self, word: Expr, name: str) -> Expr:
        self._word(word)
        name = name.upper()
        if name in self.extensions:
            return logical("or", *(self.matches(word, n) for n in self.extensions[name]))
        if name not in self.instructions:
            raise ContractError(f"unknown instruction/extension '{name}' in the checked RV32IM registry")
        enc = self.instructions[name]
        return binary("eq", binary("and_bits", word, bits(enc.base_mask, 32)), bits(enc.base_pattern, 32))

    def expression(self, word: Expr, expression) -> Expr:
        self._word(word)
        if isinstance(expression, str):
            return self.matches(word, expression)
        if isinstance(expression, RegisterRangeExpression):
            regs = expression.registers
            if not regs or any(not 0 <= r < 32 for r in regs):
                raise ContractError("register range must be nonempty and within x0..x31")
            result = []
            for name, enc in self.instructions.items():
                # System instructions have raw I-format fields but no register operands.
                if name in ("ECALL", "EBREAK", "FENCE"):
                    continue
                uses = []
                for field in ("rd", "rs1", "rs2"):
                    if field in enc.fields:
                        pos, width = enc.fields[field]
                        value = extract(word, pos+width-1, pos)
                        uses.append(logical("or", *(binary("eq", value, bits(r, width)) for r in sorted(regs))))
                result.append(logical("and", self.matches(word, name), logical("or", *uses)))
            return logical("or", *result)
        if not isinstance(expression, InstructionPattern):
            raise ContractError("unsupported ISA set expression")
        name = expression.name.upper()
        if name not in self.instructions:
            raise ContractError(f"unknown instruction '{name}'")
        enc = self.instructions[name]
        result = self.matches(word, name)
        seen = set()
        for constraint in expression.constraints:
            field, value = constraint.field_name, constraint.field_value
            if field in seen: raise ContractError(f"duplicate field {name}.{field}")
            seen.add(field)
            if isinstance(value, DataTypeSet) or field == "dtype" or field.endswith("_dtype"):
                raise ContractError("operand refinements require a sampled value contract; they are not instruction bits")
            if field not in enc.fields:
                raise ContractError(f"{name} has no encoded field '{field}'")
            pos, width = enc.fields[field]
            actual = extract(word, pos+width-1, pos)
            if value == "*": predicate = boolean(True)
            elif isinstance(value, BitPattern):
                if value.width != width:
                    raise ContractError(f"{name}.{field} expects bits[{width}], got pattern width {value.width}")
                pattern, mask = value.to_pattern_mask()
                predicate = binary("eq", binary("and_bits", actual, bits(mask, width)), bits(pattern, width))
            elif isinstance(value, RangeValue):
                if value.min_val > value.max_val: raise ContractError("descending range")
                predicate = logical("and", binary("ule", bits(value.min_val, width), actual), binary("ule", actual, bits(value.max_val, width)))
            else:
                if isinstance(value, str):
                    if field not in ("rd", "rs1", "rs2"):
                        raise ContractError(f"register names cannot constrain {name}.{field}")
                    value = encodings.parse_register(value)
                predicate = binary("eq", actual, bits(value, width))
            result = logical("and", result, predicate)
        return result

    def compile(self, source: str, word: Expr) -> Expr:
        try:
            program = parse_dsl(source)
        except (SyntaxError, ValueError) as e:
            raise ContractError(str(e)) from e
        if program.version != 2:
            raise ContractError("ISA policy imports require explicit version 2")
        allowed = boolean(False)
        for rule in program.rules:
            if isinstance(rule, VersionDirective): continue
            if not isinstance(rule, (IncludeRule, ForbidRule)):
                raise ContractError(f"line {rule.line}: configuration directives must be expressed as typed contracts")
            try:
                part = self.expression(word, rule.expr)
                allowed = logical("or", allowed, part) if isinstance(rule, IncludeRule) else logical("and", allowed, logical("not", part))
            except ContractError as e:
                raise ContractError(f"line {rule.line}: {e}") from e
        return allowed
