#!/usr/bin/env python3
"""
Parser for the instruction outlawing DSL.

Grammar:
    program = { rule }
    rule = require_rule | register_constraint_rule | instruction_rule | pattern_rule | comment
    require_rule = "require" extension_name
    register_constraint_rule = "require_registers" register_range
    instruction_rule = "instruction" identifier [ field_constraints ]
    pattern_rule = "pattern" hex_value "mask" hex_value [ description ]
    register_range = register_name "-" register_name | number "-" number
    field_constraints = "{" field_constraint { "," field_constraint } "}"
    field_constraint = field_name "=" field_value
    field_value = wildcard | number | register_name
"""

import re
import sys
from dataclasses import dataclass, field as dataclass_field
from typing import List, Optional, Dict, Union, Set
from enum import Enum

# ============================================================================
# Token Types
# ============================================================================

class TokenType(Enum):
    # Keywords
    VERSION = "version"
    REQUIRE = "require"
    REQUIRE_REGISTERS = "require_registers"
    REQUIRE_PC_BITS = "require_pc_bits"
    INSTRUCTION = "instruction"
    ALLOW = "allow"
    INCLUDE = "include"
    FORBID = "forbid"
    PATTERN = "pattern"
    MASK = "mask"

    # Timing parameters
    INSTR_HIT_LATENCY = "instr_hit_latency"
    INSTR_MISS_LATENCY = "instr_miss_latency"
    DATA_HIT_LATENCY = "data_hit_latency"
    DATA_MISS_LATENCY = "data_miss_latency"
    LOCALITY_BITS = "locality_bits"

    # Literals
    IDENTIFIER = "identifier"
    NUMBER = "number"
    WILDCARD = "wildcard"
    DTYPE = "dtype"  # Data type token (i8, u16, etc.)
    BIT_PATTERN = "bit_pattern"  # Bit pattern like 5'b00xxx

    # Symbols
    LBRACE = "{"
    RBRACE = "}"
    LPAREN = "("
    RPAREN = ")"
    COMMA = ","
    EQUALS = "="
    DASH = "-"
    PIPE = "|"
    TILDE = "~"
    APOSTROPHE = "'"
    IN = "in"

    # Other
    COMMENT = "comment"
    NEWLINE = "newline"
    EOF = "eof"

@dataclass
class Token:
    type: TokenType
    value: any
    line: int
    column: int

# ============================================================================
# AST Nodes
# ============================================================================

@dataclass(frozen=True)
class DataType:
    """Represents a data type specification (e.g., i8, u16, i32 | u32)"""
    width: int  # Bit width: 8, 16, 32, or 64
    signed: bool  # True for signed (i*), False for unsigned (u*)

    def __str__(self):
        prefix = 'i' if self.signed else 'u'
        return f"{prefix}{self.width}"

    def __repr__(self):
        return f"DataType({self})"

    @staticmethod
    def from_string(s: str) -> 'DataType':
        """Parse a data type string like 'i8', 'u16', etc."""
        if not s or len(s) < 2:
            raise ValueError(f"Invalid data type: {s}")

        prefix = s[0].lower()
        if prefix not in ('i', 'u'):
            raise ValueError(f"Invalid data type prefix: {prefix} (expected 'i' or 'u')")

        try:
            width = int(s[1:])
        except ValueError:
            raise ValueError(f"Invalid data type width: {s[1:]}")

        if width not in (8, 16, 32, 64):
            raise ValueError(f"Invalid data type width: {width} (expected 8, 16, 32, or 64)")

        return DataType(width=width, signed=(prefix == 'i'))

@dataclass
class DataTypeSet:
    """Represents a type expression: a set of data types with optional negation prefix.

    Syntax:
    - Simple: i8 | u8 | i16
    - Negated: ~(i16 | u16) or ~i8

    Note: The negation (~) is NOT a general operator - it's a type expression prefix
    that can only appear once at the very beginning, not nested or repeated.
    This is a domain-specific construct, not standard boolean negation.

    Semantics (consistent with instruction outlawing):
    - Without ~: "outlaw instructions using ANY of these types"
      Example: i8 | u8 means "forbid when using i8 OR u8"

    - With ~: "outlaw instructions using NONE of these types" = "allow ONLY these types"
      Example: ~(i16 | u16) means "forbid when NOT using i16 or u16" = "require i16 or u16"

    The negation inverts the constraint, allowing positive specifications within
    a negative (outlawing) framework.

    Validation Rules:
    For a given signedness, you can only forbid narrower types if you also forbid wider types.
    This is because wider types contain all values of narrower types (e.g., u16 contains u8).

    Valid:   forbid u8, u16 (allow only u32+)
    Invalid: forbid u8, allow u16 (u16 contains u8 values!)
    Valid:   ~(u8 | u16) = allow only u8 or u16
    Invalid: ~(u16 | u32) = allow u16 or u32 (missing u8 creates gap!)
    """
    types: Set[DataType] = dataclass_field(default_factory=set)
    negated: bool = False  # If True, entire expression is negated (~prefix)

    def add(self, dtype: DataType):
        """Add a data type to the set"""
        self.types.add(dtype)

    def validate(self) -> Optional[str]:
        """
        Validate that the dtype set is logically consistent.

        Returns None if valid, error message string if invalid.

        Rules:
        - For forbid (not negated): can skip types, but forbidden types must not create gaps
        - For allow (negated): must not have gaps in the allowed range

        The key insight: you cannot distinguish u8 from u16 when value < 256 by bit patterns.
        So forbidding u8 while allowing u16 is unenforceable.
        """
        if len(self.types) == 0:
            return "DataTypeSet cannot be empty"

        # Separate by signedness
        signed_types = sorted([t for t in self.types if t.signed], key=lambda t: t.width)
        unsigned_types = sorted([t for t in self.types if not t.signed], key=lambda t: t.width)

        # Check each signedness category
        for type_list, sign_name in [(signed_types, "signed"), (unsigned_types, "unsigned")]:
            if len(type_list) == 0:
                continue

            widths = [t.width for t in type_list]
            # Get prefix for error messages (i or u)
            prefix = 'i' if type_list[0].signed else 'u'

            if self.negated:
                # ALLOW only these types: must not have gaps at the narrow end
                # Valid:   ~(u8 | u16) = allow u8 or u16
                # Invalid: ~(u16 | u32) = allow u16 or u32 (missing u8 - can't distinguish!)
                #
                # Rule: if you allow a width W, you must allow all widths < W of same signedness
                # Check: must include all widths from 8 up to min width continuously
                min_width = min(widths)
                for w in [8, 16, 32]:  # Check 8, 16, 32 (64 would be max)
                    if w < min_width:
                        # We're missing a narrower type
                        return f"Invalid {sign_name} type set {self}: allowing {prefix}{min_width} requires allowing all narrower types (missing {prefix}{w})"
                    if w > min_width:
                        break
                # Now check no gaps within the allowed range
                for t in type_list:
                    for w in [8, 16, 32, 64]:
                        if w >= t.width:
                            break
                        if w not in widths:
                            return f"Invalid {sign_name} type set {self}: gap in allowed types (missing {prefix}{w} between allowed types)"
            else:
                # FORBID these types: must not have gaps at the wide end
                # Valid:   u16 | u32 = forbid u16 and u32
                # Invalid: u8 | u32 = forbid u8 and u32 (missing u16 - creates ambiguity!)
                #
                # Rule: if you forbid a width W, you must forbid all widths > W of same signedness
                max_width = max(widths)
                for t in type_list:
                    # Check all wider widths are present
                    for w in [8, 16, 32, 64]:
                        if w <= t.width:
                            continue
                        if w not in widths and w <= max_width:
                            return f"Invalid {sign_name} type set {self}: forbidding {prefix}{t.width} requires forbidding all wider types (missing {prefix}{w})"

        return None

    def __str__(self):
        type_str = " | ".join(str(t) for t in sorted(self.types, key=lambda t: (t.width, not t.signed)))
        if self.negated:
            # Add parentheses if multiple types
            if len(self.types) > 1:
                return f"~({type_str})"
            else:
                return f"~{type_str}"
        return type_str

    def __repr__(self):
        return f"DataTypeSet({self})"

@dataclass
class RangeValue:
    """Represents a range constraint like x0-x15 or 0-15

    Syntax: <start>-<end>
    - start: Register name or number (e.g., x0, 0)
    - end: Register name or number (e.g., x15, 15)

    Example: x0-x15 represents register range [0, 15]
    """
    min_val: int  # Minimum value (inclusive)
    max_val: int  # Maximum value (inclusive)

    def __str__(self):
        return f"x{self.min_val}-x{self.max_val}"

    def __repr__(self):
        return f"RangeValue({self.min_val}, {self.max_val})"

@dataclass
class BitPattern:
    """Represents a bit pattern constraint like 5'b00xxx

    Syntax: N'b[01xX_]+
    - N: bit width (e.g., 5, 12)
    - Pattern: string of 0, 1, x, X, or _ characters

    Example: 5'b00xxx matches values 0-7 (bits [4:2] = 000, bits [1:0] = don't care)
    """
    width: int               # Number of bits (e.g., 5 for RV32 shift amounts)
    pattern: str             # Pattern string (e.g., "00xxx")

    def to_pattern_mask(self) -> tuple[int, int]:
        """
        Convert bit pattern to (pattern, mask) pair for matching.

        Returns:
            (pattern_val, mask_val) where:
            - pattern_val: the bit values where pattern has 0/1
            - mask_val: which bits to check (1 = check, 0 = don't care)

        Example: 5'b00xxx
        - pattern_val: 0b00000 (0x00)
        - mask_val:    0b11000 (0x18) - only check bits where pattern has 0/1

        Matching logic: (value & mask) == (pattern & mask)
        """
        if len(self.pattern) != self.width:
            raise ValueError(f"Pattern length {len(self.pattern)} doesn't match width {self.width}")

        pattern_val = 0
        mask_val = 0

        # Process pattern from left to right (MSB to LSB)
        for i, bit in enumerate(self.pattern):
            bit_pos = self.width - 1 - i  # Position from LSB

            if bit == '1':
                pattern_val |= (1 << bit_pos)
                mask_val |= (1 << bit_pos)
            elif bit == '0':
                mask_val |= (1 << bit_pos)
            elif bit in ('x', 'X', '_'):
                pass  # Don't care - leave mask bit = 0
            else:
                raise ValueError(f"Invalid bit pattern character: '{bit}' (expected 0, 1, x, X, or _)")

        return (pattern_val, mask_val)

    def __str__(self):
        return f"{self.width}'b{self.pattern}"

    def __repr__(self):
        return f"BitPattern({self})"

@dataclass
class FieldConstraint:
    """Represents a field constraint like 'rd = x5' or 'rd in x0-x15' or 'dtype = i8 | u8' or 'imm = 5'b00xxx'"""
    field_name: str
    field_value: Union[str, int, DataTypeSet, BitPattern, RangeValue]  # Can be wildcard "*", register "x5", number, data type set, bit pattern, or range

@dataclass
class VersionDirective:
    """Version directive like 'version 2' specifying DSL version"""
    version: int
    line: int

@dataclass
class RequireRule:
    """Require directive like 'require RV32I' specifying valid instruction extensions (v1 only)"""
    extension: str
    line: int

@dataclass
class IncludeRule:
    """Include directive like 'include RV32I' or 'include SLLI {shamt = 5'b000xx}' (v2)"""
    expr: Union[str, 'InstructionPattern']  # Extension name or instruction pattern
    line: int

@dataclass
class ForbidRule:
    """Forbid directive like 'forbid MUL' or 'forbid SLLI {shamt = 5'b000xx}' (v2)"""
    expr: Union[str, 'InstructionPattern']  # Extension name or instruction pattern
    line: int

@dataclass
class InstructionPattern:
    """Instruction pattern for v2: name + constraints (used in IncludeRule/ForbidRule)"""
    name: str
    constraints: List[FieldConstraint]

@dataclass
class RegisterConstraintRule:
    """Register constraint like 'require_registers x0-x16' limiting which registers can be used"""
    min_reg: int  # Minimum register number (inclusive)
    max_reg: int  # Maximum register number (inclusive)
    line: int

@dataclass
class PcConstraintRule:
    """PC constraint like 'require_pc_bits 16' limiting PC to N address bits"""
    pc_bits: int  # Number of PC bits (e.g., 16 for 64KB address space)
    line: int

@dataclass
class InstructionRule:
    """High-level instruction rule like 'instruction MUL { rd = x0 }' or 'allow instruction SLLI { imm = 5'b00xxx }'"""
    name: str
    constraints: List[FieldConstraint]
    line: int
    allow: bool = False  # True if this is an "allow" rule (positive constraint)

@dataclass
class PatternRule:
    """Low-level pattern rule like 'pattern 0x02000033 mask 0xFE00707F'"""
    pattern: int
    mask: int
    description: Optional[str]
    line: int

@dataclass
class TimingConstraintRule:
    """Timing constraint rule like 'instr_hit_latency 1'"""
    param_name: str
    value: int
    line: int = 0

@dataclass
class Program:
    """Root AST node containing all rules"""
    version: Optional[int]  # DSL version (1 or 2), None means v1 (default)
    rules: List[Union[VersionDirective, RequireRule, RegisterConstraintRule, PcConstraintRule, InstructionRule, PatternRule, TimingConstraintRule, IncludeRule, ForbidRule]]

# ============================================================================
# Lexer
# ============================================================================

class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []

    def error(self, msg: str):
        raise SyntaxError(f"Lexer error at line {self.line}, column {self.column}: {msg}")

    def peek(self, offset=0):
        pos = self.pos + offset
        if pos < len(self.text):
            return self.text[pos]
        return None

    def advance(self):
        if self.pos < len(self.text):
            if self.text[self.pos] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.pos += 1

    def skip_whitespace(self):
        while self.peek() and self.peek() in ' \t\r':
            self.advance()

    def read_comment(self):
        """Read a comment starting with #"""
        start_line = self.line
        start_col = self.column
        self.advance()  # skip #

        comment = ""
        while self.peek() and self.peek() != '\n':
            comment += self.peek()
            self.advance()

        return Token(TokenType.COMMENT, comment.strip(), start_line, start_col)

    def read_number(self):
        """Read a number (hex, binary, or decimal)"""
        start_line = self.line
        start_col = self.column

        num_str = ""

        # Check for hex (0x) or binary (0b)
        if self.peek() == '0' and self.peek(1) in 'xXbB':
            num_str += self.peek()
            self.advance()
            num_str += self.peek()
            self.advance()

            if num_str[1] in 'xX':
                # Hex
                while self.peek() and self.peek() in '0123456789abcdefABCDEF_':
                    if self.peek() != '_':
                        num_str += self.peek()
                    self.advance()
                value = int(num_str, 16)
            else:
                # Binary
                while self.peek() and self.peek() in '01_':
                    if self.peek() != '_':
                        num_str += self.peek()
                    self.advance()
                value = int(num_str, 2)
        else:
            # Decimal
            while self.peek() and self.peek() in '0123456789_':
                if self.peek() != '_':
                    num_str += self.peek()
                self.advance()
            value = int(num_str)

        return Token(TokenType.NUMBER, value, start_line, start_col)

    def read_bit_pattern(self):
        """Read a bit pattern like 5'b00xxx or 12'b0000_xxxx_xxxx

        Format: <width>'b<pattern>
        - width: decimal number (1-64)
        - pattern: string of [01xX_]+ characters
        """
        start_line = self.line
        start_col = self.column

        # Read width (already started with digit)
        width_str = ""
        while self.peek() and self.peek().isdigit():
            width_str += self.peek()
            self.advance()

        try:
            width = int(width_str)
        except ValueError:
            self.error(f"Invalid bit pattern width: {width_str}")

        # Expect apostrophe
        if self.peek() != "'":
            self.error(f"Expected ' after width in bit pattern, got {self.peek()}")
        self.advance()

        # Expect 'b'
        if self.peek() not in ('b', 'B'):
            self.error(f"Expected 'b' after ' in bit pattern, got {self.peek()}")
        self.advance()

        # Read pattern (0, 1, x, X, _)
        pattern = ""
        while self.peek() and self.peek() in '01xX_':
            if self.peek() != '_':  # Underscores are for readability, skip them
                pattern += self.peek()
            self.advance()

        if not pattern:
            self.error("Bit pattern cannot be empty")

        # Validate pattern length matches width
        if len(pattern) != width:
            self.error(f"Bit pattern length {len(pattern)} doesn't match declared width {width}")

        # Return as tuple (width, pattern)
        return Token(TokenType.BIT_PATTERN, (width, pattern), start_line, start_col)

    def read_identifier(self):
        """Read an identifier or keyword"""
        start_line = self.line
        start_col = self.column

        ident = ""
        while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
            ident += self.peek()
            self.advance()

        # Check if it's a keyword
        if ident == "version":
            return Token(TokenType.VERSION, ident, start_line, start_col)
        elif ident == "require":
            return Token(TokenType.REQUIRE, ident, start_line, start_col)
        elif ident == "require_registers":
            return Token(TokenType.REQUIRE_REGISTERS, ident, start_line, start_col)
        elif ident == "require_pc_bits":
            return Token(TokenType.REQUIRE_PC_BITS, ident, start_line, start_col)
        elif ident == "instruction":
            return Token(TokenType.INSTRUCTION, ident, start_line, start_col)
        elif ident == "allow":
            return Token(TokenType.ALLOW, ident, start_line, start_col)
        elif ident == "include":
            return Token(TokenType.INCLUDE, ident, start_line, start_col)
        elif ident == "forbid":
            return Token(TokenType.FORBID, ident, start_line, start_col)
        elif ident == "in":
            return Token(TokenType.IN, ident, start_line, start_col)
        elif ident == "pattern":
            return Token(TokenType.PATTERN, ident, start_line, start_col)
        elif ident == "mask":
            return Token(TokenType.MASK, ident, start_line, start_col)
        elif ident == "instr_hit_latency":
            return Token(TokenType.INSTR_HIT_LATENCY, ident, start_line, start_col)
        elif ident == "instr_miss_latency":
            return Token(TokenType.INSTR_MISS_LATENCY, ident, start_line, start_col)
        elif ident == "data_hit_latency":
            return Token(TokenType.DATA_HIT_LATENCY, ident, start_line, start_col)
        elif ident == "data_miss_latency":
            return Token(TokenType.DATA_MISS_LATENCY, ident, start_line, start_col)
        elif ident == "locality_bits":
            return Token(TokenType.LOCALITY_BITS, ident, start_line, start_col)
        # Check if it's a data type (i8, u16, i32, u64, etc.)
        elif self._is_data_type(ident):
            return Token(TokenType.DTYPE, ident, start_line, start_col)
        else:
            return Token(TokenType.IDENTIFIER, ident, start_line, start_col)

    def _is_data_type(self, ident: str) -> bool:
        """Check if identifier is a data type like i8, u16, i32, u64"""
        if len(ident) < 2:
            return False
        if ident[0] not in ('i', 'u'):
            return False
        try:
            width = int(ident[1:])
            return width in (8, 16, 32, 64)
        except ValueError:
            return False

    def tokenize(self) -> List[Token]:
        """Tokenize the entire input"""
        tokens = []

        while self.pos < len(self.text):
            self.skip_whitespace()

            if not self.peek():
                break

            ch = self.peek()

            # Comments
            if ch == '#':
                tokens.append(self.read_comment())
                continue

            # Newlines
            if ch == '\n':
                line = self.line
                col = self.column
                self.advance()
                tokens.append(Token(TokenType.NEWLINE, '\n', line, col))
                continue

            # Single character tokens
            if ch == '{':
                tokens.append(Token(TokenType.LBRACE, ch, self.line, self.column))
                self.advance()
                continue

            if ch == '}':
                tokens.append(Token(TokenType.RBRACE, ch, self.line, self.column))
                self.advance()
                continue

            if ch == '(':
                tokens.append(Token(TokenType.LPAREN, ch, self.line, self.column))
                self.advance()
                continue

            if ch == ')':
                tokens.append(Token(TokenType.RPAREN, ch, self.line, self.column))
                self.advance()
                continue

            if ch == ',':
                tokens.append(Token(TokenType.COMMA, ch, self.line, self.column))
                self.advance()
                continue

            if ch == '=':
                tokens.append(Token(TokenType.EQUALS, ch, self.line, self.column))
                self.advance()
                continue

            if ch == '-':
                tokens.append(Token(TokenType.DASH, ch, self.line, self.column))
                self.advance()
                continue

            if ch == '|':
                tokens.append(Token(TokenType.PIPE, ch, self.line, self.column))
                self.advance()
                continue

            if ch == '~':
                tokens.append(Token(TokenType.TILDE, ch, self.line, self.column))
                self.advance()
                continue

            # Wildcards
            if ch in '*x_' and (not self.peek(1) or not self.peek(1).isalnum()):
                tokens.append(Token(TokenType.WILDCARD, ch, self.line, self.column))
                self.advance()
                continue

            # Numbers or bit patterns (N'bXXX)
            if ch.isdigit():
                # Look ahead to check if this is a bit pattern
                # Bit pattern format: N'b[01xX_]+
                temp_pos = self.pos
                temp_line = self.line
                temp_col = self.column

                # Scan ahead for digits
                while temp_pos < len(self.text) and self.text[temp_pos].isdigit():
                    temp_pos += 1

                # Check if followed by 'b or 'B
                if temp_pos + 1 < len(self.text) and self.text[temp_pos] == "'" and self.text[temp_pos + 1] in 'bB':
                    # This is a bit pattern
                    tokens.append(self.read_bit_pattern())
                else:
                    # Regular number
                    tokens.append(self.read_number())
                continue

            # Identifiers and keywords
            if ch.isalpha() or ch == '_':
                tokens.append(self.read_identifier())
                continue

            self.error(f"Unexpected character: {ch}")

        tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return tokens

# ============================================================================
# Parser
# ============================================================================

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = [t for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.COMMENT)]
        self.pos = 0

    def error(self, msg: str):
        if self.pos < len(self.tokens):
            tok = self.tokens[self.pos]
            raise SyntaxError(f"Parser error at line {tok.line}, column {tok.column}: {msg}")
        else:
            raise SyntaxError(f"Parser error at end of file: {msg}")

    def peek(self, offset=0) -> Optional[Token]:
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def advance(self) -> Token:
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, token_type: TokenType) -> Token:
        tok = self.peek()
        if not tok or tok.type != token_type:
            self.error(f"Expected {token_type}, got {tok.type if tok else 'EOF'}")
        return self.advance()

    def parse(self) -> Program:
        """Parse the entire program"""
        rules = []
        version = None

        while self.peek() and self.peek().type != TokenType.EOF:
            rule = self.parse_rule()
            if rule:
                # Extract version if present (must be first directive)
                if isinstance(rule, VersionDirective):
                    if version is not None:
                        self.error(f"Multiple version directives found (line {rule.line})")
                    if len(rules) > 0:
                        self.error(f"Version directive must be the first statement (line {rule.line})")
                    version = rule.version
                    # Don't add VersionDirective to rules list
                    continue
                rules.append(rule)

        return Program(version, rules)

    def parse_rule(self) -> Optional[Union[RequireRule, RegisterConstraintRule, PcConstraintRule, InstructionRule, PatternRule,TimingConstraintRule]]:
        """Parse a single rule"""
        tok = self.peek()

        if not tok or tok.type == TokenType.EOF:
            return None

        if tok.type == TokenType.VERSION:
            return self.parse_version_directive()
        elif tok.type == TokenType.INCLUDE:
            return self.parse_include_rule()
        elif tok.type == TokenType.FORBID:
            return self.parse_forbid_rule()
        elif tok.type == TokenType.REQUIRE:
            return self.parse_require_rule()
        elif tok.type == TokenType.REQUIRE_REGISTERS:
            return self.parse_register_constraint_rule()
        elif tok.type == TokenType.REQUIRE_PC_BITS:
            return self.parse_pc_constraint_rule()
        elif tok.type == TokenType.ALLOW:
            return self.parse_instruction_rule(allow=True)
        elif tok.type == TokenType.INSTRUCTION:
            return self.parse_instruction_rule(allow=False)
        elif tok.type == TokenType.PATTERN:
            return self.parse_pattern_rule()
        elif tok.type in (TokenType.INSTR_HIT_LATENCY, TokenType.INSTR_MISS_LATENCY,
                         TokenType.DATA_HIT_LATENCY, TokenType.DATA_MISS_LATENCY,
                         TokenType.LOCALITY_BITS):
            return self.parse_timing_parameter_rule()
        else:
            self.error(f"Expected 'version', 'include', 'forbid', 'require', 'require_registers', 'require_pc_bits', 'allow', 'instruction', 'timing parameter' or 'pattern', got {tok.type}")

    def parse_require_rule(self) -> RequireRule:
        """Parse: require IDENTIFIER"""
        require_tok = self.expect(TokenType.REQUIRE)
        extension_tok = self.expect(TokenType.IDENTIFIER)

        return RequireRule(extension_tok.value, require_tok.line)

    def parse_register_constraint_rule(self) -> RegisterConstraintRule:
        """Parse: require_registers x0-x16 or require_registers 0-16"""
        require_reg_tok = self.expect(TokenType.REQUIRE_REGISTERS)

        # Parse start register (can be identifier like "x0" or number like "0")
        start_tok = self.peek()
        if start_tok.type == TokenType.IDENTIFIER:
            self.advance()
            # Parse register name like "x0", "x10", etc.
            reg_str = start_tok.value.lower()
            if reg_str.startswith('x'):
                try:
                    min_reg = int(reg_str[1:])
                except ValueError:
                    self.error(f"Invalid register name: {start_tok.value}")
            else:
                self.error(f"Expected register name like 'x0' or number, got {start_tok.value}")
        elif start_tok.type == TokenType.NUMBER:
            self.advance()
            min_reg = start_tok.value
        else:
            self.error(f"Expected register name or number, got {start_tok.type}")

        # Expect dash
        self.expect(TokenType.DASH)

        # Parse end register
        end_tok = self.peek()
        if end_tok.type == TokenType.IDENTIFIER:
            self.advance()
            reg_str = end_tok.value.lower()
            if reg_str.startswith('x'):
                try:
                    max_reg = int(reg_str[1:])
                except ValueError:
                    self.error(f"Invalid register name: {end_tok.value}")
            else:
                self.error(f"Expected register name like 'x16' or number, got {end_tok.value}")
        elif end_tok.type == TokenType.NUMBER:
            self.advance()
            max_reg = end_tok.value
        else:
            self.error(f"Expected register name or number, got {end_tok.type}")

        # Validate range
        if min_reg < 0 or min_reg > 31:
            self.error(f"Register number {min_reg} out of range (0-31)")
        if max_reg < 0 or max_reg > 31:
            self.error(f"Register number {max_reg} out of range (0-31)")
        if min_reg > max_reg:
            self.error(f"Invalid register range: {min_reg}-{max_reg} (min > max)")

        return RegisterConstraintRule(min_reg, max_reg, require_reg_tok.line)

    def parse_pc_constraint_rule(self) -> PcConstraintRule:
        """Parse: require_pc_bits NUMBER"""
        require_pc_tok = self.expect(TokenType.REQUIRE_PC_BITS)
        pc_bits_tok = self.expect(TokenType.NUMBER)

        pc_bits = pc_bits_tok.value

        # Validate pc_bits is reasonable (1-32 for RV32)
        if pc_bits < 1 or pc_bits > 32:
            self.error(f"PC bits {pc_bits} out of range (1-32)")

        return PcConstraintRule(pc_bits, require_pc_tok.line)

    def parse_instruction_rule(self, allow: bool = False) -> InstructionRule:
        """Parse: [allow] instruction IDENTIFIER [ field_constraints ]"""
        if allow:
            allow_tok = self.expect(TokenType.ALLOW)
            instr_tok = self.expect(TokenType.INSTRUCTION)
            line = allow_tok.line
        else:
            instr_tok = self.expect(TokenType.INSTRUCTION)
            line = instr_tok.line

        name_tok = self.expect(TokenType.IDENTIFIER)

        constraints = []
        if self.peek() and self.peek().type == TokenType.LBRACE:
            constraints = self.parse_field_constraints()

        return InstructionRule(name_tok.value, constraints, line, allow=allow)

    def parse_pattern_rule(self) -> PatternRule:
        """Parse: pattern NUMBER mask NUMBER [ COMMENT ]"""
        pattern_tok = self.expect(TokenType.PATTERN)
        pattern_num = self.expect(TokenType.NUMBER)
        self.expect(TokenType.MASK)
        mask_num = self.expect(TokenType.NUMBER)

        # Description would have been captured as a comment token (already filtered out)
        # We can't easily get it here, so leave as None for now

        return PatternRule(pattern_num.value, mask_num.value, None, pattern_tok.line)

    def parse_timing_parameter_rule(self) -> TimingConstraintRule:
        """Parse: instr_hit_latency 1"""
        param_tok = self.advance()
        value_tok = self.expect(TokenType.NUMBER)
        
        return TimingConstraintRule(
            param_name=param_tok.value,
            value=value_tok.value,
            line=param_tok.line
        )

    def parse_field_constraints(self) -> List[FieldConstraint]:
        """Parse: { field_constraint , field_constraint , ... }"""
        self.expect(TokenType.LBRACE)

        constraints = []

        # First constraint
        constraints.append(self.parse_field_constraint())

        # Additional constraints
        while self.peek() and self.peek().type == TokenType.COMMA:
            self.advance()  # skip comma
            constraints.append(self.parse_field_constraint())

        self.expect(TokenType.RBRACE)

        return constraints

    def parse_field_constraint(self) -> FieldConstraint:
        """Parse: field_name = field_value or field_name in range

        field_value can be:
        - wildcard: *, x, _
        - number: 0x33, 42
        - register: x5 (as identifier)
        - data type: i8, u16
        - data type set: i8 | u8 | i16
        - negated type: ~i16 or ~(i16 | u16)
        - bit pattern: 5'b00xxx, 12'b0000_xxxx_xxxx
        - range: x0-x15, 0-15 (with 'in' keyword)
        """
        field_name = self.expect(TokenType.IDENTIFIER)

        # Check for 'in' keyword (range constraint)
        if self.peek() and self.peek().type == TokenType.IN:
            self.advance()  # consume 'in'
            range_value = self.parse_range_value()
            return FieldConstraint(field_name.value, range_value)

        # Otherwise expect '=' for regular constraint
        self.expect(TokenType.EQUALS)

        # Parse field value (wildcard, number, identifier, data type set, or bit pattern)
        value_tok = self.peek()

        if value_tok.type == TokenType.WILDCARD:
            value = self.advance().value
        elif value_tok.type == TokenType.NUMBER:
            value = self.advance().value
        elif value_tok.type == TokenType.BIT_PATTERN:
            # Bit pattern: token value is (width, pattern) tuple
            bit_tok = self.advance()
            width, pattern = bit_tok.value
            value = BitPattern(width=width, pattern=pattern)
        elif value_tok.type == TokenType.DTYPE or value_tok.type == TokenType.TILDE:
            # Parse data type or data type set (i8 | u8 | i16 | ~i16 | ~(i16 | u16))
            value = self.parse_data_type_set()
        elif value_tok.type == TokenType.IDENTIFIER:
            # Check if this looks like a dtype but isn't valid (e.g., i7, u3)
            # This provides better error messages for invalid data types
            if self._looks_like_invalid_dtype(value_tok.value):
                self.error(f"Invalid data type '{value_tok.value}'. Valid types are: i8, u8, i16, u16, i32, u32, i64, u64")
            value = self.advance().value
        else:
            self.error(f"Expected field value, got {value_tok.type}")

        return FieldConstraint(field_name.value, value)

    def parse_range_value(self) -> RangeValue:
        """Parse range syntax: x0-x15 or 0-15

        Expects: (register_name|number) - (register_name|number)
        """
        # Parse start of range
        start_tok = self.peek()
        if start_tok.type == TokenType.IDENTIFIER:
            # Register name like "x0"
            self.advance()
            from .encodings import parse_register
            min_val = parse_register(start_tok.value)
            if min_val is None:
                self.error(f"Invalid register name: {start_tok.value}")
        elif start_tok.type == TokenType.NUMBER:
            self.advance()
            min_val = start_tok.value
        else:
            self.error(f"Expected register name or number in range, got {start_tok.type}")

        # Expect dash
        self.expect(TokenType.DASH)

        # Parse end of range
        end_tok = self.peek()
        if end_tok.type == TokenType.IDENTIFIER:
            # Register name like "x15"
            self.advance()
            from .encodings import parse_register
            max_val = parse_register(end_tok.value)
            if max_val is None:
                self.error(f"Invalid register name: {end_tok.value}")
        elif end_tok.type == TokenType.NUMBER:
            self.advance()
            max_val = end_tok.value
        else:
            self.error(f"Expected register name or number in range, got {end_tok.type}")

        # Validate range
        if min_val > max_val:
            self.error(f"Invalid range: {min_val}-{max_val} (min > max)")

        return RangeValue(min_val, max_val)

    def _looks_like_invalid_dtype(self, ident: str) -> bool:
        """Check if identifier looks like a data type but isn't valid"""
        if len(ident) < 2:
            return False
        if ident[0] not in ('i', 'u'):
            return False
        # It starts with i or u, check if rest looks like a number
        try:
            int(ident[1:])
            return True  # Looks like a dtype (e.g., i7, u3)
        except ValueError:
            return False

    def parse_data_type_set(self) -> DataTypeSet:
        """Parse a type expression: data types with optional negation prefix.

        Grammar:
        type_expr = ["~"] ["("] type_list [")"]
        type_list = DTYPE { "|" DTYPE }

        Valid examples:
        - i8
        - i8 | u8
        - ~i16
        - ~(i16 | u16)

        Invalid examples:
        - ~~i8           (double negation not supported)
        - i8 | ~u8       (negation only at start)
        - ~i8 | ~u8      (negation applies to whole expression)

        Note: The ~ is a type expression prefix, not a composable operator.
        It can only appear once at the very beginning of the expression.

        Semantics:
        - Without ~: "outlaw these types" (i8 | u8 = forbid i8 or u8)
        - With ~: "outlaw everything except these" (~(i16 | u16) = allow only i16 or u16)
        """
        negated = False
        dtype_set = DataTypeSet()

        # Check for negation operator
        if self.peek() and self.peek().type == TokenType.TILDE:
            self.advance()  # consume ~
            negated = True

            # Check if parentheses follow (optional for single type, required for multiple)
            has_parens = self.peek() and self.peek().type == TokenType.LPAREN
            if has_parens:
                self.advance()  # consume (

        # Parse first data type
        dtype_tok = self.expect(TokenType.DTYPE)
        try:
            dtype = DataType.from_string(dtype_tok.value)
            dtype_set.add(dtype)
        except ValueError as e:
            self.error(f"Invalid data type '{dtype_tok.value}': {e}")

        # Parse additional data types separated by |
        while self.peek() and self.peek().type == TokenType.PIPE:
            self.advance()  # skip pipe
            dtype_tok = self.expect(TokenType.DTYPE)
            try:
                dtype = DataType.from_string(dtype_tok.value)
                dtype_set.add(dtype)
            except ValueError as e:
                self.error(f"Invalid data type '{dtype_tok.value}': {e}")

        # If we had opening paren for negation, expect closing paren
        if negated and 'has_parens' in locals() and has_parens:
            self.expect(TokenType.RPAREN)

        dtype_set.negated = negated

        # Validate the dtype set
        validation_error = dtype_set.validate()
        if validation_error:
            self.error(validation_error)

        return dtype_set

    def parse_version_directive(self) -> VersionDirective:
        """Parse: version NUMBER"""
        version_tok = self.expect(TokenType.VERSION)
        num_tok = self.expect(TokenType.NUMBER)

        version = num_tok.value
        if version not in (1, 2):
            self.error(f"Unsupported DSL version: {version} (supported: 1, 2)")

        return VersionDirective(version, version_tok.line)

    def parse_include_rule(self) -> IncludeRule:
        """Parse: include EXTENSION or include INSTRUCTION [constraints] or include * {constraints}"""
        include_tok = self.expect(TokenType.INCLUDE)

        # Check for wildcard
        if self.peek() and self.peek().type == TokenType.WILDCARD:
            name_tok = self.advance()
            # Wildcard must have constraints
            if self.peek() and self.peek().type == TokenType.LBRACE:
                constraints = self.parse_field_constraints()
                expr = InstructionPattern("*", constraints)
            else:
                self.error("Wildcard '*' must have field constraints")
        else:
            name_tok = self.expect(TokenType.IDENTIFIER)

            # Check if there are constraints
            if self.peek() and self.peek().type == TokenType.LBRACE:
                # This is an instruction pattern with constraints
                constraints = self.parse_field_constraints()
                expr = InstructionPattern(name_tok.value, constraints)
            else:
                # This is just an extension or instruction name
                expr = name_tok.value

        return IncludeRule(expr, include_tok.line)

    def parse_forbid_rule(self) -> ForbidRule:
        """Parse: forbid EXTENSION or forbid INSTRUCTION [constraints] or forbid * {constraints}"""
        forbid_tok = self.expect(TokenType.FORBID)

        # Check for wildcard
        if self.peek() and self.peek().type == TokenType.WILDCARD:
            name_tok = self.advance()
            # Wildcard must have constraints
            if self.peek() and self.peek().type == TokenType.LBRACE:
                constraints = self.parse_field_constraints()
                expr = InstructionPattern("*", constraints)
            else:
                self.error("Wildcard '*' must have field constraints")
        else:
            name_tok = self.expect(TokenType.IDENTIFIER)

            # Check if there are constraints
            if self.peek() and self.peek().type == TokenType.LBRACE:
                # This is an instruction pattern with constraints
                constraints = self.parse_field_constraints()
                expr = InstructionPattern(name_tok.value, constraints)
            else:
                # This is just an extension or instruction name
                expr = name_tok.value

        return ForbidRule(expr, forbid_tok.line)

# ============================================================================
# Main / Testing
# ============================================================================

def parse_dsl(text: str) -> Program:
    """Parse DSL text and return AST"""
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Parse and validate instruction outlawing DSL files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Parse and validate a DSL file
  python3 instruction_dsl_parser.py my_rules.dsl

  # Parse and show verbose output
  python3 instruction_dsl_parser.py my_rules.dsl -v

  # Run built-in test
  python3 instruction_dsl_parser.py --test

DSL Syntax:
  instruction <NAME> [ { <field>=<value>, ... } ]
  pattern <hex> mask <hex>

  Example:
    instruction MUL
    instruction ADD { rd = x0 }
    pattern 0x02000033 mask 0xFE00707F
        '''
    )

    parser.add_argument('file', nargs='?', help='DSL file to parse')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed parse output')
    parser.add_argument('--test', action='store_true',
                       help='Run built-in test instead of parsing a file')

    args = parser.parse_args()

    # Built-in test mode
    if args.test:
        test_input = """
        # Outlaw all multiply instructions
        instruction MUL
        instruction MULH

        # Outlaw MUL with specific register constraint
        instruction MUL { rd = x0 }

        # Data type constraints (negative semantics)
        instruction MUL { dtype = i16 }
        instruction ADD { rs1_dtype = i8 | u8, rs2_dtype = i8 }

        # Negated data type constraints (allow only these types)
        instruction DIV { dtype = ~(i16 | u16) }
        instruction DIVU { dtype = ~i32 }

        # Low-level pattern
        pattern 0x02000033 mask 0xFE00707F

        # Multiple constraints
        instruction ADD { rd = x0, rs1 = x1 }
        """

        try:
            ast = parse_dsl(test_input)
            print("✓ Built-in test passed!")
            print(f"Found {len(ast.rules)} rules:")
            for rule in ast.rules:
                print(f"  {rule}")
        except SyntaxError as e:
            print(f"✗ Test failed: {e}")
            sys.exit(1)
        return

    # File parsing mode
    if not args.file:
        parser.print_help()
        sys.exit(1)

    try:
        with open(args.file, 'r') as f:
            dsl_text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Parse the file
    try:
        ast = parse_dsl(dsl_text)
        print(f"✓ Parsed {args.file} successfully!")
        print(f"Found {len(ast.rules)} rules")

        if args.verbose:
            print("\nRules:")
            for i, rule in enumerate(ast.rules, 1):
                print(f"\n{i}. {type(rule).__name__}:")
                if isinstance(rule, RequireRule):
                    print(f"   Extension: {rule.extension}")
                elif isinstance(rule, RegisterConstraintRule):
                    print(f"   Register range: x{rule.min_reg}-x{rule.max_reg} ({rule.max_reg - rule.min_reg + 1} registers)")
                elif isinstance(rule, PcConstraintRule):
                    print(f"   PC bits: {rule.pc_bits} ({2**rule.pc_bits} bytes = {2**rule.pc_bits // 1024}KB address space)")
                elif isinstance(rule, InstructionRule):
                    print(f"   Name: {rule.name}")
                    if rule.constraints:
                        print(f"   Constraints:")
                        for c in rule.constraints:
                            print(f"     - {c.field_name} = {c.field_value}")
                elif isinstance(rule, PatternRule):
                    print(f"   Pattern: 0x{rule.pattern:08x}")
                    print(f"   Mask:    0x{rule.mask:08x}")
                    if rule.description:
                        print(f"   Description: {rule.description}")
                elif isinstance(rule, TimingConstraintRule):
                    # Show the specific parameter that was set
                    print(f"   {rule.param_name}: {rule.value}")
        else:
            # Summary
            require_count = sum(1 for r in ast.rules if isinstance(r, RequireRule))
            reg_constraint_count = sum(1 for r in ast.rules if isinstance(r, RegisterConstraintRule))
            pc_constraint_count = sum(1 for r in ast.rules if isinstance(r, PcConstraintRule))
            instr_count = sum(1 for r in ast.rules if isinstance(r, InstructionRule))
            pattern_count = sum(1 for r in ast.rules if isinstance(r, PatternRule))
            timing_count = sum(1 for r in ast.rules if isinstance(r, TimingConstraintRule))
            if require_count > 0:
                print(f"  - {require_count} require rules")
            if reg_constraint_count > 0:
                print(f"  - {reg_constraint_count} register constraint rules")
            if pc_constraint_count > 0:
                print(f"  - {pc_constraint_count} PC constraint rules")
            print(f"  - {instr_count} instruction rules")
            print(f"  - {pattern_count} pattern rules")
            if timing_count > 0:
                print(f"  - {timing_count} timing constraint rules")
            print("\nUse -v for detailed output")

    except SyntaxError as e:
        print(f"✗ Parse error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
