# Data Type Constraints - Implementation Summary

## Overview

This document describes the data type constraint feature added to the PDAT DSL. This feature allows specifying the width and signedness of data used by RISC-V instructions, capturing the semantics of C-level data types (char, short, int, long) in hardware specifications.

## Motivation

When RISC-V instructions like `mul` are used with narrow C types (e.g., `char` or `short`), the upper bits of the operands are effectively unused. This information is valuable for:

1. **Hardware Optimization**: Synthesis tools can optimize away unused logic when they know operands only use lower bits
2. **Formal Verification**: Generate assertions that verify operand ranges
3. **Power Analysis**: Model switching activity based on actual data widths
4. **Documentation**: Capture high-level intent in machine-readable form

## Design Decisions

### 1. Syntax Integration
Data type constraints extend the existing field constraint syntax:
```
instruction MUL { dtype = i16 }
instruction MUL { dtype = ~i16 }
```

This approach:
- Reuses existing parsing infrastructure
- Maintains consistency with register constraints
- Allows combining data types with other constraints

### 2. Semantic Consistency (Negative vs Positive Constraints)

The DSL follows an "outlawing" paradigm where instruction rules specify patterns to **forbid**:
```
instruction MUL              # Forbid ALL MUL instructions
instruction MUL { rd = x0 }  # Forbid MUL when rd=x0
```

Data type constraints maintain this semantic consistency with **negative constraints**:
```
instruction MUL { dtype = i16 }  # Forbid MUL when using i16
```

However, the most useful data type information is often **positive** ("only uses these types"), not negative. To address this, we added the **negation prefix `~`** which inverts the constraint:
```
instruction MUL { dtype = ~i16 }  # Forbid MUL when NOT using i16 = "allow only i16"
```

This design:
- Maintains semantic consistency with outlawing
- Provides intuitive positive constraint syntax via negation
- Makes the common case (allowing specific types) readable: `~(i16 | u16)` = "allow only i16 or u16"

### 3. Negation as Expression Prefix (Not a General Operator)

The `~` is **not** a composable unary operator—it's a **type expression prefix** that can only appear once at the very beginning:

**Valid:**
```
~i16
~(i16 | u16)
~(i8 | u8 | i16 | u16)
```

**Invalid:**
```
~~i16           # Double negation
i16 | ~u16      # Negation in middle
~i16 | ~u16     # Multiple negations
```

This design:
- Simplifies parsing (no operator precedence)
- Makes semantics clear (entire expression is negated or not)
- Avoids complex boolean logic
- Fits the domain-specific nature of the constraint language

### 4. Granularity
The implementation supports **both** instruction-level and per-operand constraints:

**Instruction-level** (all operands share type):
```
instruction MUL { dtype = i16 }
```

**Per-operand** (fine-grained control):
```
instruction MUL { rd_dtype = i16, rs1_dtype = i16, rs2_dtype = i16 }
```

This flexibility allows expressing both common cases (uniform types) and edge cases (mixed types).

### 5. Multiple Type Support
The pipe operator `|` allows specifying multiple acceptable types:
```
instruction DIV { dtype = i8 | i16 | u8 }
```

This captures situations where code uses multiple data types with the same instruction.

### 6. Signedness Distinction
Separate signed (`i*`) and unsigned (`u*`) types capture the semantic difference:
- **Signed types** (`i8`, `i16`, `i32`, `i64`): Sign-extended to full register width
- **Unsigned types** (`u8`, `u16`, `u32`, `u64`): Zero-extended to full register width

This distinction matters for hardware optimization and verification.

## Implementation Details

### AST Extensions

#### DataType Class
```python
@dataclass(frozen=True)
class DataType:
    width: int      # 8, 16, 32, or 64
    signed: bool    # True for i*, False for u*
```

Immutable (frozen) to allow use in sets and as dictionary keys.

#### DataTypeSet Class
```python
@dataclass
class DataTypeSet:
    types: Set[DataType]
```

Represents a set of acceptable data types (for `|` operator support).

#### FieldConstraint Extension
```python
@dataclass
class FieldConstraint:
    field_name: str
    field_value: Union[str, int, DataTypeSet]
```

Extended to accept DataTypeSet in addition to existing types (wildcard, number, register).

### Lexer Changes

1. **New Token Type**: `TokenType.DTYPE` for data type literals (i8, u16, etc.)
2. **New Token Type**: `TokenType.PIPE` for the `|` operator
3. **Helper Method**: `_is_data_type()` validates data type identifiers during lexing

### Parser Changes

1. **New Method**: `parse_data_type_set()` handles parsing of type expressions with `|`
2. **Extended Method**: `parse_field_constraint()` recognizes and parses data type values
3. **Updated Test**: Built-in test includes data type constraint examples

### Grammar Changes (Tree-sitter)

```javascript
field_constraint: $ => seq(
  field('field', $.identifier),
  '=',
  field('value', choice(
    $.wildcard,
    $.number,
    $.register_name,
    $.identifier,
    $.data_type_set  // New
  ))
),

data_type_set: $ => seq(
  $.data_type,
  repeat(seq('|', $.data_type))
),

data_type: $ => /[iu](8|16|32|64)/,
```

### Syntax Highlighting

Updated Tree-sitter queries to highlight:
- Data types as `@type.builtin`
- Pipe operator as `@operator`

## Usage Examples

### Example 1: Negative Constraint (Forbid Specific Types)
```
# Forbid MUL when using 16-bit signed
instruction MUL { dtype = i16 }
```

**Meaning**: Outlaw MUL instructions that operate on 16-bit signed values. MUL with other types (i8, u8, u16, i32, etc.) is allowed.

### Example 2: Positive Constraint (Allow Only Specific Types)
```
# Allow MUL only with 16-bit signed (forbid all others)
instruction MUL { dtype = ~i16 }
```

**Meaning**: MUL instructions may ONLY use 16-bit signed operands. All operands (rd, rs1, rs2) must be i16. Upper 16 bits are unused and can be optimized away.

### Example 3: Per-Operand Positive Constraint
```
# MULHU: allow only unsigned 16-bit
instruction MULHU { rd_dtype = ~u16, rs1_dtype = ~u16, rs2_dtype = ~u16 }
```

**Meaning**: All operands must be 16-bit unsigned. Upper 16 bits unused, can be optimized.

### Example 4: Mixed Signedness with Negation
```
# MULHSU: allow signed 8-bit for rd/rs1, signed or unsigned 8-bit for rs2
instruction MULHSU { rd_dtype = ~i8, rs1_dtype = ~i8, rs2_dtype = ~(i8 | u8) }
```

**Meaning**: Destination and rs1 must be signed 8-bit. rs2 can be either signed or unsigned 8-bit, but nothing wider.

### Example 5: Multiple Allowed Widths
```
# DIV: allow only 8-bit or 16-bit signed operands
instruction DIV { dtype = ~(i8 | i16) }
```

**Meaning**: DIV may only use 8-bit or 16-bit signed values. 32-bit, 64-bit, and unsigned variants are forbidden.

### Example 6: Forbidding Specific Types
```
# MUL: forbid when using 8-bit or unsigned types
instruction MUL { dtype = i8 | u8 | u16 | u32 }
```

**Meaning**: Outlaw MUL when operands are 8-bit or any unsigned type. Signed 16-bit, 32-bit, 64-bit MUL is allowed.

### Example 7: Combined Constraints
```
# MUL: specific register with allowed type
instruction MUL { rd = x5, rs1_dtype = ~i16, rs2_dtype = ~i16 }
```

**Meaning**: MUL with destination x5, where sources must be 16-bit signed.

## Testing

Comprehensive testing was performed:

1. **Parser Built-in Test**: Includes data type constraint examples
2. **Example File**: `examples/data_type_constraints.dsl` demonstrates all features
3. **Edge Cases**: Verified all widths (8, 16, 32, 64) and both signedness options
4. **Backward Compatibility**: Existing examples parse without modification
5. **Multiple Types**: Verified pipe operator with up to 4 types

All tests pass successfully.

## Future Work

### Code Generation Support

The following code generators will need updates to utilize data type constraints:

1. **codegen.py** (SystemVerilog assertions):
   - Generate assertions checking operand bit patterns
   - Example: `assert (rs1[31:16] == {16{rs1[15]}})` for i16 (sign-extended)
   - Example: `assert (rs2[31:8] == 0)` for u8 (zero-extended)

2. **smt_constraints.py** (SMT2 constraints):
   - Generate SMT constraints for data ranges
   - Example: `(assert (bvslt rs1 32768))` for i16
   - Example: `(assert (bvult rs2 256))` for u8

3. **random_constraints.py** (VCS randomization):
   - Constrain randomization to valid ranges
   - Example: `constraint rs1_c { rs1[31:16] == rs1[15] ? 16'hFFFF : 16'h0000; }`

### Semantic Validation

While the parser accepts any field name (dtype, rd_dtype, etc.), semantic validation should:
- Verify field names match instruction format (e.g., R-type has rd, rs1, rs2)
- Warn about conflicting constraints
- Validate per-operand types are compatible with instruction semantics

### Extension Ideas

Potential future enhancements:
- **Immediate constraints**: Specify immediate field widths/ranges
- **Memory access size**: Indicate actual memory transfer size for loads/stores
- **Register subsets with types**: Combine register ranges with type constraints

## Files Modified

1. `pdat_dsl/parser.py` - Lexer and parser extensions
2. `tree-sitter-pdat-dsl/grammar.js` - Tree-sitter grammar
3. `tree-sitter-pdat-dsl/queries/highlights.scm` - Syntax highlighting
4. `examples/data_type_constraints.dsl` - Comprehensive example
5. `README.md` - Documentation and usage guide

## Backward Compatibility

**Full backward compatibility maintained**. All existing DSL files parse without modification. The feature is purely additive.

## Language Grammar

Complete grammar including data type constraints:

```
program = { rule }
rule = require_rule | register_constraint_rule | instruction_rule | pattern_rule

require_rule = "require" extension_name
register_constraint_rule = "require_registers" register_range
instruction_rule = "instruction" identifier [ field_constraints ]
pattern_rule = "pattern" hex_value "mask" hex_value

register_range = register_name "-" register_name | number "-" number
field_constraints = "{" field_constraint { "," field_constraint } "}"
field_constraint = field_name "=" field_value
field_value = wildcard | number | register_name | data_type_set
data_type_set = data_type { "|" data_type }
data_type = ("i" | "u") ("8" | "16" | "32" | "64")
```

## Conclusion

The data type constraint feature provides a clean, composable way to specify operand width and signedness in the PDAT DSL. The implementation:

- Integrates naturally with existing syntax
- Supports both simple and complex use cases
- Maintains full backward compatibility
- Provides foundation for code generation enhancements

This feature bridges the gap between high-level C code and low-level hardware, enabling better optimization and verification workflows.
