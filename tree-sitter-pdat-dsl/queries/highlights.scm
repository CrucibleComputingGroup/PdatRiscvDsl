; Keywords
[
  "version"
  "require"
  "require_registers"
  "require_pc_bits"
  "instruction"
  "include"
  "forbid"
  "pattern"
  "mask"
] @keyword

; Comments
(comment) @comment

; Extension names (after require)
(require_rule
  extension: (identifier) @type)

; Instruction names
(instruction_rule
  name: (identifier) @function)

(include_rule
  name: (identifier) @function)

(forbid_rule
  name: (identifier) @function)

; Register names
(register_name) @variable.parameter

; Field names in constraints
(field_constraint
  field: (identifier) @property)

; Numbers
(number) @number

; Bit patterns
(bit_pattern) @string.special

; Wildcards
(wildcard) @constant.builtin

; Data types
(data_type) @type.builtin

; Operators
"=" @operator
"-" @operator
"|" @operator
"~" @operator

; Delimiters
["{" "}"] @punctuation.bracket
"," @punctuation.delimiter
