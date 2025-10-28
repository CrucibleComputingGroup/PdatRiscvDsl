; Keywords
[
  "require"
  "require_registers"
  "instruction"
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

; Register names
(register_name) @variable.parameter

; Field names in constraints
(field_constraint
  field: (identifier) @property)

; Numbers
(number) @number

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
