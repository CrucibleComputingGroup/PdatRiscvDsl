/**
 * Tree-sitter grammar for PDAT DSL
 *
 * Grammar for RISC-V ISA subset specification language
 * Supports: require, require_registers, instruction, pattern rules
 */

module.exports = grammar({
  name: 'pdat_dsl',

  // Target ABI version 14 for compatibility
  // Remove this line if using tree-sitter-cli >= 0.23.0
  word: $ => $.identifier,

  extras: $ => [
    /\s/,           // Whitespace
    $.comment,      // Comments
  ],

  rules: {
    source_file: $ => repeat($._statement),

    _statement: $ => choice(
      $.require_rule,
      $.require_registers_rule,
      $.instruction_rule,
      $.pattern_rule,
    ),

    // require RV32I
    require_rule: $ => seq(
      'require',
      field('extension', $.identifier)
    ),

    // require_registers x0-x15
    require_registers_rule: $ => seq(
      'require_registers',
      field('range', $.register_range)
    ),

    register_range: $ => seq(
      field('start', choice($.register_name, $.number)),
      '-',
      field('end', choice($.register_name, $.number))
    ),

    // instruction MUL
    // instruction ADD { rd = x0, rs1 = x1 }
    instruction_rule: $ => seq(
      'instruction',
      field('name', $.identifier),
      optional(field('constraints', $.field_constraints))
    ),

    // pattern 0x02000033 mask 0xFE00707F
    pattern_rule: $ => seq(
      'pattern',
      field('pattern', $.number),
      'mask',
      field('mask', $.number)
    ),

    field_constraints: $ => seq(
      '{',
      sepBy1(',', $.field_constraint),
      '}'
    ),

    field_constraint: $ => seq(
      field('field', $.identifier),
      '=',
      field('value', choice(
        $.wildcard,
        $.number,
        $.register_name,
        $.identifier
      ))
    ),

    // Terminals
    identifier: $ => /[A-Za-z_][A-Za-z0-9_]*/,

    register_name: $ => /[xX][0-9]{1,2}/,

    number: $ => choice(
      /0[xX][0-9a-fA-F_]+/,   // Hexadecimal
      /0[bB][01_]+/,          // Binary
      /[0-9][0-9_]*/          // Decimal
    ),

    wildcard: $ => choice('*', 'x', '_'),

    comment: $ => token(seq('#', /.*/)),
  }
});

/**
 * Helper function to create comma-separated lists
 */
function sepBy1(sep, rule) {
  return seq(rule, repeat(seq(sep, rule)));
}
