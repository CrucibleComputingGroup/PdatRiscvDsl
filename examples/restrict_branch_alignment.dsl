# Example: Restrict branch offsets to specific alignments
# Demonstrates bit patterns on branch instructions

require RV32I

# Branch instructions use B-type format with split immediates
# The immediate represents a 13-bit signed offset in multiples of 2
# For this example, we'll use simplified pattern matching

# Note: B-type instructions have complex split immediate encoding:
# imm[12|10:5] in bits [31:25], imm[4:1|11] in bits [11:7]
# Full implementation would require handling split fields

# For demonstration purposes, if we had unified 'offset' field:
# Restrict to 4-byte aligned branches (lower 2 bits = 00)
# instruction BEQ { offset = 13'bxxxxxxxxxxx00 }

# This is a design note - split immediate handling is a future enhancement
# Current implementation works best with unified immediate fields like in I-type
