require RV32I
require RV32M
require_registers x0-x15
require_pc_bits 16
instruction ADD { dtype = ~(u8 | u16) }
instruction SUB { dtype = ~u8 }
instruction MUL { dtype = ~(i8 | u8 | i16 | u16) }
