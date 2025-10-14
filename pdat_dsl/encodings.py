#!/usr/bin/env python3
"""
RISC-V instruction encoding database and utilities.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class InstructionEncoding:
    """Represents the encoding of a RISC-V instruction"""
    name: str
    base_pattern: int  # Base encoding with all fields as 0
    base_mask: int     # Mask for fixed bits (opcode, funct3, funct7, etc.)
    format: str        # R, I, S, B, U, J
    fields: Dict[str, Tuple[int, int]]  # field_name -> (bit_position, width)

# RISC-V instruction formats and their field layouts
# Format: field_name -> (lsb_position, width)

R_TYPE_FIELDS = {
    "opcode": (0, 7),
    "rd": (7, 5),
    "funct3": (12, 3),
    "rs1": (15, 5),
    "rs2": (20, 5),
    "funct7": (25, 7),
}

I_TYPE_FIELDS = {
    "opcode": (0, 7),
    "rd": (7, 5),
    "funct3": (12, 3),
    "rs1": (15, 5),
    "imm": (20, 12),
}

S_TYPE_FIELDS = {
    "opcode": (0, 7),
    "imm_0_4": (7, 5),
    "funct3": (12, 3),
    "rs1": (15, 5),
    "rs2": (20, 5),
    "imm_5_11": (25, 7),
}

B_TYPE_FIELDS = {
    "opcode": (0, 7),
    "imm_11": (7, 1),
    "imm_1_4": (8, 4),
    "funct3": (12, 3),
    "rs1": (15, 5),
    "rs2": (20, 5),
    "imm_5_10": (25, 6),
    "imm_12": (31, 1),
}

U_TYPE_FIELDS = {
    "opcode": (0, 7),
    "rd": (7, 5),
    "imm": (12, 20),
}

J_TYPE_FIELDS = {
    "opcode": (0, 7),
    "rd": (7, 5),
    "imm_12_19": (12, 8),
    "imm_11": (20, 1),
    "imm_1_10": (21, 10),
    "imm_20": (31, 1),
}

# RV32I Base Instruction Set
RV32I_INSTRUCTIONS = {
    # Loads
    "LB":    InstructionEncoding("LB",    0x00000003, 0x0000707F, "I", I_TYPE_FIELDS),
    "LH":    InstructionEncoding("LH",    0x00001003, 0x0000707F, "I", I_TYPE_FIELDS),
    "LW":    InstructionEncoding("LW",    0x00002003, 0x0000707F, "I", I_TYPE_FIELDS),
    "LBU":   InstructionEncoding("LBU",   0x00004003, 0x0000707F, "I", I_TYPE_FIELDS),
    "LHU":   InstructionEncoding("LHU",   0x00005003, 0x0000707F, "I", I_TYPE_FIELDS),

    # Stores
    "SB":    InstructionEncoding("SB",    0x00000023, 0x0000707F, "S", S_TYPE_FIELDS),
    "SH":    InstructionEncoding("SH",    0x00001023, 0x0000707F, "S", S_TYPE_FIELDS),
    "SW":    InstructionEncoding("SW",    0x00002023, 0x0000707F, "S", S_TYPE_FIELDS),

    # ALU Immediate
    "ADDI":  InstructionEncoding("ADDI",  0x00000013, 0x0000707F, "I", I_TYPE_FIELDS),
    "SLTI":  InstructionEncoding("SLTI",  0x00002013, 0x0000707F, "I", I_TYPE_FIELDS),
    "SLTIU": InstructionEncoding("SLTIU", 0x00003013, 0x0000707F, "I", I_TYPE_FIELDS),
    "XORI":  InstructionEncoding("XORI",  0x00004013, 0x0000707F, "I", I_TYPE_FIELDS),
    "ORI":   InstructionEncoding("ORI",   0x00006013, 0x0000707F, "I", I_TYPE_FIELDS),
    "ANDI":  InstructionEncoding("ANDI",  0x00007013, 0x0000707F, "I", I_TYPE_FIELDS),
    "SLLI":  InstructionEncoding("SLLI",  0x00001013, 0xFE00707F, "I", I_TYPE_FIELDS),
    "SRLI":  InstructionEncoding("SRLI",  0x00005013, 0xFE00707F, "I", I_TYPE_FIELDS),
    "SRAI":  InstructionEncoding("SRAI",  0x40005013, 0xFE00707F, "I", I_TYPE_FIELDS),

    # ALU Register-Register
    "ADD":   InstructionEncoding("ADD",   0x00000033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "SUB":   InstructionEncoding("SUB",   0x40000033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "SLL":   InstructionEncoding("SLL",   0x00001033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "SLT":   InstructionEncoding("SLT",   0x00002033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "SLTU":  InstructionEncoding("SLTU",  0x00003033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "XOR":   InstructionEncoding("XOR",   0x00004033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "SRL":   InstructionEncoding("SRL",   0x00005033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "SRA":   InstructionEncoding("SRA",   0x40005033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "OR":    InstructionEncoding("OR",    0x00006033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "AND":   InstructionEncoding("AND",   0x00007033, 0xFE00707F, "R", R_TYPE_FIELDS),

    # Upper Immediate
    "LUI":   InstructionEncoding("LUI",   0x00000037, 0x0000007F, "U", U_TYPE_FIELDS),
    "AUIPC": InstructionEncoding("AUIPC", 0x00000017, 0x0000007F, "U", U_TYPE_FIELDS),

    # Branches
    "BEQ":   InstructionEncoding("BEQ",   0x00000063, 0x0000707F, "B", B_TYPE_FIELDS),
    "BNE":   InstructionEncoding("BNE",   0x00001063, 0x0000707F, "B", B_TYPE_FIELDS),
    "BLT":   InstructionEncoding("BLT",   0x00004063, 0x0000707F, "B", B_TYPE_FIELDS),
    "BGE":   InstructionEncoding("BGE",   0x00005063, 0x0000707F, "B", B_TYPE_FIELDS),
    "BLTU":  InstructionEncoding("BLTU",  0x00006063, 0x0000707F, "B", B_TYPE_FIELDS),
    "BGEU":  InstructionEncoding("BGEU",  0x00007063, 0x0000707F, "B", B_TYPE_FIELDS),

    # Jump
    "JAL":   InstructionEncoding("JAL",   0x0000006F, 0x0000007F, "J", J_TYPE_FIELDS),
    "JALR":  InstructionEncoding("JALR",  0x00000067, 0x0000707F, "I", I_TYPE_FIELDS),

    # System
    "ECALL":  InstructionEncoding("ECALL",  0x00000073, 0xFFFFFFFF, "I", I_TYPE_FIELDS),
    "EBREAK": InstructionEncoding("EBREAK", 0x00100073, 0xFFFFFFFF, "I", I_TYPE_FIELDS),
    "FENCE":  InstructionEncoding("FENCE",  0x0000000F, 0x0000707F, "I", I_TYPE_FIELDS),
}

# RV32M Extension (Multiply/Divide)
RV32M_INSTRUCTIONS = {
    "MUL":    InstructionEncoding("MUL",    0x02000033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "MULH":   InstructionEncoding("MULH",   0x02001033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "MULHSU": InstructionEncoding("MULHSU", 0x02002033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "MULHU":  InstructionEncoding("MULHU",  0x02003033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "DIV":    InstructionEncoding("DIV",    0x02004033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "DIVU":   InstructionEncoding("DIVU",   0x02005033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "REM":    InstructionEncoding("REM",    0x02006033, 0xFE00707F, "R", R_TYPE_FIELDS),
    "REMU":   InstructionEncoding("REMU",   0x02007033, 0xFE00707F, "R", R_TYPE_FIELDS),
}

# Privileged instructions
PRIVILEGED_INSTRUCTIONS = {
    "MRET":   InstructionEncoding("MRET",   0x30200073, 0xFFFFFFFF, "I", I_TYPE_FIELDS),
    "SRET":   InstructionEncoding("SRET",   0x10200073, 0xFFFFFFFF, "I", I_TYPE_FIELDS),
    "URET":   InstructionEncoding("URET",   0x00200073, 0xFFFFFFFF, "I", I_TYPE_FIELDS),
    "WFI":    InstructionEncoding("WFI",    0x10500073, 0xFFFFFFFF, "I", I_TYPE_FIELDS),
}

# CSR instructions
CSR_INSTRUCTIONS = {
    "CSRRW":  InstructionEncoding("CSRRW",  0x00001073, 0x0000707F, "I", I_TYPE_FIELDS),
    "CSRRS":  InstructionEncoding("CSRRS",  0x00002073, 0x0000707F, "I", I_TYPE_FIELDS),
    "CSRRC":  InstructionEncoding("CSRRC",  0x00003073, 0x0000707F, "I", I_TYPE_FIELDS),
    "CSRRWI": InstructionEncoding("CSRRWI", 0x00005073, 0x0000707F, "I", I_TYPE_FIELDS),
    "CSRRSI": InstructionEncoding("CSRRSI", 0x00006073, 0x0000707F, "I", I_TYPE_FIELDS),
    "CSRRCI": InstructionEncoding("CSRRCI", 0x00007073, 0x0000707F, "I", I_TYPE_FIELDS),
}

# RV64I Base Instruction Set (64-bit specific instructions)
RV64I_INSTRUCTIONS = {
    # 64-bit Loads
    "LWU":   InstructionEncoding("LWU",   0x00006003, 0x0000707F, "I", I_TYPE_FIELDS),
    "LD":    InstructionEncoding("LD",    0x00003003, 0x0000707F, "I", I_TYPE_FIELDS),

    # 64-bit Stores
    "SD":    InstructionEncoding("SD",    0x00003023, 0x0000707F, "S", S_TYPE_FIELDS),

    # 64-bit ALU Immediate (operate on lower 32 bits)
    "ADDIW": InstructionEncoding("ADDIW", 0x0000001B, 0x0000707F, "I", I_TYPE_FIELDS),
    "SLLIW": InstructionEncoding("SLLIW", 0x0000101B, 0xFE00707F, "I", I_TYPE_FIELDS),
    "SRLIW": InstructionEncoding("SRLIW", 0x0000501B, 0xFE00707F, "I", I_TYPE_FIELDS),
    "SRAIW": InstructionEncoding("SRAIW", 0x4000501B, 0xFE00707F, "I", I_TYPE_FIELDS),

    # 64-bit ALU Register-Register (operate on lower 32 bits)
    "ADDW":  InstructionEncoding("ADDW",  0x0000003B, 0xFE00707F, "R", R_TYPE_FIELDS),
    "SUBW":  InstructionEncoding("SUBW",  0x4000003B, 0xFE00707F, "R", R_TYPE_FIELDS),
    "SLLW":  InstructionEncoding("SLLW",  0x0000103B, 0xFE00707F, "R", R_TYPE_FIELDS),
    "SRLW":  InstructionEncoding("SRLW",  0x0000503B, 0xFE00707F, "R", R_TYPE_FIELDS),
    "SRAW":  InstructionEncoding("SRAW",  0x4000503B, 0xFE00707F, "R", R_TYPE_FIELDS),
}

# RV64M Extension (64-bit Multiply/Divide)
RV64M_INSTRUCTIONS = {
    "MULW":  InstructionEncoding("MULW",  0x0200003B, 0xFE00707F, "R", R_TYPE_FIELDS),
    "DIVW":  InstructionEncoding("DIVW",  0x0200403B, 0xFE00707F, "R", R_TYPE_FIELDS),
    "DIVUW": InstructionEncoding("DIVUW", 0x0200503B, 0xFE00707F, "R", R_TYPE_FIELDS),
    "REMW":  InstructionEncoding("REMW",  0x0200603B, 0xFE00707F, "R", R_TYPE_FIELDS),
    "REMUW": InstructionEncoding("REMUW", 0x0200703B, 0xFE00707F, "R", R_TYPE_FIELDS),
}

# Combined instruction database
ALL_INSTRUCTIONS = {
    **RV32I_INSTRUCTIONS,
    **RV32M_INSTRUCTIONS,
    **RV64I_INSTRUCTIONS,
    **RV64M_INSTRUCTIONS,
    **PRIVILEGED_INSTRUCTIONS,
    **CSR_INSTRUCTIONS,
}

# Extension name to instruction set mapping
EXTENSION_MAP = {
    "RV32I": RV32I_INSTRUCTIONS,
    "RV32M": RV32M_INSTRUCTIONS,
    "RV64I": RV64I_INSTRUCTIONS,
    "RV64M": RV64M_INSTRUCTIONS,
    "PRIV": PRIVILEGED_INSTRUCTIONS,
    "ZICSR": CSR_INSTRUCTIONS,
}

# Register name to number mapping
REGISTER_MAP = {
    "x0": 0, "zero": 0,
    "x1": 1, "ra": 1,
    "x2": 2, "sp": 2,
    "x3": 3, "gp": 3,
    "x4": 4, "tp": 4,
    "x5": 5, "t0": 5,
    "x6": 6, "t1": 6,
    "x7": 7, "t2": 7,
    "x8": 8, "s0": 8, "fp": 8,
    "x9": 9, "s1": 9,
    "x10": 10, "a0": 10,
    "x11": 11, "a1": 11,
    "x12": 12, "a2": 12,
    "x13": 13, "a3": 13,
    "x14": 14, "a4": 14,
    "x15": 15, "a5": 15,
    "x16": 16, "a6": 16,
    "x17": 17, "a7": 17,
    "x18": 18, "s2": 18,
    "x19": 19, "s3": 19,
    "x20": 20, "s4": 20,
    "x21": 21, "s5": 21,
    "x22": 22, "s6": 22,
    "x23": 23, "s7": 23,
    "x24": 24, "s8": 24,
    "x25": 25, "s9": 25,
    "x26": 26, "s10": 26,
    "x27": 27, "s11": 27,
    "x28": 28, "t3": 28,
    "x29": 29, "t4": 29,
    "x30": 30, "t5": 30,
    "x31": 31, "t6": 31,
}

def get_extension_instructions(extension: str) -> Optional[Dict[str, InstructionEncoding]]:
    """Get all instructions for a given extension (e.g., 'RV32I', 'RV32M')"""
    return EXTENSION_MAP.get(extension.upper())

def get_instruction_encoding(name: str) -> Optional[InstructionEncoding]:
    """Get the encoding for an instruction by name (case-insensitive)"""
    return ALL_INSTRUCTIONS.get(name.upper())

def parse_register(reg: str) -> Optional[int]:
    """Parse a register name or number to register number"""
    if isinstance(reg, int):
        return reg if 0 <= reg <= 31 else None

    reg = reg.lower()
    return REGISTER_MAP.get(reg)

def set_field(value: int, field_pos: int, field_width: int, field_value: int) -> int:
    """Set a field in an instruction encoding"""
    mask = (1 << field_width) - 1
    field_value = field_value & mask
    return (value & ~(mask << field_pos)) | (field_value << field_pos)

def create_field_mask(field_pos: int, field_width: int) -> int:
    """Create a mask for a specific field"""
    mask = (1 << field_width) - 1
    return mask << field_pos
