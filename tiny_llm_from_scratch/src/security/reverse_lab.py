"""
Reverse Engineering Lab - Binary Format Analysis & Disassembly Tools
Provides educational binary analysis, opcode inspection, calling convention mapping,
and Python bytecode disassembly.
"""
import dis
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PESection:
    name: str
    virtual_size: int
    virtual_address: int
    raw_data_size: int
    raw_data_ptr: int
    characteristics: int


@dataclass
class PEHeaderInfo:
    is_valid_pe: bool
    machine_type: str
    number_of_sections: int
    entry_point: int
    image_base: int
    sections: List[PESection]


class ReverseEngineeringToolkit:
    """Educational reverse engineering and binary structure analysis toolkit."""

    MACHINE_TYPES = {
        0x014C: "Intel 386 (x86 32-bit)",
        0x8664: "AMD64 (x64 64-bit)",
        0xAA64: "ARM64 Little-Endian",
        0x01C0: "ARM Little-Endian",
    }

    CALLING_CONVENTIONS = {
        "x64_windows_fastcall": {
            "name": "Microsoft x64 Calling Convention",
            "integer_regs": ["RCX", "RDX", "R8", "R9"],
            "float_regs": ["XMM0", "XMM1", "XMM2", "XMM3"],
            "shadow_space": "32 bytes (4 quadwords allocated on stack by caller)",
            "callee_saved": ["RBX", "RBP", "RDI", "RSI", "R12", "R13", "R14", "R15"],
            "return_register": "RAX (integer) / XMM0 (float)",
        },
        "system_v_x64": {
            "name": "System V AMD64 ABI (Linux, macOS, BSD)",
            "integer_regs": ["RDI", "RSI", "RDX", "RCX", "R8", "R9"],
            "float_regs": ["XMM0" for _ in range(8)],
            "shadow_space": "0 bytes (uses 128-byte red zone below RSP)",
            "callee_saved": ["RBX", "RBP", "R12", "R13", "R14", "R15"],
            "return_register": "RAX, RDX (integer) / XMM0, XMM1 (float)",
        },
        "cdecl_x86": {
            "name": "cdecl (C standard x86 32-bit)",
            "passing": "Right-to-left on stack (caller cleanup via add esp, N)",
            "return_register": "EAX",
        },
    }

    @classmethod
    def parse_pe_headers(cls, data: bytes) -> PEHeaderInfo:
        """Parses DOS Header, PE Signature, COFF Header, and Section Table without external C libraries."""
        if len(data) < 64 or data[:2] != b"MZ":
            return PEHeaderInfo(False, "Not a PE file", 0, 0, 0, [])

        # e_lfanew at offset 0x3C
        e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
        if len(data) < e_lfanew + 4 or data[e_lfanew:e_lfanew + 4] != b"PE\x00\x00":
            return PEHeaderInfo(False, "Invalid PE signature", 0, 0, 0, [])

        coff_offset = e_lfanew + 4
        machine, n_sections, time_date, sym_ptr, n_syms, opt_hdr_size, charact = struct.unpack_from(
            "<HHIIIHH", data, coff_offset
        )

        machine_name = cls.MACHINE_TYPES.get(machine, f"Unknown (0x{machine:04X})")

        opt_offset = coff_offset + 20
        magic = struct.unpack_from("<H", data, opt_offset)[0] if opt_hdr_size > 2 else 0
        entry_point = 0
        image_base = 0

        if magic == 0x10B:  # PE32
            entry_point = struct.unpack_from("<I", data, opt_offset + 16)[0]
            image_base = struct.unpack_from("<I", data, opt_offset + 28)[0]
        elif magic == 0x20B:  # PE32+ (64-bit)
            entry_point = struct.unpack_from("<I", data, opt_offset + 16)[0]
            image_base = struct.unpack_from("<Q", data, opt_offset + 24)[0]

        # Section Headers
        sections: List[PESection] = []
        sec_offset = opt_offset + opt_hdr_size
        for i in range(n_sections):
            if len(data) < sec_offset + 40:
                break
            name_raw = data[sec_offset:sec_offset + 8].rstrip(b"\x00")
            sec_name = name_raw.decode("ascii", errors="replace")
            v_size, v_addr, raw_size, raw_ptr, _, _, _, _, charact_sec = struct.unpack_from(
                "<IIIIIIHHI", data, sec_offset + 8
            )
            sections.append(PESection(
                name=sec_name,
                virtual_size=v_size,
                virtual_address=v_addr,
                raw_data_size=raw_size,
                raw_data_ptr=raw_ptr,
                characteristics=charact_sec,
            ))
            sec_offset += 40

        return PEHeaderInfo(
            is_valid_pe=True,
            machine_type=machine_name,
            number_of_sections=n_sections,
            entry_point=entry_point,
            image_base=image_base,
            sections=sections,
        )

    @classmethod
    def disassemble_python_code(cls, code_snippet: str) -> str:
        """Disassembles Python code snippet into bytecode instructions."""
        try:
            compiled = compile(code_snippet, "<snippet>", "exec")
            out = io.StringIO()
            dis.dis(compiled, file=out)
            return out.getvalue()
        except Exception as e:
            return f"Bytecode compilation error: {str(e)}"

    @classmethod
    def explain_calling_convention(cls, convention_name: str) -> Dict[str, Any]:
        """Provides detailed reference breakdown for binary calling conventions."""
        key = convention_name.lower().strip().replace(" ", "_").replace("-", "_")
        for k, v in cls.CALLING_CONVENTIONS.items():
            if key in k:
                return v
        return {
            "error": f"Unknown convention '{convention_name}'. Available: {list(cls.CALLING_CONVENTIONS.keys())}"
        }

