//go:build amd64

#include "textflag.h"

// tailMask is a 32-byte (1 ymm) mask used to zero the high two lanes of
// the second YMM diff register, so when we over-read 16 floats per ref
// (because each ref is 14 floats wide) the next ref's first two floats
// don't contribute to the distance.
DATA tailMask<>+0(SB)/4,  $0xffffffff
DATA tailMask<>+4(SB)/4,  $0xffffffff
DATA tailMask<>+8(SB)/4,  $0xffffffff
DATA tailMask<>+12(SB)/4, $0xffffffff
DATA tailMask<>+16(SB)/4, $0xffffffff
DATA tailMask<>+20(SB)/4, $0xffffffff
DATA tailMask<>+24(SB)/4, $0x00000000
DATA tailMask<>+28(SB)/4, $0x00000000
GLOBL tailMask<>(SB), (NOPTR+RODATA), $32

// func dist14SquaredAVX2Block(q, refs *float32, n int, out *float32)
//
// Computes squared Euclidean distance between q (a 16-float buffer with
// the last two lanes zeroed by the caller) and each of the n contiguous
// 14-float reference vectors at refs, writing each distance to out[i].
//
// Stride is 56 bytes per ref (14 * 4). The loads use 32-byte unaligned
// VMOVUPS so the strided layout is fine. The kernel relies on TailPadFloats
// at the end of the file so the very last ref can also be loaded as 2 YMMs.
//
// Register map (caller-saved, no Go pointers in regs):
//   DI = q ptr            SI = refs ptr            CX = n            DX = out ptr
//   Y14 = q[0:8]   Y15 = q[8:16]   Y13 = tail mask
//   Y0/Y1 = scratch (diff/squared)   X0/X1 = horizontal-sum scratch
TEXT ·dist14SquaredAVX2Block(SB), NOSPLIT, $0-32
	MOVQ q+0(FP), DI
	MOVQ refs+8(FP), SI
	MOVQ n+16(FP), CX
	MOVQ out+24(FP), DX

	VMOVUPS (DI), Y14
	VMOVUPS 32(DI), Y15
	VMOVUPS tailMask<>(SB), Y13

	TESTQ CX, CX
	JZ done

loop:
	VMOVUPS (SI), Y0
	VMOVUPS 32(SI), Y1
	VSUBPS  Y14, Y0, Y0      // Y0 = r[0:8]  - q[0:8]
	VSUBPS  Y15, Y1, Y1      // Y1 = r[8:16] - q[8:16]
	VANDPS  Y13, Y1, Y1      // zero lanes 14, 15 of the diff
	VMULPS  Y0, Y0, Y0       // Y0 = diff^2 (lanes 0..7)

	VFMADD231PS Y1, Y1, Y0   // Y0 += diff^2 (lanes 8..15)

	// Horizontal reduce 8 lanes of Y0 into X0 lane 0.
	VEXTRACTF128 $1, Y0, X1
	VADDPS  X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

	VMOVSS X0, (DX)

	ADDQ $56, SI
	ADDQ $4, DX
	DECQ CX
	JNZ loop

done:
	VZEROUPPER
	RET
