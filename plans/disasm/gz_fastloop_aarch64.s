__ZN6gzippy10decompress7inflate20consume_first_decode31decode_huffman_fastloop_bounded17h8c3aea6f1fa3a989E:
	ldp	x9, x8, [x0, #8]
	subs	x10, x9, #32
	csel	x10, xzr, x10, lo
	add	x11, x8, #8
	cmp	x11, x9
	b.ls	LBB1115_2
	mov	w8, #1
	mov	x0, x8
	mov	x1, x2
	ret
LBB1115_2:
	stp	x28, x27, [sp, #-80]!
	stp	x26, x25, [sp, #16]
	stp	x24, x23, [sp, #32]
	stp	x22, x21, [sp, #48]
	stp	x20, x19, [sp, #64]
	ldr	x11, [x0]
	ldr	w9, [x0, #32]
	ldr	x12, [x0, #24]
	ldr	x13, [x11, x8]
	lsl	x13, x13, x9
	orr	x6, x13, x12
	mov	w12, #7
	bic	w13, w12, w9, lsr #3
	orr	w7, w9, #0x38
	add	x14, x2, #320
	add	x9, x8, x13
	cmp	x9, x10
	ccmp	x14, x3, #2, lo
	b.ls	LBB1115_5
	mov	w8, #1
LBB1115_4:
	and	w10, w7, #0xff
	str	w10, [x0, #32]
	stp	x9, x6, [x0, #16]
	ldp	x20, x19, [sp, #64]
	ldp	x22, x21, [sp, #48]
	ldp	x24, x23, [sp, #32]
	ldp	x26, x25, [sp, #16]
	ldp	x28, x27, [sp], #80
	mov	x0, x8
	mov	x1, x2
	ret
LBB1115_5:
	and	x8, x6, #0x7ff
	ldr	w19, [x4, x8, lsl #2]
	add	x13, x1, #16
	add	x14, x1, #32
	mov	w8, #1
	mov	x15, #-1
	mov	x16, #72340172838076673
	mov	w17, #199
LBB1115_6:
	and	w21, w7, #0xff
	cmp	w21, #44
	b.hs	LBB1115_11
	ldr	x20, [x11, x9]
	lsl	x20, x20, x7
	orr	x20, x20, x6
	sub	w6, w12, w21, lsr #3
	add	x9, x9, x6
	mov	w6, #56
	bfxil	w6, w7, #0, #3
	mov	x7, x6
	lsr	x6, x20, x19
	sub	w7, w7, w19
	tbnz	w19, #31, LBB1115_12
LBB1115_8:
	tbnz	w19, #15, LBB1115_15
	and	w22, w7, #0xff
	cmp	w22, #32
	b.hs	LBB1115_20
	ldr	x21, [x11, x9]
	and	w23, w7, #0x1f
	lsl	x21, x21, x23
	orr	x21, x21, x6
	sub	w22, w12, w22, lsr #3
	add	x9, x9, x22
	mov	w22, #56
	bfxil	w22, w7, #0, #3
	mov	x7, x22
	and	x6, x6, #0xff
	ldr	w6, [x5, x6, lsl #2]
	tbnz	w6, #14, LBB1115_21
	b	LBB1115_22
LBB1115_11:
	mov	x20, x6
	lsr	x6, x6, x19
	sub	w7, w7, w19
	tbz	w19, #31, LBB1115_8
LBB1115_12:
	and	x20, x6, #0x7ff
	ldr	w21, [x4, x20, lsl #2]
	tbnz	w21, #31, LBB1115_38
	lsr	w19, w19, #16
	strb	w19, [x1, x2]
	and	w19, w7, #0xff
	cmp	w19, #32
	b.hs	LBB1115_41
	ldr	x20, [x11, x9]
	and	w22, w7, #0x1f
	lsl	x20, x20, x22
	orr	x6, x20, x6
	sub	w19, w12, w19, lsr #3
	add	x9, x9, x19
	mov	w20, #56
	bfxil	w20, w7, #0, #3
	mov	x19, x21
	mov	x7, x20
	mov	w20, #1
	b	LBB1115_98
LBB1115_15:
	tbnz	w19, #13, LBB1115_101
	lsr	w20, w19, #16
	ubfx	w19, w19, #8, #5
	lsl	x19, x15, x19
	bic	x19, x6, x19
	add	x19, x4, x19, lsl #2
	ldr	w19, [x19, w20, uxtw #2]
	lsr	x20, x6, x19
	sub	w7, w7, w19
	tbnz	w19, #31, LBB1115_52
	tbnz	w19, #13, LBB1115_102
	ldr	x21, [x11, x9]
	lsl	x21, x21, x7
	orr	x20, x21, x20
	and	w21, w7, w17
	and	x22, x20, #0xff
	ldr	w23, [x5, x22, lsl #2]
	tbnz	w23, #14, LBB1115_53
	orr	w22, w21, #0x38
	b	LBB1115_54
LBB1115_20:
	mov	x21, x6
	and	x6, x6, #0xff
	ldr	w6, [x5, x6, lsl #2]
	tbz	w6, #14, LBB1115_22
LBB1115_21:
	lsr	x21, x21, #8
	sub	w7, w7, #8
	ubfx	x22, x6, #8, #4
	lsl	x22, x15, x22
	bic	x22, x21, x22
	add	x6, x22, x6, lsr #16
	ldr	w6, [x5, x6, lsl #2]
LBB1115_22:
	mov	x22, x6
	lsr	x6, x21, x6
	sub	w7, w7, w22
	lsl	x23, x15, x22
	bic	x21, x21, x23
	lsr	w23, w22, #8
	lsr	x21, x21, x23
	add	w22, w21, w22, lsr #16
	cmp	w22, #0
	ccmp	x2, x22, #0, ne
	b.lo	LBB1115_100
	and	w21, w19, #0x3f
	lsl	x21, x15, x21
	bic	x20, x20, x21
	lsr	w21, w19, #8
	lsr	x20, x20, x21
	add	w20, w20, w19, lsr #16
	ldr	x19, [x11, x9]
	lsl	x19, x19, x7
	orr	x6, x19, x6
	bic	w19, w12, w7, lsr #3
	add	x9, x9, x19
	and	x19, x6, #0x7ff
	ldr	w19, [x4, x19, lsl #2]
	orr	w7, w7, #0x38
	sub	x21, x2, x22
	add	x24, x1, x21
	add	x23, x1, x2
	add	x21, x23, x20
	cmp	w20, #41
	b.lo	LBB1115_34
	add	x25, x24, #40
	; InlineAsm Start
	prfm	pldl1keep, [x25]
	; InlineAsm End
	cmp	w20, #64
	b.lo	LBB1115_34
	cmp	w22, #32
	b.lo	LBB1115_34
	neg	x24, x22
	add	x26, x14, x2
LBB1115_27:
	mov	x25, x26
	add	x26, x23, x24
	ldp	q0, q1, [x26]
	stp	q0, q1, [x23]
	add	x27, x23, #64
	add	x23, x23, #32
	add	x26, x25, #32
	cmp	x27, x21
	b.ls	LBB1115_27
	add	x26, x23, #16
	cmp	x26, x21
	b.ls	LBB1115_30
	sub	x22, x23, x22
	b	LBB1115_33
LBB1115_30:
	ldr	q0, [x25, x24]
	str	q0, [x25]
	add	x22, x25, #32
	add	x25, x25, #16
	cmp	x22, x21
	b.ls	LBB1115_30
	add	x22, x25, x24
	mov	x23, x25
	b	LBB1115_33
LBB1115_32:
	ldr	x24, [x22], #8
	str	x24, [x23], #8
LBB1115_33:
	cmp	x23, x21
	b.lo	LBB1115_32
	b	LBB1115_98
LBB1115_34:
	cmp	w22, #7
	b.ls	LBB1115_42
	ldr	x25, [x24]
	str	x25, [x23]
	ldr	x25, [x24, #8]
	str	x25, [x23, #8]
	ldr	x25, [x24, #16]
	str	x25, [x23, #16]
	ldr	x25, [x24, #24]
	str	x25, [x23, #24]
	ldr	x24, [x24, #32]
	str	x24, [x23, #32]
	cmp	w20, #41
	b.lo	LBB1115_98
	add	x23, x23, #40
	neg	x22, x22
LBB1115_37:
	ldr	x24, [x23, x22]
	str	x24, [x23], #8
	cmp	x23, x21
	b.lo	LBB1115_37
	b	LBB1115_98
LBB1115_38:
	lsr	x6, x6, x21
	sub	w7, w7, w21
	and	x20, x6, #0x7ff
	ldr	w20, [x4, x20, lsl #2]
	tbnz	w20, #31, LBB1115_82
	lsr	w21, w21, #8
	and	w21, w21, #0xff00
	bfxil	w21, w19, #16, #8
	strh	w21, [x1, x2]
	and	w19, w7, #0xff
	cmp	w19, #32
	b.hs	LBB1115_85
	ldr	x21, [x11, x9]
	and	w22, w7, #0x1f
	lsl	x21, x21, x22
	orr	x6, x21, x6
	sub	w19, w12, w19, lsr #3
	add	x9, x9, x19
	mov	w21, #56
	bfxil	w21, w7, #0, #3
	mov	x19, x20
	mov	x7, x21
	mov	w20, #2
	b	LBB1115_98
LBB1115_41:
	mov	x19, x21
	mov	w20, #1
	b	LBB1115_98
LBB1115_42:
	cmp	w22, #1
	b.ne	LBB1115_80
	ldrb	w22, [x24]
	cmp	w20, #31
	b.ls	LBB1115_49
	dup.16b	v0, w22
	add	x23, x13, x2
	add	x25, x14, x2
LBB1115_45:
	mov	x24, x25
	stp	q0, q0, [x23, #-16]
	add	x26, x23, #48
	add	x23, x23, #32
	add	x25, x25, #32
	cmp	x26, x21
	b.ls	LBB1115_45
	cmp	x23, x21
	b.ls	LBB1115_48
	sub	x23, x23, #16
	b	LBB1115_49
LBB1115_48:
	add	x23, x24, #16
	str	q0, [x24], #32
	cmp	x24, x21
	mov	x24, x23
	b.ls	LBB1115_48
LBB1115_49:
	cmp	x23, x21
	b.hs	LBB1115_98
	mul	x22, x22, x16
LBB1115_51:
	str	x22, [x23], #8
	cmp	x23, x21
	b.lo	LBB1115_51
	b	LBB1115_98
LBB1115_52:
	ldr	x6, [x11, x9]
	lsr	w21, w19, #16
	lsl	x6, x6, x7
	orr	x6, x6, x20
	bic	w19, w12, w7, lsr #3
	add	x9, x9, x19
	orr	w7, w7, #0x38
	and	x19, x6, #0x7ff
	ldr	w19, [x4, x19, lsl #2]
	strb	w21, [x1, x2]
	mov	w20, #1
	b	LBB1115_98
LBB1115_53:
	lsr	x20, x20, #8
	orr	w22, w21, #0x30
	ubfx	x21, x23, #8, #4
	lsl	x21, x15, x21
	bic	x21, x20, x21
	add	x21, x21, x23, lsr #16
	ldr	w23, [x5, x21, lsl #2]
LBB1115_54:
	bic	w7, w12, w7, lsr #3
	add	x9, x9, x7
	lsr	x21, x20, x23
	sub	w7, w22, w23
	lsl	x22, x15, x23
	bic	x20, x20, x22
	lsr	w22, w23, #8
	lsr	x20, x20, x22
	add	w22, w20, w23, lsr #16
	cmp	w22, #0
	ccmp	x2, x22, #0, ne
	b.lo	LBB1115_103
	and	x20, x19, #0x3f
	lsl	x20, x15, x20
	bic	x6, x6, x20
	ubfx	x20, x19, #8, #5
	lsr	x6, x6, x20
	add	w20, w6, w19, lsr #16
	ldr	x6, [x11, x9]
	lsl	x6, x6, x7
	orr	x6, x6, x21
	bic	w19, w12, w7, lsr #3
	add	x9, x9, x19
	and	x19, x6, #0x7ff
	ldr	w19, [x4, x19, lsl #2]
	orr	w7, w7, #0x38
	sub	x21, x2, x22
	add	x24, x1, x21
	add	x23, x1, x2
	add	x21, x23, x20
	cmp	w20, #41
	b.lo	LBB1115_66
	add	x25, x24, #40
	; InlineAsm Start
	prfm	pldl1keep, [x25]
	; InlineAsm End
	cmp	w20, #64
	b.lo	LBB1115_66
	cmp	w22, #32
	b.lo	LBB1115_66
	neg	x24, x22
	add	x26, x14, x2
LBB1115_59:
	mov	x25, x26
	add	x26, x23, x24
	ldp	q0, q1, [x26]
	stp	q0, q1, [x23]
	add	x27, x23, #64
	add	x23, x23, #32
	add	x26, x25, #32
	cmp	x27, x21
	b.ls	LBB1115_59
	add	x26, x23, #16
	cmp	x26, x21
	b.ls	LBB1115_62
	sub	x22, x23, x22
	b	LBB1115_65
LBB1115_62:
	ldr	q0, [x25, x24]
	str	q0, [x25]
	add	x22, x25, #32
	add	x25, x25, #16
	cmp	x22, x21
	b.ls	LBB1115_62
	add	x22, x25, x24
	mov	x23, x25
	b	LBB1115_65
LBB1115_64:
	ldr	x24, [x22], #8
	str	x24, [x23], #8
LBB1115_65:
	cmp	x23, x21
	b.lo	LBB1115_64
	b	LBB1115_98
LBB1115_66:
	cmp	w22, #7
	b.ls	LBB1115_70
	ldr	x25, [x24]
	str	x25, [x23]
	ldr	x25, [x24, #8]
	str	x25, [x23, #8]
	ldr	x25, [x24, #16]
	str	x25, [x23, #16]
	ldr	x25, [x24, #24]
	str	x25, [x23, #24]
	ldr	x24, [x24, #32]
	str	x24, [x23, #32]
	cmp	w20, #41
	b.lo	LBB1115_98
	add	x23, x23, #40
	neg	x22, x22
LBB1115_69:
	ldr	x24, [x23, x22]
	str	x24, [x23], #8
	cmp	x23, x21
	b.lo	LBB1115_69
	b	LBB1115_98
LBB1115_70:
	cmp	w22, #1
	b.ne	LBB1115_86
	ldrb	w22, [x24]
	cmp	w20, #31
	b.ls	LBB1115_77
	dup.16b	v0, w22
	add	x23, x13, x2
	add	x25, x14, x2
LBB1115_73:
	mov	x24, x25
	stp	q0, q0, [x23, #-16]
	add	x26, x23, #48
	add	x23, x23, #32
	add	x25, x25, #32
	cmp	x26, x21
	b.ls	LBB1115_73
	cmp	x23, x21
	b.ls	LBB1115_76
	sub	x23, x23, #16
	b	LBB1115_77
LBB1115_76:
	add	x23, x24, #16
	str	q0, [x24], #32
	cmp	x24, x21
	mov	x24, x23
	b.ls	LBB1115_76
LBB1115_77:
	cmp	x23, x21
	b.hs	LBB1115_98
	mul	x22, x22, x16
LBB1115_79:
	str	x22, [x23], #8
	cmp	x23, x21
	b.lo	LBB1115_79
	b	LBB1115_98
LBB1115_80:
	ldr	x25, [x24]
	str	x25, [x23]
	add	x23, x23, x22
	add	x24, x23, x22
	str	x25, [x23]
	str	x25, [x24]
	add	x23, x24, x22
	add	x24, x23, x22
	str	x25, [x23]
	cmp	x24, x21
	b.hs	LBB1115_98
LBB1115_81:
	ldr	x25, [x23]
	str	x25, [x24]
	add	x23, x23, x22
	add	x24, x24, x22
	cmp	x24, x21
	b.lo	LBB1115_81
	b	LBB1115_98
LBB1115_82:
	lsr	w25, w21, #16
	lsr	x6, x6, x20
	sub	w7, w7, w20
	and	x21, x6, #0x7ff
	ldr	w21, [x4, x21, lsl #2]
	tbnz	w21, #31, LBB1115_88
	ubfx	w19, w19, #16, #8
	bfi	w19, w25, #8, #8
	and	w20, w20, #0xff0000
	orr	w19, w19, w20
	str	w19, [x1, x2]
	and	w19, w7, #0xff
	cmp	w19, #32
	b.hs	LBB1115_90
	ldr	x20, [x11, x9]
	and	w22, w7, #0x1f
	lsl	x20, x20, x22
	orr	x6, x20, x6
	sub	w19, w12, w19, lsr #3
	add	x9, x9, x19
	mov	w20, #56
	bfxil	w20, w7, #0, #3
	mov	x19, x21
	mov	x7, x20
	mov	w20, #3
	b	LBB1115_98
LBB1115_85:
	mov	x19, x20
	mov	w20, #2
	b	LBB1115_98
LBB1115_86:
	ldr	x25, [x24]
	str	x25, [x23]
	add	x23, x23, x22
	add	x24, x23, x22
	str	x25, [x23]
	str	x25, [x24]
	add	x23, x24, x22
	add	x24, x23, x22
	str	x25, [x23]
	cmp	x24, x21
	b.hs	LBB1115_98
LBB1115_87:
	ldr	x25, [x23]
	str	x25, [x24]
	add	x23, x23, x22
	add	x24, x24, x22
	cmp	x24, x21
	b.lo	LBB1115_87
	b	LBB1115_98
LBB1115_88:
	lsr	w24, w21, #16
	lsr	x6, x6, x21
	sub	w7, w7, w21
	ldr	x21, [x11, x9]
	lsl	x21, x21, x7
	orr	x6, x21, x6
	bic	w21, w12, w7, lsr #3
	add	x9, x9, x21
	orr	w7, w7, #0x38
	and	x21, x6, #0x7ff
	ldr	w22, [x4, x21, lsl #2]
	tbnz	w22, #31, LBB1115_91
	ubfx	w19, w19, #16, #8
	bfi	w19, w25, #8, #8
	and	w20, w20, #0xff0000
	orr	w19, w19, w20
	orr	w19, w19, w24, lsl #24
	str	w19, [x1, x2]
	mov	x19, x22
	mov	w20, #4
	b	LBB1115_98
LBB1115_90:
	mov	x19, x21
	mov	w20, #3
	b	LBB1115_98
LBB1115_91:
	lsr	x6, x6, x22
	sub	w7, w7, w22
	and	x21, x6, #0x7ff
	ldr	w21, [x4, x21, lsl #2]
	tbnz	w21, #31, LBB1115_93
	ubfx	w19, w19, #16, #8
	bfi	w19, w25, #8, #8
	and	w20, w20, #0xff0000
	orr	w19, w19, w20
	orr	w19, w19, w24, lsl #24
	ubfx	w20, w22, #16, #8
	orr	x19, x19, x20, lsl #32
	str	x19, [x1, x2]
	ldr	x19, [x11, x9]
	bic	w20, w12, w7, lsr #3
	lsl	x19, x19, x7
	orr	x6, x19, x6
	add	x9, x9, x20
	orr	w7, w7, #0x38
	mov	x19, x21
	mov	w20, #5
	b	LBB1115_98
LBB1115_93:
	lsr	x6, x6, x21
	sub	w7, w7, w21
	and	x23, x6, #0x7ff
	ldr	w23, [x4, x23, lsl #2]
	tbnz	w23, #31, LBB1115_95
	ubfx	w19, w19, #16, #8
	bfi	w19, w25, #8, #8
	and	w20, w20, #0xff0000
	orr	w19, w19, w20
	orr	w19, w19, w24, lsl #24
	ubfx	w20, w22, #16, #8
	orr	x19, x19, x20, lsl #32
	ubfx	w20, w21, #16, #8
	orr	x19, x19, x20, lsl #40
	str	x19, [x1, x2]
	ldr	x19, [x11, x9]
	bic	w20, w12, w7, lsr #3
	lsl	x19, x19, x7
	orr	x6, x19, x6
	add	x9, x9, x20
	orr	w7, w7, #0x38
	mov	x19, x23
	mov	w20, #6
	b	LBB1115_98
LBB1115_95:
	lsr	x6, x6, x23
	sub	w7, w7, w23
	ldr	x26, [x11, x9]
	lsl	x26, x26, x7
	orr	x6, x26, x6
	bic	w26, w12, w7, lsr #3
	add	x9, x9, x26
	orr	w7, w7, #0x38
	and	x26, x6, #0x7ff
	ldr	w26, [x4, x26, lsl #2]
	tbnz	w26, #31, LBB1115_97
	ubfx	w19, w19, #16, #8
	bfi	w19, w25, #8, #8
	and	w20, w20, #0xff0000
	orr	w19, w19, w20
	orr	w19, w19, w24, lsl #24
	ubfx	w20, w22, #16, #8
	orr	x19, x19, x20, lsl #32
	ubfx	w20, w21, #16, #8
	orr	x19, x19, x20, lsl #40
	ubfx	w20, w23, #16, #8
	orr	x19, x19, x20, lsl #48
	str	x19, [x1, x2]
	mov	x19, x26
	mov	w20, #7
	b	LBB1115_98
LBB1115_97:
	lsr	x27, x26, #16
	lsr	x6, x6, x26
	sub	w7, w7, w26
	and	x26, x6, #0x7ff
	ubfx	w19, w19, #16, #8
	ldr	w26, [x4, x26, lsl #2]
	bfi	w19, w25, #8, #8
	and	w20, w20, #0xff0000
	orr	w19, w19, w20
	orr	w19, w19, w24, lsl #24
	ubfx	w20, w22, #16, #8
	orr	x19, x19, x20, lsl #32
	ubfx	w20, w21, #16, #8
	orr	x19, x19, x20, lsl #40
	ubfx	w20, w23, #16, #8
	orr	x19, x19, x20, lsl #48
	bfi	x19, x27, #56, #8
	str	x19, [x1, x2]
	mov	x19, x26
	mov	w20, #8
LBB1115_98:
	add	x2, x2, x20
	cmp	x9, x10
	b.hs	LBB1115_4
	add	x20, x2, #320
	cmp	x20, x3
	b.ls	LBB1115_6
	b	LBB1115_4
LBB1115_100:
	mov	w8, #2
	b	LBB1115_4
LBB1115_101:
	mov	x8, #0
	b	LBB1115_4
LBB1115_102:
	mov	x8, #0
	mov	x6, x20
	b	LBB1115_4
LBB1115_103:
	mov	x6, x21
	mov	w8, #2
	b	LBB1115_4

	.p2align	2
__ZN6gzippy10decompress7inflate20consume_first_decode31decode_huffman_libdeflate_style17h79c0fe8185c15ff2E:
