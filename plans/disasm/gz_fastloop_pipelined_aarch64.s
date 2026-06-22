__ZN6gzippy10decompress7inflate20consume_first_decode41decode_huffman_fastloop_bounded_pipelined17h224aeb460fa80ef8E:
	ldp	x10, x9, [x0, #8]
	subs	x8, x10, #32
	csel	x8, xzr, x8, lo
	add	x11, x9, #8
	cmp	x11, x10
	b.ls	LBB1120_2
	mov	w8, #1
	mov	x0, x8
	mov	x1, x2
	ret
LBB1120_2:
	stp	x26, x25, [sp, #-64]!
	stp	x24, x23, [sp, #16]
	stp	x22, x21, [sp, #32]
	stp	x20, x19, [sp, #48]
	ldr	x10, [x0]
	ldr	w12, [x0, #32]
	ldr	x11, [x0, #24]
	ldr	x13, [x10, x9]
	lsl	x13, x13, x12
	orr	x17, x13, x11
	mov	w11, #7
	bic	w13, w11, w12, lsr #3
	and	w12, w12, #0xff
	orr	w7, w12, #0x38
	add	x12, x2, #320
	add	x9, x9, x13
	cmp	x9, x8
	ccmp	x12, x3, #2, lo
	b.ls	LBB1120_5
LBB1120_3:
	str	x17, [x0, #24]
	mov	w8, #1
LBB1120_4:
	str	w7, [x0, #32]
	str	x9, [x0, #16]
	ldp	x20, x19, [sp, #48]
	ldp	x22, x21, [sp, #32]
	ldp	x24, x23, [sp, #16]
	ldp	x26, x25, [sp], #64
	mov	x0, x8
	mov	x1, x2
	ret
LBB1120_5:
	and	x12, x17, #0x7ff
	ldr	w6, [x4, x12, lsl #2]
	add	x12, x1, #16
	add	x13, x1, #32
	mov	x14, #-1
	mov	x15, #72340172838076673
	mov	w16, #199
LBB1120_6:
	lsr	x19, x17, x6
	sub	w7, w7, w6
	tbnz	w6, #31, LBB1120_10
	mov	x20, x6
	tbnz	w20, #15, LBB1120_42
LBB1120_8:
	and	w21, w7, #0xff
	cmp	w21, #32
	b.hs	LBB1120_12
	ldr	x6, [x10, x9]
	and	w22, w7, #0x1f
	lsl	x6, x6, x22
	orr	x6, x6, x19
	sub	w21, w11, w21, lsr #3
	add	x9, x9, x21
	mov	w21, #56
	bfxil	w21, w7, #0, #3
	mov	x7, x21
	and	x19, x19, #0xff
	ldr	w19, [x5, x19, lsl #2]
	tbnz	w19, #14, LBB1120_13
	b	LBB1120_14
LBB1120_10:
	lsr	w17, w6, #16
	and	x6, x19, #0x7ff
	ldr	w20, [x4, x6, lsl #2]
	lsr	x6, x19, x20
	sub	w7, w7, w20
	strb	w17, [x1, x2]
	add	x21, x2, #1
	tbnz	w20, #31, LBB1120_40
	mov	x17, x19
	mov	x19, x6
	mov	x2, x21
	tbz	w20, #15, LBB1120_8
	b	LBB1120_42
LBB1120_12:
	mov	x6, x19
	and	x19, x19, #0xff
	ldr	w19, [x5, x19, lsl #2]
	tbz	w19, #14, LBB1120_14
LBB1120_13:
	lsr	x6, x6, #8
	sub	w7, w7, #8
	ubfx	x21, x19, #8, #4
	lsl	x21, x14, x21
	bic	x21, x6, x21
	add	x19, x21, x19, lsr #16
	ldr	w19, [x5, x19, lsl #2]
LBB1120_14:
	mov	x21, x19
	lsr	x19, x6, x19
	sub	w7, w7, w21
	lsl	x22, x14, x21
	bic	x6, x6, x22
	lsr	w22, w21, #8
	lsr	x6, x6, x22
	add	w21, w6, w21, lsr #16
	cmp	w21, #0
	ccmp	x2, x21, #0, ne
	b.lo	LBB1120_82
	lsl	x6, x14, x20
	bic	x17, x17, x6
	lsr	w6, w20, #8
	ldr	x22, [x10, x9]
	lsr	x23, x17, x6
	lsl	x17, x22, x7
	orr	x17, x17, x19
	and	x6, x17, #0x7ff
	ldr	w6, [x4, x6, lsl #2]
	add	w19, w23, w20, lsr #16
	sub	x20, x2, x21
	add	x23, x1, x20
	add	x22, x1, x2
	add	x20, x22, x19
	cmp	w19, #41
	b.lo	LBB1120_26
	add	x24, x23, #40
	; InlineAsm Start
	prfm	pldl1keep, [x24]
	; InlineAsm End
	cmp	w19, #64
	b.lo	LBB1120_26
	cmp	w21, #32
	b.lo	LBB1120_26
	neg	x23, x21
	add	x25, x13, x2
LBB1120_19:
	mov	x24, x25
	add	x25, x22, x23
	ldp	q0, q1, [x25]
	stp	q0, q1, [x22]
	add	x26, x22, #64
	add	x22, x22, #32
	add	x25, x24, #32
	cmp	x26, x20
	b.ls	LBB1120_19
	add	x25, x22, #16
	cmp	x25, x20
	b.ls	LBB1120_22
	sub	x21, x22, x21
	b	LBB1120_25
LBB1120_22:
	ldr	q0, [x24, x23]
	str	q0, [x24]
	add	x21, x24, #32
	add	x24, x24, #16
	cmp	x21, x20
	b.ls	LBB1120_22
	add	x21, x24, x23
	mov	x22, x24
	b	LBB1120_25
LBB1120_24:
	ldr	x23, [x21], #8
	str	x23, [x22], #8
LBB1120_25:
	cmp	x22, x20
	b.lo	LBB1120_24
	b	LBB1120_75
LBB1120_26:
	cmp	w21, #7
	b.ls	LBB1120_30
	ldr	x24, [x23]
	str	x24, [x22]
	ldr	x24, [x23, #8]
	str	x24, [x22, #8]
	ldr	x24, [x23, #16]
	str	x24, [x22, #16]
	ldr	x24, [x23, #24]
	str	x24, [x22, #24]
	ldr	x23, [x23, #32]
	str	x23, [x22, #32]
	cmp	w19, #41
	b.lo	LBB1120_75
	add	x22, x22, #40
	neg	x21, x21
LBB1120_29:
	ldr	x23, [x22, x21]
	str	x23, [x22], #8
	cmp	x22, x20
	b.lo	LBB1120_29
	b	LBB1120_75
LBB1120_30:
	cmp	w21, #1
	b.ne	LBB1120_77
	ldrb	w21, [x23]
	cmp	w19, #31
	b.ls	LBB1120_37
	dup.16b	v0, w21
	add	x22, x12, x2
	add	x24, x13, x2
LBB1120_33:
	mov	x23, x24
	stp	q0, q0, [x22, #-16]
	add	x25, x22, #48
	add	x22, x22, #32
	add	x24, x24, #32
	cmp	x25, x20
	b.ls	LBB1120_33
	cmp	x22, x20
	b.ls	LBB1120_36
	sub	x22, x22, #16
	b	LBB1120_37
LBB1120_36:
	add	x22, x23, #16
	str	q0, [x23], #32
	cmp	x23, x20
	mov	x23, x22
	b.ls	LBB1120_36
LBB1120_37:
	cmp	x22, x20
	b.hs	LBB1120_75
	mul	x21, x21, x15
LBB1120_39:
	str	x21, [x22], #8
	cmp	x22, x20
	b.lo	LBB1120_39
	b	LBB1120_75
LBB1120_40:
	lsr	w17, w20, #16
	and	x19, x6, #0x7ff
	ldr	w20, [x4, x19, lsl #2]
	lsr	x19, x6, x20
	sub	w7, w7, w20
	strb	w17, [x1, x21]
	add	x21, x2, #2
	tbnz	w20, #31, LBB1120_79
	mov	x17, x6
	mov	x2, x21
	tbz	w20, #15, LBB1120_8
LBB1120_42:
	tbnz	w20, #13, LBB1120_83
	lsr	w17, w20, #16
	ubfx	w6, w20, #8, #5
	lsl	x6, x14, x6
	bic	x6, x19, x6
	add	x6, x4, x6, lsl #2
	ldr	w20, [x6, w17, uxtw #2]
	lsr	x17, x19, x20
	sub	w7, w7, w20
	tbnz	w20, #31, LBB1120_47
	tbnz	w20, #13, LBB1120_84
	ldr	x6, [x10, x9]
	lsl	x6, x6, x7
	orr	x17, x6, x17
	and	w6, w7, w16
	and	x21, x17, #0xff
	ldr	w22, [x5, x21, lsl #2]
	tbnz	w22, #14, LBB1120_48
	orr	w21, w6, #0x38
	b	LBB1120_49
LBB1120_47:
	ldr	x6, [x10, x9]
	lsr	w19, w20, #16
	lsl	x6, x6, x7
	orr	x17, x6, x17
	bic	w6, w11, w7, lsr #3
	add	x9, x9, x6
	and	x6, x17, #0x7ff
	ldr	w6, [x4, x6, lsl #2]
	strb	w19, [x1, x2]
	add	x2, x2, #1
	and	w7, w7, #0xff
	orr	w7, w7, #0x38
	cmp	x9, x8
	b.lo	LBB1120_76
	b	LBB1120_3
LBB1120_48:
	lsr	x17, x17, #8
	orr	w21, w6, #0x30
	ubfx	x6, x22, #8, #4
	lsl	x6, x14, x6
	bic	x6, x17, x6
	add	x6, x6, x22, lsr #16
	ldr	w22, [x5, x6, lsl #2]
LBB1120_49:
	bic	w6, w11, w7, lsr #3
	add	x9, x9, x6
	lsr	x6, x17, x22
	sub	w7, w21, w22
	lsl	x21, x14, x22
	bic	x17, x17, x21
	lsr	w21, w22, #8
	lsr	x17, x17, x21
	add	w21, w17, w22, lsr #16
	cmp	w21, #0
	ccmp	x2, x21, #0, ne
	b.lo	LBB1120_86
	and	x17, x20, #0x3f
	lsl	x17, x14, x17
	bic	x17, x19, x17
	ubfx	x19, x20, #8, #5
	ldr	x22, [x10, x9]
	lsr	x19, x17, x19
	lsl	x17, x22, x7
	orr	x17, x17, x6
	and	x6, x17, #0x7ff
	ldr	w6, [x4, x6, lsl #2]
	add	w19, w19, w20, lsr #16
	sub	x20, x2, x21
	add	x23, x1, x20
	add	x22, x1, x2
	add	x20, x22, x19
	cmp	w19, #41
	b.lo	LBB1120_61
	add	x24, x23, #40
	; InlineAsm Start
	prfm	pldl1keep, [x24]
	; InlineAsm End
	cmp	w19, #64
	b.lo	LBB1120_61
	cmp	w21, #32
	b.lo	LBB1120_61
	neg	x23, x21
	add	x25, x13, x2
LBB1120_54:
	mov	x24, x25
	add	x25, x22, x23
	ldp	q0, q1, [x25]
	stp	q0, q1, [x22]
	add	x26, x22, #64
	add	x22, x22, #32
	add	x25, x24, #32
	cmp	x26, x20
	b.ls	LBB1120_54
	add	x25, x22, #16
	cmp	x25, x20
	b.ls	LBB1120_57
	sub	x21, x22, x21
	b	LBB1120_60
LBB1120_57:
	ldr	q0, [x24, x23]
	str	q0, [x24]
	add	x21, x24, #32
	add	x24, x24, #16
	cmp	x21, x20
	b.ls	LBB1120_57
	add	x21, x24, x23
	mov	x22, x24
	b	LBB1120_60
LBB1120_59:
	ldr	x23, [x21], #8
	str	x23, [x22], #8
LBB1120_60:
	cmp	x22, x20
	b.lo	LBB1120_59
	b	LBB1120_75
LBB1120_61:
	cmp	w21, #7
	b.ls	LBB1120_65
	ldr	x24, [x23]
	str	x24, [x22]
	ldr	x24, [x23, #8]
	str	x24, [x22, #8]
	ldr	x24, [x23, #16]
	str	x24, [x22, #16]
	ldr	x24, [x23, #24]
	str	x24, [x22, #24]
	ldr	x23, [x23, #32]
	str	x23, [x22, #32]
	cmp	w19, #41
	b.lo	LBB1120_75
	add	x22, x22, #40
	neg	x21, x21
LBB1120_64:
	ldr	x23, [x22, x21]
	str	x23, [x22], #8
	cmp	x22, x20
	b.lo	LBB1120_64
	b	LBB1120_75
LBB1120_65:
	cmp	w21, #1
	b.ne	LBB1120_80
	ldrb	w21, [x23]
	cmp	w19, #31
	b.ls	LBB1120_72
	dup.16b	v0, w21
	add	x22, x12, x2
	add	x24, x13, x2
LBB1120_68:
	mov	x23, x24
	stp	q0, q0, [x22, #-16]
	add	x25, x22, #48
	add	x22, x22, #32
	add	x24, x24, #32
	cmp	x25, x20
	b.ls	LBB1120_68
	cmp	x22, x20
	b.ls	LBB1120_71
	sub	x22, x22, #16
	b	LBB1120_72
LBB1120_71:
	add	x22, x23, #16
	str	q0, [x23], #32
	cmp	x23, x20
	mov	x23, x22
	b.ls	LBB1120_71
LBB1120_72:
	cmp	x22, x20
	b.hs	LBB1120_75
	mul	x21, x21, x15
LBB1120_74:
	str	x21, [x22], #8
	cmp	x22, x20
	b.lo	LBB1120_74
LBB1120_75:
	bic	w20, w11, w7, lsr #3
	add	x9, x9, x20
	add	x2, x2, x19
	and	w7, w7, #0xff
	orr	w7, w7, #0x38
	cmp	x9, x8
	b.hs	LBB1120_3
LBB1120_76:
	add	x19, x2, #320
	cmp	x19, x3
	b.ls	LBB1120_6
	b	LBB1120_3
LBB1120_77:
	ldr	x24, [x23]
	str	x24, [x22]
	add	x22, x22, x21
	add	x23, x22, x21
	str	x24, [x22]
	str	x24, [x23]
	add	x22, x23, x21
	add	x23, x22, x21
	str	x24, [x22]
	cmp	x23, x20
	b.hs	LBB1120_75
LBB1120_78:
	ldr	x24, [x22]
	str	x24, [x23]
	add	x22, x22, x21
	add	x23, x23, x21
	cmp	x23, x20
	b.lo	LBB1120_78
	b	LBB1120_75
LBB1120_79:
	lsr	w20, w20, #16
	and	x17, x19, #0x7ff
	ldr	w6, [x4, x17, lsl #2]
	ldr	x17, [x10, x9]
	lsl	x17, x17, x7
	orr	x17, x17, x19
	bic	w19, w11, w7, lsr #3
	add	x9, x9, x19
	strb	w20, [x1, x21]
	add	x2, x2, #3
	and	w7, w7, #0xff
	orr	w7, w7, #0x38
	cmp	x9, x8
	b.lo	LBB1120_76
	b	LBB1120_3
LBB1120_80:
	ldr	x24, [x23]
	str	x24, [x22]
	add	x22, x22, x21
	add	x23, x22, x21
	str	x24, [x22]
	str	x24, [x23]
	add	x22, x23, x21
	add	x23, x22, x21
	str	x24, [x22]
	cmp	x23, x20
	b.hs	LBB1120_75
LBB1120_81:
	ldr	x24, [x22]
	str	x24, [x23]
	add	x22, x22, x21
	add	x23, x23, x21
	cmp	x23, x20
	b.lo	LBB1120_81
	b	LBB1120_75
LBB1120_82:
	str	x19, [x0, #24]
	b	LBB1120_87
LBB1120_83:
	mov	x8, #0
	str	x19, [x0, #24]
	b	LBB1120_85
LBB1120_84:
	mov	x8, #0
	str	x17, [x0, #24]
LBB1120_85:
	and	w7, w7, #0xff
	b	LBB1120_4
LBB1120_86:
	str	x6, [x0, #24]
LBB1120_87:
	and	w7, w7, #0xff
	mov	w8, #2
	b	LBB1120_4

	.globl	__ZN6gzippy10decompress7inflate20consume_first_decode4Bits11refill_slow17h37f32271dc607a29E
	.p2align	2
__ZN6gzippy10decompress7inflate20consume_first_decode4Bits11refill_slow17h37f32271dc607a29E:
	ldr	w1, [x0, #32]
	and	w8, w1, #0xff
	cmp	w8, #65
	b.lo	LBB1121_2
	mov	w1, #0
	str	xzr, [x0, #24]
LBB1121_2:
	b	__ZN6gzippy10decompress7inflate20consume_first_decode4Bits21refill_slow_with_bits17h8dd98e668ad5089fE

	.globl	__ZN6gzippy10decompress7inflate20consume_first_decode4Bits12bit_position17h34f6cf2a33820ba7E
	.p2align	2
__ZN6gzippy10decompress7inflate20consume_first_decode4Bits12bit_position17h34f6cf2a33820ba7E:
	.cfi_startproc
	ldr	w8, [x0, #32]
	ldr	x9, [x0, #16]
	lsl	x10, x9, #3
	lsr	x9, x9, #61
	mov	x11, #-1
	cmp	x9, #0
	csel	x9, x10, x11, eq
	and	w10, w8, #0xff
	tst	w8, #0xc0
	mov	w8, #64
	csel	w8, w10, w8, eq
	subs	x8, x9, x8
	csel	x0, xzr, x8, lo
	ret
	.cfi_endproc
