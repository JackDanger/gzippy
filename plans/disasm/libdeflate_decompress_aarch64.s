
deflate_decompress.c.o:     file format mach-o-arm64


Disassembly of section .text:

0000000000000000 <_libdeflate_deflate_decompress_ex>:
   0:	sub	sp, sp, #0xd0
   4:	stp	x28, x27, [sp, #112]
   8:	stp	x26, x25, [sp, #128]
   c:	stp	x24, x23, [sp, #144]
  10:	stp	x22, x21, [sp, #160]
  14:	stp	x20, x19, [sp, #176]
  18:	stp	x29, x30, [sp, #192]
  1c:	add	x29, sp, #0xc0
  20:	stp	x5, x6, [sp, #16]
  24:	mov	x28, x3
  28:	mov	x26, x1
  2c:	mov	x23, x0
  30:	mov	x6, #0x0                   	// #0
  34:	mov	w9, #0x0                   	// #0
  38:	mov	x3, #0x0                   	// #0
  3c:	mov	w8, #0x12b                 	// #299
  40:	cmp	x4, #0x12b
  44:	csel	x8, x4, x8, cc	// cc = lo, ul, last
  48:	add	x21, x28, x4
  4c:	sub	x4, x21, x8
  50:	mov	w8, #0x19                  	// #25
  54:	cmp	x2, #0x19
  58:	csel	x8, x2, x8, cc	// cc = lo, ul, last
  5c:	add	x22, x1, x2
  60:	sub	x2, x22, x8
  64:	add	x8, x0, #0x1cc
  68:	str	x8, [sp, #88]
  6c:	mov	w8, #0x2498                	// #9368
  70:	add	x25, x0, x8
  74:	mov	w5, #0x2d24                	// #11556
  78:	mov	x24, #0xffffffffffffffff    	// #-1
  7c:	mov	w7, #0xdf                  	// #223
  80:	str	x1, [sp, #8]
  84:	mov	x20, x28
  88:	str	x2, [sp, #64]
  8c:	sub	x8, x22, x26
  90:	cmp	x8, #0x7
  94:	b.ls	9d8 <_libdeflate_deflate_decompress_ex+0x9d8>  // b.plast
  98:	ldr	x8, [x26]
  9c:	lsl	x8, x8, x9
  a0:	orr	x3, x8, x3
  a4:	ubfx	w8, w9, #3, #3
  a8:	sub	x8, x26, x8
  ac:	add	x26, x8, #0x7
  b0:	orr	w9, w9, #0x38
  b4:	ubfx	w8, w3, #1, #2
  b8:	cmp	w8, #0x1
  bc:	b.gt	158 <_libdeflate_deflate_decompress_ex+0x158>
  c0:	cbnz	w8, 40c <_libdeflate_deflate_decompress_ex+0x40c>
  c4:	add	w8, w9, #0xfd
  c8:	ubfx	w8, w8, #3, #5
  cc:	subs	x8, x6, x8
  d0:	b.hi	ab4 <_libdeflate_deflate_decompress_ex+0xab4>  // b.pmore
  d4:	add	x8, x26, x8
  d8:	sub	x9, x22, x8
  dc:	cmp	x9, #0x4
  e0:	b.lt	ab4 <_libdeflate_deflate_decompress_ex+0xab4>  // b.tstop
  e4:	ldrh	w19, [x8]
  e8:	ldrh	w9, [x8, #2]
  ec:	eor	w9, w9, w19
  f0:	mov	w10, #0xffff                	// #65535
  f4:	cmp	w9, w10
  f8:	b.ne	ab4 <_libdeflate_deflate_decompress_ex+0xab4>  // b.any
  fc:	sub	x9, x21, x20
 100:	cmp	x9, x19
 104:	b.lt	a74 <_libdeflate_deflate_decompress_ex+0xa74>  // b.tstop
 108:	add	x26, x8, #0x4
 10c:	sub	x8, x22, x26
 110:	cmp	x8, x19
 114:	b.lt	ab4 <_libdeflate_deflate_decompress_ex+0xab4>  // b.tstop
 118:	mov	x0, x20
 11c:	mov	x1, x26
 120:	mov	x2, x19
 124:	str	x3, [sp, #72]
 128:	mov	x27, x4
 12c:	bl	0 <_memcpy>
 130:	mov	w7, #0xdf                  	// #223
 134:	mov	w5, #0x2d24                	// #11556
 138:	ldp	x2, x3, [sp, #64]
 13c:	mov	x4, x27
 140:	mov	x6, #0x0                   	// #0
 144:	mov	w9, #0x0                   	// #0
 148:	mov	x10, #0x0                   	// #0
 14c:	add	x26, x26, x19
 150:	add	x20, x20, x19
 154:	b	9c8 <_libdeflate_deflate_decompress_ex+0x9c8>
 158:	cmp	w8, #0x2
 15c:	b.ne	ab4 <_libdeflate_deflate_decompress_ex+0xab4>  // b.any
 160:	mov	w13, #0x2ae0                	// #10976
 164:	mov	w8, #0x2d20                	// #11552
 168:	strb	wzr, [x23, x8]
 16c:	ubfx	w8, w3, #17, #3
 170:	strb	w8, [x23, #16]
 174:	lsr	x27, x3, #20
 178:	sub	w19, w9, #0x14
 17c:	sub	x8, x22, x26
 180:	cmp	x8, #0x7
 184:	b.ls	a1c <_libdeflate_deflate_decompress_ex+0xa1c>  // b.plast
 188:	ldr	x8, [x26]
 18c:	lsl	x8, x8, x19
 190:	orr	x27, x8, x27
 194:	ubfx	w8, w19, #3, #3
 198:	sub	x8, x26, x8
 19c:	add	x26, x8, #0x7
 1a0:	orr	w19, w19, #0x38
 1a4:	adrp	x14, f88 <_deflate_decompress_default.deflate_precode_lens_permutation>
 1a8:	add	x14, x14, #0x0
 1ac:	mov	x9, #0x0                   	// #0
 1b0:	ubfx	w8, w3, #3, #5
 1b4:	add	w8, w8, #0x101
 1b8:	str	w8, [sp, #84]
 1bc:	ubfx	w8, w3, #8, #5
 1c0:	add	w8, w8, #0x1
 1c4:	str	w8, [sp, #60]
 1c8:	ubfx	w8, w3, #13, #4
 1cc:	stur	w8, [x29, #-84]
 1d0:	mov	x8, x3
 1d4:	ubfx	x8, x8, #13, #4
 1d8:	add	x10, x8, #0x3
 1dc:	and	w11, w27, #0x7
 1e0:	add	x12, x14, x9
 1e4:	ldrb	w12, [x12, #1]
 1e8:	strb	w11, [x23, x12]
 1ec:	lsr	x27, x27, #3
 1f0:	add	x9, x9, #0x1
 1f4:	cmp	x10, x9
 1f8:	b.ne	1dc <_libdeflate_deflate_decompress_ex+0x1dc>  // b.any
 1fc:	stp	x4, x6, [sp, #40]
 200:	str	x3, [sp, #72]
 204:	cmp	x9, #0x11
 208:	b.hi	228 <_libdeflate_deflate_decompress_ex+0x228>  // b.pmore
 20c:	add	x9, x14, x8
 210:	ldrb	w9, [x9, #4]
 214:	strb	wzr, [x23, x9]
 218:	add	x9, x8, #0x4
 21c:	add	x8, x8, #0x1
 220:	cmp	x9, #0x12
 224:	b.cc	20c <_libdeflate_deflate_decompress_ex+0x20c>  // b.lo, b.ul, b.last
 228:	add	x6, x23, x13
 22c:	ldr	x0, [sp, #88]
 230:	mov	x1, x23
 234:	mov	w2, #0x13                  	// #19
 238:	adrp	x3, f9c <_precode_decode_results>
 23c:	add	x3, x3, #0x0
 240:	mov	w4, #0x7                   	// #7
 244:	mov	w5, #0x7                   	// #7
 248:	mov	x7, #0x0                   	// #0
 24c:	bl	b94 <_build_decode_table>
 250:	cbz	w0, ab4 <_libdeflate_deflate_decompress_ex+0xab4>
 254:	str	x28, [sp, #32]
 258:	mov	w28, #0x0                   	// #0
 25c:	ldur	w8, [x29, #-84]
 260:	sub	w8, w8, w8, lsl #2
 264:	add	w8, w8, w19
 268:	sub	w19, w8, #0x9
 26c:	ldr	w8, [sp, #84]
 270:	ldr	w9, [sp, #60]
 274:	add	w11, w8, w9
 278:	ldr	x10, [sp, #88]
 27c:	stur	w11, [x29, #-84]
 280:	and	w8, w19, #0xff
 284:	cmp	w8, #0xd
 288:	b.hi	2b4 <_libdeflate_deflate_decompress_ex+0x2b4>  // b.pmore
 28c:	sub	x9, x22, x26
 290:	cmp	x9, #0x7
 294:	b.ls	3cc <_libdeflate_deflate_decompress_ex+0x3cc>  // b.plast
 298:	ldr	x9, [x26]
 29c:	lsl	x8, x9, x8
 2a0:	orr	x27, x8, x27
 2a4:	ubfx	w8, w19, #3, #3
 2a8:	sub	x8, x26, x8
 2ac:	add	x26, x8, #0x7
 2b0:	orr	w19, w19, #0x38
 2b4:	and	x8, x27, #0x7f
 2b8:	ldr	w8, [x10, x8, lsl #2]
 2bc:	lsr	x27, x27, x8
 2c0:	sub	w19, w19, w8
 2c4:	ubfx	x9, x8, #20, #12
 2c8:	lsr	w8, w8, #16
 2cc:	cbnz	w9, 2dc <_libdeflate_deflate_decompress_ex+0x2dc>
 2d0:	strb	w8, [x23, w28, uxtw]
 2d4:	add	w28, w28, #0x1
 2d8:	b	3c0 <_libdeflate_deflate_decompress_ex+0x3c0>
 2dc:	cmp	w8, #0x11
 2e0:	b.eq	338 <_libdeflate_deflate_decompress_ex+0x338>  // b.none
 2e4:	cmp	w8, #0x10
 2e8:	b.ne	398 <_libdeflate_deflate_decompress_ex+0x398>  // b.any
 2ec:	cbz	w28, ab4 <_libdeflate_deflate_decompress_ex+0xab4>
 2f0:	sub	w8, w28, #0x1
 2f4:	ldrb	w8, [x23, w8, uxtw]
 2f8:	and	w9, w27, #0x3
 2fc:	add	w12, w9, #0x3
 300:	lsr	x27, x27, #2
 304:	sub	w19, w19, #0x2
 308:	strb	w8, [x23, w28, uxtw]
 30c:	add	w9, w28, #0x1
 310:	strb	w8, [x23, w9, uxtw]
 314:	add	w9, w28, #0x2
 318:	strb	w8, [x23, w9, uxtw]
 31c:	add	w9, w28, #0x3
 320:	strb	w8, [x23, w9, uxtw]
 324:	add	w9, w28, #0x4
 328:	strb	w8, [x23, w9, uxtw]
 32c:	add	w9, w28, #0x5
 330:	strb	w8, [x23, w9, uxtw]
 334:	b	3b8 <_libdeflate_deflate_decompress_ex+0x3b8>
 338:	and	w8, w27, #0x7
 33c:	add	w12, w8, #0x3
 340:	lsr	x27, x27, #3
 344:	sub	w19, w19, #0x3
 348:	strb	wzr, [x23, w28, uxtw]
 34c:	add	w8, w28, #0x1
 350:	strb	wzr, [x23, w8, uxtw]
 354:	add	w8, w28, #0x2
 358:	strb	wzr, [x23, w8, uxtw]
 35c:	add	w8, w28, #0x3
 360:	strb	wzr, [x23, w8, uxtw]
 364:	add	w8, w28, #0x4
 368:	strb	wzr, [x23, w8, uxtw]
 36c:	add	w8, w28, #0x5
 370:	strb	wzr, [x23, w8, uxtw]
 374:	add	w8, w28, #0x6
 378:	strb	wzr, [x23, w8, uxtw]
 37c:	add	w8, w28, #0x7
 380:	strb	wzr, [x23, w8, uxtw]
 384:	add	w8, w28, #0x8
 388:	strb	wzr, [x23, w8, uxtw]
 38c:	add	w8, w28, #0x9
 390:	strb	wzr, [x23, w8, uxtw]
 394:	b	3bc <_libdeflate_deflate_decompress_ex+0x3bc>
 398:	and	w8, w27, #0x7f
 39c:	add	w1, w8, #0xb
 3a0:	str	x1, [sp, #96]
 3a4:	lsr	x27, x27, #7
 3a8:	sub	w19, w19, #0x7
 3ac:	add	x0, x23, w28, uxtw
 3b0:	bl	0 <_bzero>
 3b4:	ldp	x10, x12, [sp, #88]
 3b8:	ldur	w11, [x29, #-84]
 3bc:	add	w28, w12, w28
 3c0:	cmp	w28, w11
 3c4:	b.cc	280 <_libdeflate_deflate_decompress_ex+0x280>  // b.lo, b.ul, b.last
 3c8:	b	484 <_libdeflate_deflate_decompress_ex+0x484>
 3cc:	cmp	x26, x22
 3d0:	b.eq	3f4 <_libdeflate_deflate_decompress_ex+0x3f4>  // b.none
 3d4:	ldrb	w9, [x26], #1
 3d8:	lsl	x8, x9, x8
 3dc:	orr	x27, x8, x27
 3e0:	add	w19, w19, #0x8
 3e4:	and	w8, w19, #0xff
 3e8:	cmp	w8, #0x38
 3ec:	b.cc	3cc <_libdeflate_deflate_decompress_ex+0x3cc>  // b.lo, b.ul, b.last
 3f0:	b	2b4 <_libdeflate_deflate_decompress_ex+0x2b4>
 3f4:	ldr	x8, [sp, #48]
 3f8:	add	x8, x8, #0x1
 3fc:	str	x8, [sp, #48]
 400:	cmp	x8, #0x8
 404:	b.ls	3e0 <_libdeflate_deflate_decompress_ex+0x3e0>  // b.plast
 408:	b	ab4 <_libdeflate_deflate_decompress_ex+0xab4>
 40c:	lsr	x27, x3, #3
 410:	sub	w19, w9, #0x3
 414:	mov	w8, #0x2d20                	// #11552
 418:	ldrb	w8, [x23, x8]
 41c:	tbnz	w8, #0, 508 <_libdeflate_deflate_decompress_ex+0x508>
 420:	stp	x4, x6, [sp, #40]
 424:	str	x3, [sp, #72]
 428:	mov	w8, #0x2d20                	// #11552
 42c:	mov	w9, #0x1                   	// #1
 430:	strb	w9, [x23, x8]
 434:	movi	v1.16b, #0x8
 438:	stp	q1, q1, [x23]
 43c:	stp	q1, q1, [x23, #32]
 440:	stp	q1, q1, [x23, #64]
 444:	stp	q1, q1, [x23, #96]
 448:	movi	v0.16b, #0x9
 44c:	stp	q1, q0, [x23, #128]
 450:	stp	q0, q0, [x23, #160]
 454:	stp	q0, q0, [x23, #192]
 458:	stp	q0, q0, [x23, #224]
 45c:	mov	x9, #0x707070707070707     	// #506381209866536711
 460:	stp	x9, x9, [x23, #256]
 464:	mov	x8, #0x808080808080808     	// #578721382704613384
 468:	stp	x9, x8, [x23, #272]
 46c:	mov	w8, #0x20                  	// #32
 470:	str	w8, [sp, #60]
 474:	mov	w8, #0x120                 	// #288
 478:	movi	v0.16b, #0x5
 47c:	stp	q0, q0, [x23, #288]
 480:	b	490 <_libdeflate_deflate_decompress_ex+0x490>
 484:	ldr	x28, [sp, #32]
 488:	ldr	w8, [sp, #84]
 48c:	b.ne	ab4 <_libdeflate_deflate_decompress_ex+0xab4>  // b.any
 490:	str	w8, [sp, #84]
 494:	add	x1, x23, w8, uxtw
 498:	mov	w9, #0x2ae0                	// #10976
 49c:	add	x6, x23, x9
 4a0:	mov	x0, x25
 4a4:	ldr	w2, [sp, #60]
 4a8:	adrp	x3, fe8 <_offset_decode_results>
 4ac:	add	x3, x3, #0x0
 4b0:	mov	w4, #0x8                   	// #8
 4b4:	mov	w5, #0xf                   	// #15
 4b8:	mov	x7, #0x0                   	// #0
 4bc:	bl	b94 <_build_decode_table>
 4c0:	cbz	w0, ab4 <_libdeflate_deflate_decompress_ex+0xab4>
 4c4:	mov	w8, #0x2ae0                	// #10976
 4c8:	add	x6, x23, x8
 4cc:	mov	w8, #0x2d24                	// #11556
 4d0:	add	x7, x23, x8
 4d4:	mov	x0, x23
 4d8:	mov	x1, x23
 4dc:	ldr	w2, [sp, #84]
 4e0:	adrp	x3, 1068 <_litlen_decode_results>
 4e4:	add	x3, x3, #0x0
 4e8:	mov	w4, #0xb                   	// #11
 4ec:	mov	w5, #0xf                   	// #15
 4f0:	bl	b94 <_build_decode_table>
 4f4:	mov	w5, #0x2d24                	// #11556
 4f8:	ldp	x2, x3, [sp, #64]
 4fc:	ldp	x4, x6, [sp, #40]
 500:	mov	w7, #0xdf                  	// #223
 504:	cbz	w0, ab4 <_libdeflate_deflate_decompress_ex+0xab4>
 508:	ldr	w8, [x23, x5]
 50c:	lsl	x8, x24, x8
 510:	mvn	x8, x8
 514:	cmp	x26, x2
 518:	b.cs	84c <_libdeflate_deflate_decompress_ex+0x84c>  // b.hs, b.nlast
 51c:	cmp	x20, x4
 520:	b.cs	84c <_libdeflate_deflate_decompress_ex+0x84c>  // b.hs, b.nlast
 524:	ldr	x9, [x26]
 528:	lsl	x9, x9, x19
 52c:	orr	x27, x9, x27
 530:	ubfx	w9, w19, #3, #3
 534:	sub	x9, x26, x9
 538:	add	x26, x9, #0x7
 53c:	orr	w19, w19, #0x38
 540:	and	x9, x27, x8
 544:	ldr	w10, [x23, x9, lsl #2]
 548:	lsr	x12, x27, x10
 54c:	sub	w9, w19, w10
 550:	tbnz	w10, #31, 6a4 <_libdeflate_deflate_decompress_ex+0x6a4>
 554:	mov	x11, x10
 558:	mov	x10, x12
 55c:	tbnz	w11, #15, 6d0 <_libdeflate_deflate_decompress_ex+0x6d0>
 560:	and	w12, w11, #0xff
 564:	and	x13, x10, #0xff
 568:	ldr	w13, [x25, x13, lsl #2]
 56c:	and	w14, w9, #0xff
 570:	tbnz	w13, #15, 7b8 <_libdeflate_deflate_decompress_ex+0x7b8>
 574:	cmp	w14, #0x1e
 578:	b.ls	7e0 <_libdeflate_deflate_decompress_ex+0x7e0>  // b.plast
 57c:	mov	x14, x13
 580:	lsl	x13, x24, x14
 584:	bic	x13, x10, x13
 588:	lsr	w15, w14, #8
 58c:	lsr	x13, x13, x15
 590:	add	w13, w13, w14, lsr #16
 594:	sub	x15, x20, x28
 598:	cmp	x15, x13
 59c:	b.lt	ab4 <_libdeflate_deflate_decompress_ex+0xab4>  // b.tstop
 5a0:	lsl	x12, x24, x12
 5a4:	bic	x12, x27, x12
 5a8:	lsr	w15, w11, #8
 5ac:	lsr	x12, x12, x15
 5b0:	add	w12, w12, w11, lsr #16
 5b4:	and	w11, w14, #0xff
 5b8:	sub	w9, w9, w14
 5bc:	ldr	x14, [x26]
 5c0:	lsl	x15, x14, x9
 5c4:	ubfx	w16, w9, #3, #3
 5c8:	lsr	x17, x10, x11
 5cc:	sub	x14, x20, x13
 5d0:	add	x11, x20, w12, uxtw
 5d4:	and	x10, x17, x8
 5d8:	ldr	w10, [x23, x10, lsl #2]
 5dc:	orr	x27, x15, x17
 5e0:	sub	x15, x26, x16
 5e4:	add	x26, x15, #0x7
 5e8:	cmp	w13, #0x8
 5ec:	b.cc	664 <_libdeflate_deflate_decompress_ex+0x664>  // b.lo, b.ul, b.last
 5f0:	ldr	x15, [x14]
 5f4:	str	x15, [x20]
 5f8:	ldr	x15, [x14, #8]
 5fc:	str	x15, [x20, #8]
 600:	ldr	x15, [x14, #16]
 604:	str	x15, [x20, #16]
 608:	ldr	x15, [x14, #24]
 60c:	str	x15, [x20, #24]
 610:	ldr	x14, [x14, #32]
 614:	str	x14, [x20, #32]
 618:	cmp	w12, #0x29
 61c:	b.cc	744 <_libdeflate_deflate_decompress_ex+0x744>  // b.lo, b.ul, b.last
 620:	neg	x12, x13
 624:	add	x13, x20, #0x28
 628:	add	x14, x13, x12
 62c:	ldr	x15, [x14]
 630:	str	x15, [x13]
 634:	ldr	x15, [x14, #8]
 638:	str	x15, [x13, #8]
 63c:	ldr	x15, [x14, #16]
 640:	str	x15, [x13, #16]
 644:	ldr	x15, [x14, #24]
 648:	str	x15, [x13, #24]
 64c:	ldr	x14, [x14, #32]
 650:	str	x14, [x13, #32]
 654:	add	x13, x13, #0x28
 658:	cmp	x13, x11
 65c:	b.cc	628 <_libdeflate_deflate_decompress_ex+0x628>  // b.lo, b.ul, b.last
 660:	b	744 <_libdeflate_deflate_decompress_ex+0x744>
 664:	cmp	w13, #0x1
 668:	b.ne	70c <_libdeflate_deflate_decompress_ex+0x70c>  // b.any
 66c:	ldrb	w13, [x14]
 670:	mov	x14, #0x101010101010101     	// #72340172838076673
 674:	mul	x13, x13, x14
 678:	stp	x13, x13, [x20]
 67c:	stp	x13, x13, [x20, #16]
 680:	cmp	w12, #0x21
 684:	b.cc	744 <_libdeflate_deflate_decompress_ex+0x744>  // b.lo, b.ul, b.last
 688:	add	x12, x20, #0x20
 68c:	stp	x13, x13, [x12]
 690:	stp	x13, x13, [x12, #16]
 694:	add	x12, x12, #0x20
 698:	cmp	x12, x11
 69c:	b.cc	68c <_libdeflate_deflate_decompress_ex+0x68c>  // b.lo, b.ul, b.last
 6a0:	b	744 <_libdeflate_deflate_decompress_ex+0x744>
 6a4:	lsr	w14, w10, #16
 6a8:	and	x10, x12, x8
 6ac:	ldr	w11, [x23, x10, lsl #2]
 6b0:	lsr	x10, x12, x11
 6b4:	sub	w9, w9, w11
 6b8:	mov	x13, x20
 6bc:	strb	w14, [x13], #1
 6c0:	tbnz	w11, #31, 760 <_libdeflate_deflate_decompress_ex+0x760>
 6c4:	mov	x27, x12
 6c8:	mov	x20, x13
 6cc:	tbz	w11, #15, 560 <_libdeflate_deflate_decompress_ex+0x560>
 6d0:	tbnz	w11, #13, a68 <_libdeflate_deflate_decompress_ex+0xa68>
 6d4:	lsr	w12, w11, #16
 6d8:	ubfx	w11, w11, #8, #5
 6dc:	lsl	x11, x24, x11
 6e0:	bic	x11, x10, x11
 6e4:	add	x11, x11, x12
 6e8:	ldr	w11, [x23, x11, lsl #2]
 6ec:	lsr	x13, x10, x11
 6f0:	sub	w9, w9, w11
 6f4:	tbnz	w11, #31, 800 <_libdeflate_deflate_decompress_ex+0x800>
 6f8:	tbnz	w11, #13, a6c <_libdeflate_deflate_decompress_ex+0xa6c>
 6fc:	and	x12, x11, #0xff
 700:	mov	x27, x10
 704:	mov	x10, x13
 708:	b	564 <_libdeflate_deflate_decompress_ex+0x564>
 70c:	ldr	x12, [x14]
 710:	str	x12, [x20]
 714:	str	x12, [x20, x13]
 718:	lsl	x12, x13, #1
 71c:	add	x14, x13, x13, lsl #1
 720:	add	x15, x20, x13
 724:	add	x16, x15, x13
 728:	ldr	x15, [x15]
 72c:	str	x15, [x20, x12]
 730:	str	x15, [x20, x14]
 734:	add	x15, x16, x12
 738:	cmp	x15, x11
 73c:	mov	x20, x16
 740:	b.cc	720 <_libdeflate_deflate_decompress_ex+0x720>  // b.lo, b.ul, b.last
 744:	mov	x20, x11
 748:	orr	w19, w9, #0x38
 74c:	cmp	x26, x2
 750:	b.cs	84c <_libdeflate_deflate_decompress_ex+0x84c>  // b.hs, b.nlast
 754:	cmp	x20, x4
 758:	b.cc	548 <_libdeflate_deflate_decompress_ex+0x548>  // b.lo, b.ul, b.last
 75c:	b	84c <_libdeflate_deflate_decompress_ex+0x84c>
 760:	lsr	w13, w11, #16
 764:	and	x11, x10, x8
 768:	ldr	w11, [x23, x11, lsl #2]
 76c:	lsr	x12, x10, x11
 770:	sub	w9, w9, w11
 774:	strb	w13, [x20, #1]
 778:	tbnz	w11, #31, 788 <_libdeflate_deflate_decompress_ex+0x788>
 77c:	add	x20, x20, #0x2
 780:	mov	x27, x10
 784:	b	558 <_libdeflate_deflate_decompress_ex+0x558>
 788:	lsr	w11, w11, #16
 78c:	and	x10, x12, x8
 790:	ldr	w10, [x23, x10, lsl #2]
 794:	ldr	x13, [x26]
 798:	lsl	x13, x13, x9
 79c:	orr	x27, x13, x12
 7a0:	ubfx	w12, w9, #3, #3
 7a4:	sub	x12, x26, x12
 7a8:	add	x26, x12, #0x7
 7ac:	strb	w11, [x20, #2]
 7b0:	add	x20, x20, #0x3
 7b4:	b	748 <_libdeflate_deflate_decompress_ex+0x748>
 7b8:	cmp	w14, #0x25
 7bc:	b.ls	82c <_libdeflate_deflate_decompress_ex+0x82c>  // b.plast
 7c0:	lsr	x10, x10, #8
 7c4:	sub	w9, w9, #0x8
 7c8:	lsr	x14, x13, #8
 7cc:	lsl	x14, x24, x14
 7d0:	bic	x14, x10, x14
 7d4:	add	x13, x14, x13, lsr #16
 7d8:	ldr	w14, [x25, x13, lsl #2]
 7dc:	b	580 <_libdeflate_deflate_decompress_ex+0x580>
 7e0:	ldr	x15, [x26]
 7e4:	lsl	x14, x15, x14
 7e8:	orr	x10, x14, x10
 7ec:	ubfx	w14, w9, #3, #3
 7f0:	sub	x14, x26, x14
 7f4:	add	x26, x14, #0x7
 7f8:	orr	w9, w9, #0x38
 7fc:	b	57c <_libdeflate_deflate_decompress_ex+0x57c>
 800:	lsr	w11, w11, #16
 804:	and	x10, x13, x8
 808:	ldr	w10, [x23, x10, lsl #2]
 80c:	ldr	x12, [x26]
 810:	lsl	x12, x12, x9
 814:	orr	x27, x12, x13
 818:	ubfx	w12, w9, #3, #3
 81c:	sub	x12, x26, x12
 820:	add	x26, x12, #0x7
 824:	strb	w11, [x20], #1
 828:	b	748 <_libdeflate_deflate_decompress_ex+0x748>
 82c:	ldr	x15, [x26]
 830:	lsl	x14, x15, x14
 834:	orr	x10, x14, x10
 838:	ubfx	w14, w9, #3, #3
 83c:	sub	x14, x26, x14
 840:	add	x26, x14, #0x7
 844:	orr	w9, w9, #0x38
 848:	b	7c0 <_libdeflate_deflate_decompress_ex+0x7c0>
 84c:	sub	x9, x22, x26
 850:	cmp	x9, #0x7
 854:	b.ls	93c <_libdeflate_deflate_decompress_ex+0x93c>  // b.plast
 858:	ldr	x9, [x26]
 85c:	lsl	x9, x9, x19
 860:	orr	x27, x9, x27
 864:	ubfx	w9, w19, #3, #3
 868:	sub	x9, x26, x9
 86c:	add	x26, x9, #0x7
 870:	orr	w19, w19, #0x38
 874:	and	x9, x27, x8
 878:	ldr	w9, [x23, x9, lsl #2]
 87c:	lsr	x10, x27, x9
 880:	sub	w19, w19, w9
 884:	tbnz	w9, #14, 980 <_libdeflate_deflate_decompress_ex+0x980>
 888:	lsr	w11, w9, #16
 88c:	tbnz	w9, #31, 928 <_libdeflate_deflate_decompress_ex+0x928>
 890:	tbnz	w9, #13, 9c4 <_libdeflate_deflate_decompress_ex+0x9c4>
 894:	lsl	x12, x24, x9
 898:	bic	x12, x27, x12
 89c:	and	w9, w7, w9, lsr #8
 8a0:	lsr	x9, x12, x9
 8a4:	add	w9, w11, w9
 8a8:	sub	x11, x21, x20
 8ac:	cmp	x11, x9
 8b0:	b.lt	a74 <_libdeflate_deflate_decompress_ex+0xa74>  // b.tstop
 8b4:	and	x11, x10, #0xff
 8b8:	ldr	w11, [x25, x11, lsl #2]
 8bc:	tbnz	w11, #15, 9a4 <_libdeflate_deflate_decompress_ex+0x9a4>
 8c0:	lsl	x12, x24, x11
 8c4:	bic	x12, x10, x12
 8c8:	lsr	w13, w11, #8
 8cc:	lsr	x12, x12, x13
 8d0:	add	w12, w12, w11, lsr #16
 8d4:	sub	x13, x20, x28
 8d8:	cmp	x13, x12
 8dc:	b.lt	ab4 <_libdeflate_deflate_decompress_ex+0xab4>  // b.tstop
 8e0:	lsr	x27, x10, x11
 8e4:	sub	w19, w19, w11
 8e8:	neg	x10, x12
 8ec:	sub	x11, x20, x12
 8f0:	add	x9, x20, x9
 8f4:	ldrb	w12, [x11]
 8f8:	strb	w12, [x20]
 8fc:	ldrb	w12, [x11, #1]
 900:	add	x11, x20, #0x2
 904:	strb	w12, [x20, #1]
 908:	ldrb	w12, [x11, x10]
 90c:	strb	w12, [x11]
 910:	add	x12, x11, #0x1
 914:	mov	x11, x12
 918:	cmp	x12, x9
 91c:	b.cc	908 <_libdeflate_deflate_decompress_ex+0x908>  // b.lo, b.ul, b.last
 920:	mov	x20, x9
 924:	b	84c <_libdeflate_deflate_decompress_ex+0x84c>
 928:	cmp	x20, x21
 92c:	b.eq	a74 <_libdeflate_deflate_decompress_ex+0xa74>  // b.none
 930:	strb	w11, [x20], #1
 934:	mov	x27, x10
 938:	b	84c <_libdeflate_deflate_decompress_ex+0x84c>
 93c:	and	w9, w19, #0xff
 940:	cmp	w9, #0x37
 944:	b.hi	874 <_libdeflate_deflate_decompress_ex+0x874>  // b.pmore
 948:	cmp	x26, x22
 94c:	b.eq	970 <_libdeflate_deflate_decompress_ex+0x970>  // b.none
 950:	ldrb	w10, [x26], #1
 954:	lsl	x9, x10, x9
 958:	orr	x27, x9, x27
 95c:	add	w19, w19, #0x8
 960:	and	w9, w19, #0xff
 964:	cmp	w9, #0x38
 968:	b.cc	948 <_libdeflate_deflate_decompress_ex+0x948>  // b.lo, b.ul, b.last
 96c:	b	874 <_libdeflate_deflate_decompress_ex+0x874>
 970:	add	x6, x6, #0x1
 974:	cmp	x6, #0x8
 978:	b.ls	95c <_libdeflate_deflate_decompress_ex+0x95c>  // b.plast
 97c:	b	ab4 <_libdeflate_deflate_decompress_ex+0xab4>
 980:	lsr	x11, x9, #8
 984:	lsl	x11, x24, x11
 988:	bic	x11, x10, x11
 98c:	add	x9, x11, x9, lsr #16
 990:	ldr	w9, [x23, x9, lsl #2]
 994:	mov	x27, x10
 998:	lsr	x10, x10, x9
 99c:	sub	w19, w19, w9
 9a0:	b	888 <_libdeflate_deflate_decompress_ex+0x888>
 9a4:	lsr	x10, x10, #8
 9a8:	sub	w19, w19, #0x8
 9ac:	lsr	x12, x11, #8
 9b0:	lsl	x12, x24, x12
 9b4:	bic	x12, x10, x12
 9b8:	add	x11, x12, x11, lsr #16
 9bc:	ldr	w11, [x25, x11, lsl #2]
 9c0:	b	8c0 <_libdeflate_deflate_decompress_ex+0x8c0>
 9c4:	mov	x9, x19
 9c8:	mov	x8, x3
 9cc:	mov	x3, x10
 9d0:	tbz	w8, #0, 8c <_libdeflate_deflate_decompress_ex+0x8c>
 9d4:	b	a7c <_libdeflate_deflate_decompress_ex+0xa7c>
 9d8:	and	w8, w9, #0xff
 9dc:	cmp	w8, #0x37
 9e0:	b.hi	b4 <_libdeflate_deflate_decompress_ex+0xb4>  // b.pmore
 9e4:	cmp	x26, x22
 9e8:	b.eq	a0c <_libdeflate_deflate_decompress_ex+0xa0c>  // b.none
 9ec:	ldrb	w10, [x26], #1
 9f0:	lsl	x8, x10, x8
 9f4:	orr	x3, x8, x3
 9f8:	add	w9, w9, #0x8
 9fc:	and	w8, w9, #0xff
 a00:	cmp	w8, #0x38
 a04:	b.cc	9e4 <_libdeflate_deflate_decompress_ex+0x9e4>  // b.lo, b.ul, b.last
 a08:	b	b4 <_libdeflate_deflate_decompress_ex+0xb4>
 a0c:	add	x6, x6, #0x1
 a10:	cmp	x6, #0x8
 a14:	b.ls	9f8 <_libdeflate_deflate_decompress_ex+0x9f8>  // b.plast
 a18:	b	ab4 <_libdeflate_deflate_decompress_ex+0xab4>
 a1c:	and	w8, w19, #0xff
 a20:	cmp	w8, #0x37
 a24:	adrp	x14, f88 <_deflate_decompress_default.deflate_precode_lens_permutation>
 a28:	add	x14, x14, #0x0
 a2c:	b.hi	1ac <_libdeflate_deflate_decompress_ex+0x1ac>  // b.pmore
 a30:	cmp	x26, x22
 a34:	b.eq	a58 <_libdeflate_deflate_decompress_ex+0xa58>  // b.none
 a38:	ldrb	w9, [x26], #1
 a3c:	lsl	x8, x9, x8
 a40:	orr	x27, x8, x27
 a44:	add	w19, w19, #0x8
 a48:	and	w8, w19, #0xff
 a4c:	cmp	w8, #0x38
 a50:	b.cc	a30 <_libdeflate_deflate_decompress_ex+0xa30>  // b.lo, b.ul, b.last
 a54:	b	1ac <_libdeflate_deflate_decompress_ex+0x1ac>
 a58:	add	x6, x6, #0x1
 a5c:	cmp	x6, #0x8
 a60:	b.ls	a44 <_libdeflate_deflate_decompress_ex+0xa44>  // b.plast
 a64:	b	ab4 <_libdeflate_deflate_decompress_ex+0xab4>
 a68:	b	9c8 <_libdeflate_deflate_decompress_ex+0x9c8>
 a6c:	mov	x10, x13
 a70:	b	9c8 <_libdeflate_deflate_decompress_ex+0x9c8>
 a74:	mov	w0, #0x3                   	// #3
 a78:	b	ab8 <_libdeflate_deflate_decompress_ex+0xab8>
 a7c:	ubfx	w8, w9, #3, #5
 a80:	subs	x8, x6, x8
 a84:	b.hi	ab4 <_libdeflate_deflate_decompress_ex+0xab4>  // b.pmore
 a88:	ldp	x10, x9, [sp, #16]
 a8c:	cbz	x10, aa0 <_libdeflate_deflate_decompress_ex+0xaa0>
 a90:	add	x8, x26, x8
 a94:	ldr	x11, [sp, #8]
 a98:	sub	x8, x8, x11
 a9c:	str	x8, [x10]
 aa0:	cbz	x9, ad8 <_libdeflate_deflate_decompress_ex+0xad8>
 aa4:	sub	x8, x20, x28
 aa8:	str	x8, [x9]
 aac:	mov	w0, #0x0                   	// #0
 ab0:	b	ab8 <_libdeflate_deflate_decompress_ex+0xab8>
 ab4:	mov	w0, #0x1                   	// #1
 ab8:	ldp	x29, x30, [sp, #192]
 abc:	ldp	x20, x19, [sp, #176]
 ac0:	ldp	x22, x21, [sp, #160]
 ac4:	ldp	x24, x23, [sp, #144]
 ac8:	ldp	x26, x25, [sp, #128]
 acc:	ldp	x28, x27, [sp, #112]
 ad0:	add	sp, sp, #0xd0
 ad4:	ret
 ad8:	cmp	x20, x21
 adc:	b.eq	aac <_libdeflate_deflate_decompress_ex+0xaac>  // b.none
 ae0:	mov	w0, #0x2                   	// #2
 ae4:	b	ab8 <_libdeflate_deflate_decompress_ex+0xab8>

0000000000000ae8 <_libdeflate_deflate_decompress>:
 ae8:	mov	x6, x5
 aec:	mov	x5, #0x0                   	// #0
 af0:	b	0 <_libdeflate_deflate_decompress_ex>

0000000000000af4 <_libdeflate_alloc_decompressor_ex>:
 af4:	stp	x20, x19, [sp, #-32]!
 af8:	stp	x29, x30, [sp, #16]
 afc:	add	x29, sp, #0x10
 b00:	ldr	x8, [x0]
 b04:	cmp	x8, #0x18
 b08:	b.ne	b64 <_libdeflate_alloc_decompressor_ex+0x70>  // b.any
 b0c:	mov	x19, x0
 b10:	ldr	x8, [x0, #8]
 b14:	adrp	x9, 0 <_libdeflate_default_malloc_func>
 b18:	ldr	x9, [x9]
 b1c:	ldr	x9, [x9]
 b20:	cmp	x8, #0x0
 b24:	csel	x8, x9, x8, eq	// eq = none
 b28:	mov	w0, #0x2d30                	// #11568
 b2c:	blr	x8
 b30:	mov	x20, x0
 b34:	cbz	x0, b68 <_libdeflate_alloc_decompressor_ex+0x74>
 b38:	mov	x0, x20
 b3c:	mov	w1, #0x2d30                	// #11568
 b40:	bl	0 <_bzero>
 b44:	ldr	x8, [x19, #16]
 b48:	adrp	x9, 0 <_libdeflate_default_free_func>
 b4c:	ldr	x9, [x9]
 b50:	ldr	x9, [x9]
 b54:	cmp	x8, #0x0
 b58:	csel	x8, x9, x8, eq	// eq = none
 b5c:	str	x8, [x20, #11560]
 b60:	b	b68 <_libdeflate_alloc_decompressor_ex+0x74>
 b64:	mov	x20, #0x0                   	// #0
 b68:	mov	x0, x20
 b6c:	ldp	x29, x30, [sp, #16]
 b70:	ldp	x20, x19, [sp], #32
 b74:	ret

0000000000000b78 <_libdeflate_alloc_decompressor>:
 b78:	adrp	x0, f70 <_libdeflate_alloc_decompressor.defaults>
 b7c:	add	x0, x0, #0x0
 b80:	b	af4 <_libdeflate_alloc_decompressor_ex>

0000000000000b84 <_libdeflate_free_decompressor>:
 b84:	cbz	x0, b90 <_libdeflate_free_decompressor+0xc>
 b88:	ldr	x1, [x0, #11560]
 b8c:	br	x1
 b90:	ret

0000000000000b94 <_build_decode_table>:
 b94:	sub	sp, sp, #0xf0
 b98:	stp	x28, x27, [sp, #144]
 b9c:	stp	x26, x25, [sp, #160]
 ba0:	stp	x24, x23, [sp, #176]
 ba4:	stp	x22, x21, [sp, #192]
 ba8:	stp	x20, x19, [sp, #208]
 bac:	stp	x29, x30, [sp, #224]
 bb0:	add	x29, sp, #0xe0
 bb4:	mov	x26, x7
 bb8:	mov	x23, x6
 bbc:	mov	x22, x5
 bc0:	mov	x20, x4
 bc4:	mov	x21, x3
 bc8:	mov	x25, x2
 bcc:	mov	x24, x1
 bd0:	mov	x19, x0
 bd4:	adrp	x8, 0 <___stack_chk_guard>
 bd8:	ldr	x8, [x8]
 bdc:	ldr	x8, [x8]
 be0:	stur	x8, [x29, #-88]
 be4:	ubfiz	x8, x22, #2, #32
 be8:	add	x27, sp, #0x48
 bec:	add	x0, sp, #0x48
 bf0:	add	x1, x8, #0x4
 bf4:	bl	0 <_bzero>
 bf8:	cbz	w25, c1c <_build_decode_table+0x88>
 bfc:	mov	w8, w25
 c00:	mov	x9, x24
 c04:	ldrb	w10, [x9], #1
 c08:	ldr	w11, [x27, x10, lsl #2]
 c0c:	add	w11, w11, #0x1
 c10:	str	w11, [x27, x10, lsl #2]
 c14:	subs	x8, x8, #0x1
 c18:	b.ne	c04 <_build_decode_table+0x70>  // b.any
 c1c:	cmp	w22, #0x2
 c20:	b.cc	c40 <_build_decode_table+0xac>  // b.lo, b.ul, b.last
 c24:	add	x8, sp, #0x48
 c28:	ldr	w9, [x8, w22, uxtw #2]
 c2c:	cbnz	w9, c40 <_build_decode_table+0xac>
 c30:	sub	w22, w22, #0x1
 c34:	cmp	w22, #0x1
 c38:	b.hi	c28 <_build_decode_table+0x94>  // b.pmore
 c3c:	mov	w22, #0x1                   	// #1
 c40:	cbz	x26, c50 <_build_decode_table+0xbc>
 c44:	cmp	w22, w20
 c48:	csel	w20, w22, w20, cc	// cc = lo, ul, last
 c4c:	str	w20, [x26]
 c50:	ldr	w9, [sp, #72]
 c54:	stp	wzr, w9, [sp, #8]
 c58:	cmp	w22, #0x2
 c5c:	b.cc	c9c <_build_decode_table+0x108>  // b.lo, b.ul, b.last
 c60:	mov	w10, #0x0                   	// #0
 c64:	mov	w8, w22
 c68:	add	x11, sp, #0x48
 c6c:	add	x11, x11, #0x4
 c70:	sub	x12, x8, #0x1
 c74:	add	x13, sp, #0x8
 c78:	add	x13, x13, #0x8
 c7c:	ldr	w14, [x11], #4
 c80:	add	w9, w14, w9
 c84:	str	w9, [x13], #4
 c88:	add	w10, w14, w10, lsl #1
 c8c:	subs	x12, x12, #0x1
 c90:	b.ne	c7c <_build_decode_table+0xe8>  // b.any
 c94:	lsl	w9, w10, #1
 c98:	b	ca4 <_build_decode_table+0x110>
 c9c:	mov	w9, #0x0                   	// #0
 ca0:	mov	w8, #0x1                   	// #1
 ca4:	add	x10, sp, #0x48
 ca8:	ldr	w8, [x10, x8, lsl #2]
 cac:	add	w8, w8, w9
 cb0:	cbz	w25, ce0 <_build_decode_table+0x14c>
 cb4:	mov	x9, #0x0                   	// #0
 cb8:	mov	w10, w25
 cbc:	add	x11, sp, #0x8
 cc0:	ldrb	w12, [x24, x9]
 cc4:	ldr	w13, [x11, x12, lsl #2]
 cc8:	add	w14, w13, #0x1
 ccc:	str	w14, [x11, x12, lsl #2]
 cd0:	strh	w9, [x23, x13, lsl #1]
 cd4:	add	x9, x9, #0x1
 cd8:	cmp	x10, x9
 cdc:	b.ne	cc0 <_build_decode_table+0x12c>  // b.any
 ce0:	mov	w9, #0x1                   	// #1
 ce4:	lsl	w9, w9, w22
 ce8:	cmp	w8, w9
 cec:	b.hi	f1c <_build_decode_table+0x388>  // b.pmore
 cf0:	ldr	w9, [sp, #8]
 cf4:	add	x23, x23, x9, lsl #1
 cf8:	b.cc	efc <_build_decode_table+0x368>  // b.lo, b.ul, b.last
 cfc:	mov	w22, #0x0                   	// #0
 d00:	add	x9, sp, #0x48
 d04:	add	w22, w22, #0x1
 d08:	ldr	w8, [x9, w22, uxtw #2]
 d0c:	cbz	w8, d04 <_build_decode_table+0x170>
 d10:	cmp	w22, w20
 d14:	b.ls	e28 <_build_decode_table+0x294>  // b.plast
 d18:	mov	w24, #0x0                   	// #0
 d1c:	mov	w14, #0x0                   	// #0
 d20:	mov	w9, #0x1                   	// #1
 d24:	lsl	w2, w9, w20
 d28:	sub	w10, w2, #0x1
 d2c:	mov	w11, #0xffffffff            	// #-1
 d30:	mov	w12, #0x80000000            	// #-2147483648
 d34:	add	x13, sp, #0x48
 d38:	mov	w3, #0xffffffff            	// #-1
 d3c:	sub	w15, w22, w20
 d40:	lsl	w16, w9, w15
 d44:	add	w17, w15, w15, lsl #8
 d48:	lsl	w0, w11, w22
 d4c:	mvn	w0, w0
 d50:	add	w1, w22, #0x1
 d54:	mov	x4, x2
 d58:	and	w5, w24, w10
 d5c:	cmp	w5, w3
 d60:	b.ne	d6c <_build_decode_table+0x1d8>  // b.any
 d64:	mov	x2, x4
 d68:	b	dc4 <_build_decode_table+0x230>
 d6c:	mov	x14, x15
 d70:	mov	x3, x16
 d74:	cmp	w8, w16
 d78:	b.cs	da4 <_build_decode_table+0x210>  // b.hs, b.nlast
 d7c:	mov	x2, x1
 d80:	mov	x14, x15
 d84:	mov	x6, x8
 d88:	add	w14, w14, #0x1
 d8c:	ldr	w3, [x13, w2, uxtw #2]
 d90:	add	w6, w3, w6, lsl #1
 d94:	lsl	w3, w9, w14
 d98:	add	w2, w2, #0x1
 d9c:	cmp	w6, w3
 da0:	b.cc	d88 <_build_decode_table+0x1f4>  // b.lo, b.ul, b.last
 da4:	add	w2, w3, w4
 da8:	lsl	w3, w4, #16
 dac:	orr	w14, w3, w14, lsl #8
 db0:	orr	w14, w14, w20
 db4:	orr	w14, w14, #0xc000
 db8:	str	w14, [x19, w5, uxtw #2]
 dbc:	mov	x3, x5
 dc0:	mov	x14, x4
 dc4:	ldrh	w4, [x23]
 dc8:	ldr	w4, [x21, x4, lsl #2]
 dcc:	add	w4, w17, w4
 dd0:	lsr	w5, w24, w20
 dd4:	add	w5, w14, w5
 dd8:	str	w4, [x19, w5, uxtw #2]
 ddc:	add	w5, w5, w16
 de0:	cmp	w5, w2
 de4:	b.cc	dd8 <_build_decode_table+0x244>  // b.lo, b.ul, b.last
 de8:	cmp	w24, w0
 dec:	b.eq	ec0 <_build_decode_table+0x32c>  // b.none
 df0:	add	x23, x23, #0x2
 df4:	eor	w4, w24, w0
 df8:	clz	w4, w4
 dfc:	lsr	w4, w12, w4
 e00:	sub	w5, w4, #0x1
 e04:	and	w5, w5, w24
 e08:	orr	w24, w5, w4
 e0c:	mov	x4, x2
 e10:	subs	w8, w8, #0x1
 e14:	b.ne	d58 <_build_decode_table+0x1c4>  // b.any
 e18:	add	w22, w22, #0x1
 e1c:	ldr	w8, [x13, w22, uxtw #2]
 e20:	cbz	w8, e18 <_build_decode_table+0x284>
 e24:	b	d3c <_build_decode_table+0x1a8>
 e28:	mov	w24, #0x0                   	// #0
 e2c:	mov	w9, #0x1                   	// #1
 e30:	lsl	w25, w9, w22
 e34:	mov	w26, #0x80000000            	// #-2147483648
 e38:	add	x27, sp, #0x48
 e3c:	add	w9, w22, w22, lsl #8
 e40:	sub	w10, w25, #0x1
 e44:	ldrh	w11, [x23]
 e48:	ldr	w11, [x21, x11, lsl #2]
 e4c:	add	w11, w9, w11
 e50:	str	w11, [x19, w24, uxtw #2]
 e54:	cmp	w24, w10
 e58:	b.eq	ea8 <_build_decode_table+0x314>  // b.none
 e5c:	add	x23, x23, #0x2
 e60:	eor	w11, w10, w24
 e64:	clz	w11, w11
 e68:	lsr	w11, w26, w11
 e6c:	sub	w12, w11, #0x1
 e70:	and	w12, w12, w24
 e74:	orr	w24, w12, w11
 e78:	subs	w8, w8, #0x1
 e7c:	b.ne	e44 <_build_decode_table+0x2b0>  // b.any
 e80:	add	w22, w22, #0x1
 e84:	cmp	w22, w20
 e88:	b.hi	e94 <_build_decode_table+0x300>  // b.pmore
 e8c:	bl	f5c <_OUTLINED_FUNCTION_0>
 e90:	lsl	w25, w25, #1
 e94:	ldr	w8, [x27, w22, uxtw #2]
 e98:	cbz	w8, e80 <_build_decode_table+0x2ec>
 e9c:	cmp	w22, w20
 ea0:	b.ls	e3c <_build_decode_table+0x2a8>  // b.plast
 ea4:	b	d1c <_build_decode_table+0x188>
 ea8:	subs	w20, w20, w22
 eac:	b.ls	ec0 <_build_decode_table+0x32c>  // b.plast
 eb0:	bl	f5c <_OUTLINED_FUNCTION_0>
 eb4:	lsl	w25, w25, #1
 eb8:	subs	w20, w20, #0x1
 ebc:	b.ne	eb0 <_build_decode_table+0x31c>  // b.any
 ec0:	mov	w0, #0x1                   	// #1
 ec4:	ldur	x8, [x29, #-88]
 ec8:	adrp	x9, 0 <___stack_chk_guard>
 ecc:	ldr	x9, [x9]
 ed0:	ldr	x9, [x9]
 ed4:	cmp	x9, x8
 ed8:	b.ne	f58 <_build_decode_table+0x3c4>  // b.any
 edc:	ldp	x29, x30, [sp, #224]
 ee0:	ldp	x20, x19, [sp, #208]
 ee4:	ldp	x22, x21, [sp, #192]
 ee8:	ldp	x24, x23, [sp, #176]
 eec:	ldp	x26, x25, [sp, #160]
 ef0:	ldp	x28, x27, [sp, #144]
 ef4:	add	sp, sp, #0xf0
 ef8:	ret
 efc:	cbz	w8, f24 <_build_decode_table+0x390>
 f00:	sub	w9, w22, #0x1
 f04:	mov	w10, #0x1                   	// #1
 f08:	lsl	w9, w10, w9
 f0c:	ldr	w10, [sp, #76]
 f10:	cmp	w8, w9
 f14:	ccmp	w10, #0x1, #0x0, eq	// eq = none
 f18:	b.eq	f2c <_build_decode_table+0x398>  // b.none
 f1c:	mov	w0, #0x0                   	// #0
 f20:	b	ec4 <_build_decode_table+0x330>
 f24:	mov	x8, #0x0                   	// #0
 f28:	b	f30 <_build_decode_table+0x39c>
 f2c:	ldrh	w8, [x23]
 f30:	ldr	w8, [x21, x8, lsl #2]
 f34:	add	w8, w8, #0x101
 f38:	mov	w9, #0x1                   	// #1
 f3c:	mov	w0, #0x1                   	// #1
 f40:	sub	w10, w9, #0x1
 f44:	str	w8, [x19, w10, uxtw #2]
 f48:	lsr	w10, w9, w20
 f4c:	add	w9, w9, #0x1
 f50:	cbz	w10, f40 <_build_decode_table+0x3ac>
 f54:	b	ec4 <_build_decode_table+0x330>
 f58:	bl	0 <___stack_chk_fail>

0000000000000f5c <_OUTLINED_FUNCTION_0>:
 f5c:	ubfiz	x2, x25, #2, #32
 f60:	add	x0, x19, w25, uxtw #2
 f64:	mov	x1, x19
 f68:	b	0 <_memcpy>
