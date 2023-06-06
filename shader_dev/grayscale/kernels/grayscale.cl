#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void grayscale(__global unsigned char *src,
                        __global unsigned char *dst, int width, int height) {
  unsigned int tid = get_global_id(0);
  int offset = 30 * tid;

  half8 coef_r = (half8)0.299h;
  half8 coef_g = (half8)0.587h;
  half8 coef_b = (half8)0.114h;

  uchar16 pixels0 = vload16(0, src + offset);
  uchar16 pixels1 = vload16(0, src + offset + 16);

  half8 r0, g0, b0;

  r0.s0 = (half)pixels0.s0;
  r0.s1 = (half)pixels0.s3;
  r0.s2 = (half)pixels0.s6;
  r0.s3 = (half)pixels0.s9;
  r0.s4 = (half)pixels0.sc;

  g0.s0 = (half)pixels0.s1;
  g0.s1 = (half)pixels0.s4;
  g0.s2 = (half)pixels0.s7;
  g0.s3 = (half)pixels0.sa;
  g0.s4 = (half)pixels0.sd;

  b0.s0 = (half)pixels0.s2;
  b0.s1 = (half)pixels0.s5;
  b0.s2 = (half)pixels0.s8;
  b0.s3 = (half)pixels0.sb;
  b0.s4 = (half)pixels0.se;

  r0 = r0 * coef_r;
  g0 = g0 * coef_g;
  b0 = b0 * coef_b;
  half8 sum0 = r0 + g0 + b0;
  sum0 = select((half8) 255.0h, (half8) 0.0h, (short8) islessequal(sum0, (half8) 200.0h));
  uchar4 res0;
  res0.s0 = (uchar)sum0.s0;
  res0.s1 = (uchar)sum0.s1;
  res0.s2 = (uchar)sum0.s2;
  res0.s3 = (uchar)sum0.s3;

  half8 r1, g1, b1;

  r1.s0 = (half)pixels1.s0;
  r1.s1 = (half)pixels1.s3;
  r1.s2 = (half)pixels1.s6;
  r1.s3 = (half)pixels1.s9;
  r1.s4 = (half)pixels1.sc;

  g1.s0 = (half)pixels1.s1;
  g1.s1 = (half)pixels1.s4;
  g1.s2 = (half)pixels1.s7;
  g1.s3 = (half)pixels1.sa;
  g1.s4 = (half)pixels1.sd;

  b1.s0 = (half)pixels1.s2;
  b1.s1 = (half)pixels1.s5;
  b1.s2 = (half)pixels1.s8;
  b1.s3 = (half)pixels1.sb;
  b1.s4 = (half)pixels1.se;

  r1 = r1 * coef_r;
  g1 = g1 * coef_g;
  b1 = b1 * coef_b;
  half8 sum1 = r1 + g1 + b1;
  sum1 = select((half8) 255.0h, (half8) 0.0h, (short8) islessequal(sum1, (half8) 200.0h));
  

  uchar4 res1;
  res1.s0 = (uchar)sum1.s0;
  res1.s1 = (uchar)sum1.s1;
  res1.s2 = (uchar)sum1.s2;
  res1.s3 = (uchar)sum1.s3;
  vstore4(res0, 0, dst + 10 * tid);
  dst[10 * tid + 4] = (uchar)sum0.s4;

  vstore4(res1, 0, dst + 10 * tid + 5);
  dst[10 * tid + 9] = (uchar)sum1.s4;
}