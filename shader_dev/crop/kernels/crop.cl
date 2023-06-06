
#define PIXELS_PER_THREAD 15
#define CHANNELS 3
__kernel void crop(__global unsigned char *src, __global unsigned char *dst,
                   int width, int height, int crop_width, int crop_height,
                   int x_offset, int y_offset) {
  const int tid = get_global_id(0);
  const int buckets_per_row =
      (crop_width + PIXELS_PER_THREAD - 1) / PIXELS_PER_THREAD;

  int row_id = tid / buckets_per_row;
  int bucket = tid % buckets_per_row;

  unsigned int src_base =
      CHANNELS * (x_offset + row_id * width + PIXELS_PER_THREAD * bucket);
  unsigned int dst_base =
      CHANNELS * (row_id * crop_width + PIXELS_PER_THREAD * bucket);

  uchar16 pixels0 = vload16(0, src + src_base);
  uchar16 pixels1 = vload16(0, src + src_base + 15);
  uchar16 pixels2 = vload16(0, src + src_base + 30);

  vstore16(pixels0, 0, dst + dst_base);
  vstore16(pixels1, 0, dst + dst_base + 15);
  vstore16(pixels2, 0, dst + dst_base + 30);
}