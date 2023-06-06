__kernel void crop(__global unsigned char *src, __global unsigned char *dst,
                   int width, int height, int crop_width, int crop_height,
                   int x_offset, int y_offset) {
  int tid = get_global_id(0);

  int dst_row = tid / crop_width;
  int dst_col = tid % crop_width;

  int src_row = dst_row + y_offset;
  int src_col = dst_col + x_offset;

  if (dst_row < crop_height && dst_col < crop_width &&
      src_row < height && src_col < width) {
    int src_index = (src_row * width + src_col) * 3;
    int dst_index = (dst_row * crop_width + dst_col) * 3;

    dst[dst_index] = src[src_index];
    dst[dst_index + 1] = src[src_index + 1];
    dst[dst_index + 2] = src[src_index + 2];
  }
}