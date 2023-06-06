#define X_SIZE 514
#define PI 3.1415f

struct Point3D {
  float x, y, z;
};

struct Point3D rotatePoint(int x, int y, float angle) {
  struct Point3D point;
  point.x = (2 * (x - X_SIZE)) * cos(angle * PI / 180.0f);
  point.y = (2 * (x - X_SIZE)) * sin(angle * PI / 180.0f);
  point.z = y;
  return point;
}

__kernel void contour(__global unsigned char *src, __global float *dst,
                      int width, int height, float angle) {
  const int tid = get_global_id(0);
  
  // unsigned int src_base = width * tid;
  unsigned int src_base = width * tid;
  int processed = 0;
  while (processed < width) {

    uint4 pack16 =  as_uint4(vload16(0, src + src_base + processed));
    if (pack16.s0 > 0 || pack16.s1 > 0 || pack16.s2 > 0 || pack16.s3 > 0) {
      // break;
      uchar4 pack4;
      if (pack16.s0) {
        pack4 = as_uchar4(pack16.s0);
      } else if (pack16.s1) {
        pack4 = as_uchar4(pack16.s1);
        processed += 4;
      } else if (pack16.s2) {
        pack4 = as_uchar4(pack16.s2);
        processed += 8;
      } else if (pack16.s3) {
        pack4 = as_uchar4(pack16.s3);
        processed += 12;
      }

      if (pack4.s0) {
        break;
      } else if (pack4.s1) {
        processed += 1;
        break;
      } else if (pack4.s2) {
        processed += 2;
        break;
      } else if (pack4.s3) {
        processed += 3;
        break;
      }

    } else {
      processed += 16;
    }
  }

  __global float *dst_ptr = dst + 3 * tid;

  if (processed < width) {
    struct Point3D p = rotatePoint(processed, tid, angle);
    dst_ptr[0] = p.x;
    dst_ptr[1] = p.y;
    dst_ptr[2] = p.z;

  } else {
    dst_ptr[0] = 0.0f;
    dst_ptr[1] = 0.0f;
    dst_ptr[2] = 0.0f;
  }
}
