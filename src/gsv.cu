#include <cuda.h>
#include <gmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "../include/cgbn/cgbn.h"
#include "support.h"

// The CGBN context uses the following three parameters:
//   TBP             - threads per block (zero means to use the blockDim.x)
//   MAX_ROTATION    - must be small power of 2, imperically, 4 works well
//   SHM_LIMIT       - number of bytes of dynamic shared memory available to the kernel
//   CONSTANT_TIME   - require constant time algorithms (currently, constant time algorithms are not available)

// Locally it will also be helpful to have several parameters:
//   TPI             - threads per instance
//   BITS            - number of bits per instance

template <uint32_t tpi, uint32_t bits>
class gsv_params_t {
 public:
  // parameters used by the CGBN context
  static const uint32_t TPB = 0;            // get TPB from blockDim.x
  static const uint32_t MAX_ROTATION = 4;   // good default value
  static const uint32_t SHM_LIMIT = 0;      // no shared mem available
  static const bool CONSTANT_TIME = false;  // constant time implementations aren't available yet

  // parameters used locally in the application
  static const uint32_t TPI = tpi;    // threads per instance
  static const uint32_t BITS = bits;  // instance size
};

template <class params>
class gsv_t {
 public:
  // instance_t should be 128-byte aligned
  typedef struct {
    cgbn_mem_t<params::BITS> r;      // sig->r
    cgbn_mem_t<params::BITS> s;      // sig->s
    cgbn_mem_t<params::BITS> e;      // digest
    cgbn_mem_t<params::BITS> key_x;  // public key
    cgbn_mem_t<params::BITS> key_y;  // public key
  } instance_t;

  typedef struct {
    cgbn_mem_t<params::BITS> order;  // group order
    cgbn_mem_t<params::BITS> g_x;    // base point (generator)
    cgbn_mem_t<params::BITS> g_y;    // base point (generator)
    cgbn_mem_t<params::BITS> field;  // prime p
    cgbn_mem_t<params::BITS> g_a;
  } ec_t;

  typedef cgbn_context_t<params::TPI> context_t;
  typedef cgbn_env_t<context_t, params::BITS> env_t;
  typedef typename env_t::cgbn_t bn_t;
  typedef typename env_t::cgbn_local_t bn_local_t;
  typedef typename env_t::cgbn_wide_t bn_wide_t;

  context_t _context;
  env_t _env;
  int32_t _instance;

  __device__ __forceinline__ gsv_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance)
      : _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {}

  __device__ __forceinline__ void mod(bn_t &r, const bn_t &m) {
    while (_env.compare(r, m) >= 0) {
      _env.sub(r, r, m);
    }
  }

  // both a and b should be non-negative and less than m
  __device__ __forceinline__ void mod_add(bn_t &r, const bn_t &a, const bn_t &b, const bn_t &m) {
    _env.add(r, a, b);
    if (_env.compare(r, m) >= 0) {
      _env.sub(r, r, m);
    }
  }

  __device__ __forceinline__ void mod_sub(bn_t &r, const bn_t &a, const bn_t &b, const bn_t &m) {
    if (_env.sub(r, a, b)) {  // a < b
      _env.add(r, r, m);
    }
  }

  __device__ __forceinline__ void mod_lshift1(bn_t &r, const bn_t &a, const bn_t &m) {
    _env.shift_left(r, a, 1);
    if (_env.compare(r, m) >= 0) {
      _env.sub(r, r, m);
    }
  }

  // TODO: optimize
  __device__ __forceinline__ void mod_lshift(bn_t &r, const bn_t &a, const bn_t &m, uint32_t n) {
    for (int i = 0; i < n; i++) {
      mod_lshift1(r, a, m);
    }
  }

  // TODO: buggy, do not use; needs debugging
  __device__ __forceinline__ void point_dbl(bn_t &r_x, bn_t &r_y, bn_t &r_z, const bn_t &a_x, const bn_t &a_y, const bn_t &a_z,
                                            const bn_t &field, const bn_t &g_a, const uint32_t np0) {
    if (_env.equals_ui32(a_z, 0)) {
      _env.set_ui32(r_z, 0);
      return;
    }

    bn_t n0, n1, n2, n3;

    _env.mont_sqr(n0, a_x, field, np0);      // n0 = a_x^2
    mod_lshift1(n1, n0, field);              // n1 = 2 * a_x^2
    mod_add(n0, n0, n1, field);              // n0 = 3 * a_x^2
    _env.mont_sqr(n1, a_z, field, np0);      // n1 = a_z^2
    _env.mont_sqr(n1, n1, field, np0);       // n1 = a_z^4
    _env.mont_mul(n1, n1, g_a, field, np0);  // n1 = g_a * a_z^4
    mod_add(n1, n1, n0, field);              // n1 = 3 * a_x^2 + g_a * a_z^4

    _env.mont_mul(n0, a_y, a_z, field, np0);  // n0 = a_y * a_z
    mod_lshift1(r_z, n0, field);              // r_z = 2 * a_y * a_z

    _env.mont_sqr(n3, a_y, field, np0);      // n3 = a_y^2
    _env.mont_mul(n2, a_x, n3, field, np0);  // n2 = a_x * a_y^2
    mod_lshift(n2, n2, field, 2);            // n2 = 4 * a_x * a_y^2

    mod_lshift1(n0, n2, field);          // n0 = 2 * n2
    _env.mont_sqr(r_x, n1, field, np0);  // r_x = n1^2
    mod_sub(r_x, r_x, n0, field);        // r_x = n1^2 - 2 * n2

    _env.mont_sqr(n0, n3, field, np0);  // n0 = a_y^4
    mod_lshift(n3, n0, field, 3);       // n3 = 8 * a_y^4

    mod_sub(n0, n2, r_x, field);            // n0 = n2 - r_x
    _env.mont_mul(n0, n1, n0, field, np0);  // n0 = n1 * (n2 - r_x)
    mod_sub(r_y, n0, n3, field);            // r_y = n1 * (n2 - r_x) - n3
  }

  __device__ __forceinline__ void point_dbl_ipp(bn_t &r_x, bn_t &r_y, bn_t &r_z, const bn_t &a_x, const bn_t &a_y,
                                                const bn_t &a_z, const bn_t &field, const bn_t &g_a, const uint32_t np0) {
    if (_env.equals_ui32(a_z, 0)) {
      _env.set_ui32(r_z, 0);
      return;
    }

    bn_t u, m, s, t;

    mod_lshift1(s, a_y, field);         // s = 2 * a_y
    _env.mont_sqr(u, a_z, field, np0);  // u = a_z^2

    _env.mont_sqr(m, s, field, np0);         // m = 4 * a_y^2
    _env.mont_mul(r_z, s, a_z, field, np0);  // r_z = 2 * a_y * a_z

    _env.mont_sqr(r_y, m, field, np0);  // r_y = 16 * a_y^4

    _env.mont_mul(s, m, a_x, field, np0);  // s = 4 * a_x * a_y^2
    if (_env.ctz(r_y) == 0) {
      _env.add(r_y, r_y, field);
    }
    _env.shift_right(r_y, r_y, 1);  // r_y = 8 * a_y^4

    _env.mont_sqr(m, a_x, field, np0);  // m = a_x ^ 2
    mod_lshift1(t, m, field);           // t = 2 * a_x^2
    mod_add(m, m, t, field);            // m = 3 * a_x^2

    _env.mont_sqr(u, u, field, np0);       // u = a_z^4
    _env.mont_mul(u, u, g_a, field, np0);  // u = g_a * a_z^4
    mod_add(m, m, u, field);               // m = 3 * a_x^2 + g_a * a_z^4

    mod_lshift1(u, s, field);           // u = 8 * a_x * a_y^2
    _env.mont_sqr(r_x, m, field, np0);  // r_x = m^2
    mod_sub(r_x, r_x, u, field);        // r_x = m^2 - u

    mod_sub(s, s, r_x, field);           // s = 4 * a_x * a_y^2 - r_x
    _env.mont_mul(s, s, m, field, np0);  // s = (4 * a_x * a_y^2 - r_x) * m
    mod_sub(r_y, s, r_y, field);         // r_y = (4 * a_x * a_y^2 - r_x) * m - 8 * a_y^4
  }

  __device__ __forceinline__ void point_add(bn_t &r_x, bn_t &r_y, bn_t &r_z, const bn_t &a_x, const bn_t &a_y, const bn_t &a_z,
                                            const bn_t &b_x, const bn_t &b_y, const bn_t &b_z, const bn_t &field,
                                            const bn_t &g_a, const uint32_t np0) {
    if (_env.compare(a_x, b_x) == 0 && _env.compare(a_y, b_y) == 0 && _env.compare(a_z, b_z) == 0) {
      // if (threadIdx.x == 0) printf("DOUBLE\n");
      point_dbl(r_x, r_y, r_z, a_x, a_y, a_z, field, g_a, np0);
      return;
    }
    if (_env.equals_ui32(a_z, 0)) {
      _env.set(r_x, b_x);
      _env.set(r_y, b_y);
      _env.set(r_z, b_z);
      return;
    }
    if (_env.equals_ui32(b_z, 0)) {
      _env.set(r_x, a_x);
      _env.set(r_y, a_y);
      _env.set(r_z, a_z);
      return;
    }

    bn_t n0, n1, n2, n3, n4, n5, n6;

    _env.mont_sqr(n0, b_z, field, np0);      // n0 = b_z^2
    _env.mont_mul(n1, a_x, n0, field, np0);  // n1 = a_x * b_z^2

    _env.mont_mul(n0, n0, b_z, field, np0);  // n0 = b_z^3
    _env.mont_mul(n2, a_y, n0, field, np0);  // n2 = a_y * b_z^3

    _env.mont_sqr(n0, a_z, field, np0);      // n0 = a_z^2
    _env.mont_mul(n3, b_x, n0, field, np0);  // n3 = b_x * a_z^2

    _env.mont_mul(n0, n0, a_z, field, np0);  // n0 = a_z^3
    _env.mont_mul(n4, b_y, n0, field, np0);  // n4 = b_y * a_z^3

    mod_sub(n5, n1, n3, field);  // n5 = n1 - n3
    mod_sub(n6, n2, n4, field);  // n6 = n2 - n4

    if (_env.equals_ui32(n5, 0)) {
      if (_env.equals_ui32(n6, 0)) {
#ifdef DEBUG
        if (threadIdx.x == 0) printf("EQUAL\n");
#endif
        point_dbl(r_x, r_y, r_z, a_x, a_y, a_z, field, g_a, np0);
        return;
      } else {
        _env.set_ui32(r_z, 0);
        return;
      }
    }

    mod_add(n1, n1, n3, field);  // 'n7' = n1 + n3
    mod_add(n2, n2, n4, field);  // 'n8' = n2 + n4

    _env.mont_mul(n0, a_z, b_z, field, np0);  // n0 = a_z * b_z
    _env.mont_mul(r_z, n0, n5, field, np0);   // r_z = a_z * b_z * n5

    _env.mont_sqr(n0, n6, field, np0);      // n0 = n6^2
    _env.mont_sqr(n4, n5, field, np0);      // n4 = n5^2
    _env.mont_mul(n3, n1, n4, field, np0);  // n3 = n5^2 * 'n7'
    mod_sub(r_x, n0, n3, field);            // r_x = n6^2 - n5^2 * 'n7'

    mod_lshift1(n0, r_x, field);  // n0 = 2 * r_x
    mod_sub(n0, n3, n0, field);   // 'n9' = n5^2 * 'n7' - 2 * r_x

    _env.mont_mul(n0, n0, n6, field, np0);  // n0 = n6 * 'n9'
    _env.mont_mul(n5, n4, n5, field, np0);  // 'n5' = n5^3
    _env.mont_mul(n1, n2, n5, field, np0);  // n1 = 'n8' * n5^3
    mod_sub(n0, n0, n1, field);             // n0 = n6 * 'n9' - 'n8' * n5^3
    if (_env.ctz(n0) == 0) {                // if n0 is odd
      _env.add(n0, n0, field);              // 0 <= n0 < 2 * field, n0 is even
    }
    _env.shift_right(r_y, n0, 1);  // r_y = (n6 * 'n9' - 'n8' * n5^3) / 2
  }

  __device__ __forceinline__ void point_add_ipp(bn_t &r_x, bn_t &r_y, bn_t &r_z, const bn_t &a_x, const bn_t &a_y,
                                                const bn_t &a_z, const bn_t &b_x, const bn_t &b_y, const bn_t &b_z,
                                                const bn_t &field, const bn_t &g_a, const uint32_t np0) {
    if (_env.compare(a_x, b_x) == 0 && _env.compare(a_y, b_y) == 0 && _env.compare(a_z, b_z) == 0) {
      // if (threadIdx.x == 0) printf("DOUBLE\n");
      point_dbl(r_x, r_y, r_z, a_x, a_y, a_z, field, g_a, np0);
      return;
    }
    if (_env.equals_ui32(a_z, 0)) {
      _env.set(r_x, b_x);
      _env.set(r_y, b_y);
      _env.set(r_z, b_z);
      return;
    }
    if (_env.equals_ui32(b_z, 0)) {
      _env.set(r_x, a_x);
      _env.set(r_y, a_y);
      _env.set(r_z, a_z);
      return;
    }

    bn_t u1, u2, s1, s2, h, r;

    _env.mont_mul(s1, a_y, b_z, field, np0);  // s1 = a_y * b_z
    _env.mont_sqr(u1, b_z, field, np0);       // u1 = b_z^2

    _env.mont_mul(s2, b_y, a_z, field, np0);  // s2 = b_y * a_z
    _env.mont_sqr(u2, a_z, field, np0);       // u2 = a_z^2

    _env.mont_mul(s1, s1, u1, field, np0);  // s1 = a_y * b_z^3
    _env.mont_mul(s2, s2, u2, field, np0);  // s2 = b_y * a_z^3

    _env.mont_mul(u1, a_x, u1, field, np0);  // u1 = a_x * b_z^2
    _env.mont_mul(u2, b_x, u2, field, np0);  // u2 = b_x * a_z^2

    mod_sub(r, s2, s1, field);  // r = s2 - s1
    mod_sub(h, u2, u1, field);  // h = u2 - u1

    if (_env.equals_ui32(h, 0)) {
      if (_env.equals_ui32(r, 0)) {
        // if (threadIdx.x == 0) printf("EQUAL\n");
        point_dbl(r_x, r_y, r_z, a_x, a_y, a_z, field, g_a, np0);
        return;
      } else {
        _env.set_ui32(r_z, 0);
        return;
      }
    }

    _env.mont_mul(r_z, a_z, b_z, field, np0);  // r_z = a_z * b_z
    _env.mont_sqr(u2, h, field, np0);          // u2 = h^2
    _env.mont_mul(r_z, r_z, h, field, np0);    // r_z = a_z * b_z * h
    _env.mont_sqr(s2, r, field, np0);          // s2 = r^2
    _env.mont_mul(h, h, u2, field, np0);       // h = h^3

    _env.mont_mul(u1, u1, u2, field, np0);  // u1 = u1 * h^2
    mod_sub(r_x, s2, h, field);             // r_x = r^2 - h^3
    mod_lshift1(u2, u1, field);             // u2 = 2 * u1 * h^2
    _env.mont_mul(s1, s1, h, field, np0);   // s1 = s1 * h^3
    mod_sub(r_x, r_x, u2, field);           // r_x = r^2 - h^3 - 2 * u1 * h^2

    mod_sub(r_y, u1, r_x, field);            // r_y = u1 * h^2 - r_x
    _env.mont_mul(r_y, r_y, r, field, np0);  // r_y = r * (u1 * h^2 - r_x)
    mod_sub(r_y, r_y, s1, field);            // r_y = r * (u1 * h^2 - r_x) - s1 * h^3
  }

  // double-and-add, index increasing
  __device__ __forceinline__ void point_mult(bn_t &r_x, bn_t &r_y, bn_t &r_z, const bn_t &p_x, const bn_t &p_y, const bn_t &p_z,
                                             const bn_t &d, const bn_t &field, const bn_t &g_a, const uint32_t np0) {
    bn_t q_x, q_y, q_z;
    uint32_t limb;
    __shared__ cgbn_mem_t<params::BITS> s_d;

    _env.store(&s_d, d);
    _env.set(q_x, p_x);
    _env.set(q_y, p_y);
    _env.set(q_z, p_z);
    _env.set_ui32(r_z, 0);

    for (int i = 0; i < (params::BITS + 31) / 32; i++) {
      limb = s_d._limbs[i];
      if (limb == 0) {
        break;
      }
      uint32_t mask = limb;
      for (int j = 0; j < 32; j++) {
        if (mask & 1) {
          // if (threadIdx.x == 0) printf("%d\t%d:\t%d\t%d\t%u\t%u\n", blockIdx.x, threadIdx.x, i, j, limb, mask);
          point_add_ipp(r_x, r_y, r_z, r_x, r_y, r_z, q_x, q_y, q_z, field, g_a, np0);
        }
        point_dbl_ipp(q_x, q_y, q_z, q_x, q_y, q_z, field, g_a, np0);
        mask >>= 1;
      }
    }
  }

  // double-and-add, index decreasing
  __device__ __forceinline__ void point_mult_alt(bn_t &r_x, bn_t &r_y, bn_t &r_z, const bn_t &p_x, const bn_t &p_y,
                                                 const bn_t &p_z, const bn_t &d, const bn_t &field, const bn_t &g_a,
                                                 const uint32_t np0) {
    bn_t q_x, q_y, q_z;
    uint32_t limb;
    __shared__ cgbn_mem_t<params::BITS> s_d;

    _env.store(&s_d, d);
    _env.set_ui32(r_z, 0);
    _env.set(q_x, p_x);
    _env.set(q_y, p_y);
    _env.set(q_z, p_z);

    int flag = 0;
    for (int i = (params::BITS + 31) / 32 - 1; i >= 0; i--) {
      limb = s_d._limbs[i];
      if (limb == 0) continue;
      uint32_t mask = 0x80000000L;
      for (int j = 0; j < 32; j++) {
        if ((!flag) && (limb & mask)) {
          flag = 1;
        }
        if (flag) point_dbl(r_x, r_y, r_z, r_x, r_y, r_z, field, g_a, np0);
        if (limb & mask) {
          // if (threadIdx.x == 0) printf("%d\t%d:\t%d\t%d\t%u\t%u\n", blockIdx.x, threadIdx.x, i, j, limb, mask);
          point_add_ipp(r_x, r_y, r_z, r_x, r_y, r_z, q_x, q_y, q_z, field, g_a, np0);
        }
        mask >>= 1;
      }
    }
  }

  // transform (X, Y, Z) into (x, y) := (X/Z^2, Y/Z^3)
  __device__ __forceinline__ void conv_affine_x_y(bn_t &a_x, bn_t &a_y, const bn_t &j_x, const bn_t &j_y, const bn_t &j_z,
                                                  const bn_t &field, const uint32_t np0) {
    if (_env.equals_ui32(j_z, 0)) {
      _env.set_ui32(a_x, 1);
      _env.set_ui32(a_y, 1);
      return;
    }

    bn_t Z_, Z_1, Z_2, Z_3;

    _env.mont2bn(Z_, j_z, field, np0);

    if (_env.equals_ui32(Z_, 1)) {
      _env.mont2bn(a_x, j_x, field, np0);
      _env.mont2bn(a_y, j_y, field, np0);
    } else {
      _env.modular_inverse(Z_1, Z_, field);
      _env.bn2mont(Z_1, Z_1, field);
      _env.mont_sqr(Z_2, Z_1, field, np0);
      _env.mont_mul(a_x, j_x, Z_2, field, np0);
      _env.mont2bn(a_x, a_x, field, np0);
    }
  }

  __device__ __forceinline__ void conv_affine_x(bn_t &a_x, const bn_t &j_x, const bn_t &j_z, const bn_t &field,
                                                const uint32_t np0) {
    if (_env.equals_ui32(j_z, 0)) {
      _env.set_ui32(a_x, 1);
      return;
    }

    bn_t Z_, Z_1, Z_2;

    _env.mont2bn(Z_, j_z, field, np0);

    if (_env.equals_ui32(Z_, 1)) {
      _env.mont2bn(a_x, j_x, field, np0);
    } else {
      _env.modular_inverse(Z_1, Z_, field);
      _env.bn2mont(Z_1, Z_1, field);
      _env.mont_sqr(Z_2, Z_1, field, np0);
      _env.mont_mul(a_x, j_x, Z_2, field, np0);
      _env.mont2bn(a_x, a_x, field, np0);
    }

    // _env.modular_inverse(Z_1, j_z, field);
  }

#ifdef DEBUG
  __device__ __forceinline__ int32_t debug_kernel(const bn_t &r, const bn_t &s, const bn_t &e, const bn_t &key_x,
                                                  const bn_t &key_y, const bn_t &order, const bn_t &g_x, const bn_t &g_y,
                                                  const bn_t &field, bn_t &g_a, bn_t &tmp) {
    bn_t x1, y1, z1, x2, y2, one, zero;
    uint32_t np0;

    _env.set_ui32(zero, 0);
    _env.set_ui32(one, 1);
    np0 = _env.bn2mont(one, one, field);
    mod(g_a, field);
    _env.bn2mont(g_a, g_a, field);

    _env.set(x1, g_x);
    _env.set(y1, g_y);
    mod(x1, field);
    _env.bn2mont(x1, x1, field);
    mod(y1, field);
    _env.bn2mont(y1, y1, field);

    _env.set(x2, key_x);
    _env.set(y2, key_y);
    mod(x2, field);
    _env.bn2mont(x2, x2, field);
    mod(y2, field);
    _env.bn2mont(y2, y2, field);

    // point_add(x1, y1, z1, x1, y1, one, x2, y2, one, field, g_a, np0);
    point_add(x1, y1, z1, x2, y2, one, x1, y1, one, field, g_a, np0);
    // point_add_ipp(x1, y1, z1, x1, y1, one, x2, y2, one, field, g_a, np0);
    // point_add_ipp(x1, y1, z1, x2, y2, one, x1, y1, one, field, g_a, np0);
    // point_add(x1, y1, z1, x1, y1, one, one, one, zero, field, g_a, np0);
    // point_add(x1, y1, z1, one, one, zero, x1, y1, one, field, g_a, np0);

    // _env.set(tmp, z1);
    conv_affine_x(tmp, x1, z1, field, np0);

    // point_add(x1, y1, z1, x1, y1, z1, r, s, one, field, g_a, np0);
    // point_add_ipp(x1, y1, z1, x1, y1, z1, r, s, one, field, g_a, np0);

    return 0;
  }
#endif

  /*
   * B1: verify whether r' in [1,n-1], verification failed if not
   * B2: verify whether s' in [1,n-1], verification failed if not
   * B3: set M'~=ZA || M'
   * B4: calculate e'=Hv(M'~)
   * B5: calculate t = (r' + s') modn, verification failed if t=0
   * B6: calculate the point (x1', y1')=[s']G + [t]PA
   * B7: calculate R=(e'+x1') modn, verification pass if yes, otherwise failed
   */
#ifdef DEBUG
  __device__ __forceinline__ int32_t sig_verify(const bn_t &r, bn_t &s, const bn_t &e, const bn_t &key_x, const bn_t &key_y,
                                                const bn_t &order, const bn_t &g_x, const bn_t &g_y, const bn_t &field,
                                                bn_t &g_a, bn_t &tmp)
#else
  __device__ __forceinline__ int32_t sig_verify(const bn_t &r, const bn_t &s, const bn_t &e, const bn_t &key_x,
                                                const bn_t &key_y, const bn_t &order, const bn_t &g_x, const bn_t &g_y,
                                                const bn_t &field, bn_t &g_a)
#endif
  {
    bn_t t, x1, y1, z1, x2, y2, z2, one;
    uint32_t np0;

    if (_env.compare_ui32(r, 1) < 0 || _env.compare_ui32(s, 1) < 0 || _env.compare(order, r) <= 0 ||
        _env.compare(order, s) <= 0) {
      return 0;
    }

    mod_add(t, r, s, order);

    if (_env.equals_ui32(t, 0)) {
      return 0;
    }

    _env.set_ui32(one, 1);
    np0 = _env.bn2mont(one, one, field);
    mod(g_a, field);
    _env.bn2mont(g_a, g_a, field);

    // s * generator + t * pkey
    _env.set(x1, g_x);
    _env.set(y1, g_y);
    _env.set(z1, one);
    mod(x1, field);
    _env.bn2mont(x1, x1, field);
    mod(y1, field);
    _env.bn2mont(y1, y1, field);
    point_mult(x1, y1, z1, x1, y1, z1, s, field, g_a, np0);

    _env.set(x2, key_x);
    _env.set(y2, key_y);
    _env.set(z2, one);
    mod(x2, field);
    _env.bn2mont(x2, x2, field);
    mod(y2, field);
    _env.bn2mont(y2, y2, field);
    point_mult(x2, y2, z2, x2, y2, z2, t, field, g_a, np0);

    point_add(x1, y1, z1, x1, y1, z1, x2, y2, z2, field, g_a, np0);

    conv_affine_x(x1, x1, z1, field, np0);

    mod_add(t, e, x1, order);

    if (_env.compare(r, t)) {
      return 0;
    }

    return 1;
  }

  __host__ static instance_t *generate_instances(uint32_t count) {
    instance_t *instances = (instance_t *)malloc(sizeof(instance_t) * count);

    for (int index = 0; index < count; index++) {
      // #ifdef DEBUG
      //       set_words(instances[index].r._limbs, "40F1EC59F793D9F49E09DCEF49130D4194F79FB1EED2CAA55BACDB49C4E755D1",
      //                 params::BITS / 32);
      //       set_words(instances[index].s._limbs, "6FC6DAC32C5D5CF10C77DFB20F7C2EB667A457872FB09EC56327A67EC7DEEBE7",
      //                 params::BITS / 32);
      //       set_words(instances[index].key_x._limbs, "7DEACE5FD121BC385A3C6317249F413D28C17291A60DFD83B835A45392D22B0A",
      //                 params::BITS / 32);
      //       set_words(instances[index].key_y._limbs, "2E49D5E5279E5FA91E71FD8F693A64A3C4A9461115A4FC9D79F34EDC8BDDEBD0",
      //                 params::BITS / 32);
      // #else
      set_words(instances[index].r._limbs, "40F1EC59F793D9F49E09DCEF49130D4194F79FB1EED2CAA55BACDB49C4E755D1",
                params::BITS / 32);
      set_words(instances[index].s._limbs, "6FC6DAC32C5D5CF10C77DFB20F7C2EB667A457872FB09EC56327A67EC7DEEBE7",
                params::BITS / 32);
      set_words(instances[index].key_x._limbs, "AE4C7798AA0F119471BEE11825BE46202BB79E2A5844495E97C04FF4DF2548A",
                params::BITS / 32);
      set_words(instances[index].key_y._limbs, "7C0240F88F1CD4E16352A73C17B7F16F07353E53A176D684A9FE0C6BB798E857",
                params::BITS / 32);
      // #endif
      set_words(instances[index].e._limbs, "B524F552CD82B8B028476E005C377FB19A87E6FC682D48BB5D42E3D9B9EFFE76",
                params::BITS / 32);
    }
    return instances;
  }

  __host__ static void verify_results(instance_t *instances, uint32_t count, int32_t *results) {
    for (int index = 0; index < count; index++) {
      int openssl_result = -1;

      // TODO: call OpenSSL sig verify here
      openssl_result = 1;

#ifdef DEBUG
      print_words(instances[index].r._limbs, params::BITS / 32);
#endif

      if (openssl_result != results[index]) {
        printf("Wrong result %d on instance %d\n", results[index], index);
        break;
      }
    }
  }
};

template <class params>
__global__ void kernel_sig_verify(cgbn_error_report_t *report, typename gsv_t<params>::instance_t *instances,
                                  uint32_t instance_count, typename gsv_t<params>::ec_t ec, int32_t *results) {
  int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;
  if (instance >= instance_count) return;

  typedef gsv_t<params> local_gsv_t;

  local_gsv_t gsv(cgbn_report_monitor, report, instance);
  typename local_gsv_t::bn_t r, s, e, key_x, key_y, order, g_x, g_y, field, g_a;

#ifdef DEBUG
  typename local_gsv_t::bn_t tmp;
#endif

  cgbn_load(gsv._env, r, &(instances[instance].r));
  cgbn_load(gsv._env, s, &(instances[instance].s));
  cgbn_load(gsv._env, e, &(instances[instance].e));
  cgbn_load(gsv._env, key_x, &(instances[instance].key_x));
  cgbn_load(gsv._env, key_y, &(instances[instance].key_y));

  cgbn_load(gsv._env, order, &(ec.order));
  cgbn_load(gsv._env, g_x, &(ec.g_x));
  cgbn_load(gsv._env, g_y, &(ec.g_y));
  cgbn_load(gsv._env, field, &(ec.field));
  cgbn_load(gsv._env, g_a, &(ec.g_a));

#ifdef DEBUG
  results[instance] = gsv.sig_verify(r, s, e, key_x, key_y, order, g_x, g_y, field, g_a, tmp);
  // results[instance] = gsv.debug_kernel(r, s, e, key_x, key_y, order, g_x, g_y, field, g_a, tmp);
  cgbn_store(gsv._env, &(instances[instance].r), tmp);
#else
  results[instance] = gsv.sig_verify(r, s, e, key_x, key_y, order, g_x, g_y, field, g_a);
#endif
}

template <class params>
void test_sig_verify(uint32_t instance_count) {
  long t_start, t_end;
  struct timeval timecheck;

  typedef typename gsv_t<params>::instance_t instance_t;
  typedef typename gsv_t<params>::ec_t ec_t;

  instance_t *instances, *d_instances;
  ec_t sm2;
  int32_t *results, *d_results;  // signature verification result, 1 is true, 0 is false
  cgbn_error_report_t *report;
  int32_t TPB = (params::TPB == 0) ? 128 : params::TPB;
  int32_t TPI = params::TPI, IPB = TPB / TPI;

  results = (int32_t *)malloc(sizeof(int32_t) * instance_count);
  instances = gsv_t<params>::generate_instances(instance_count);

  set_words(sm2.order._limbs, "8542D69E4C044F18E8B92435BF6FF7DD297720630485628D5AE74EE7C32E79B7", params::BITS / 32);
  // #ifdef DEBUG
  //   set_words(sm2.g_x._limbs, "1657FA75BF2ADCDC3C1F6CF05AB7B45E04D3ACBE8E4085CFA669CB2564F17A9F", params::BITS / 32);
  //   set_words(sm2.g_y._limbs, "19F0115F21E16D2F5C3A485F8575A128BBCDDF80296A62F6AC2EB842DD058E50", params::BITS / 32);
  // #else
  set_words(sm2.g_x._limbs, "421DEBD61B62EAB6746434EBC3CC315E32220B3BADD50BDC4C4E6C147FEDD43D", params::BITS / 32);
  set_words(sm2.g_y._limbs, "0680512BCBB42C07D47349D2153B70C4E5D7FDFCBFA36EA1A85841B9E46E09A2", params::BITS / 32);
  // #endif
  set_words(sm2.field._limbs, "8542D69E4C044F18E8B92435BF6FF7DE457283915C45517D722EDB8B08F1DFC3", params::BITS / 32);
  set_words(sm2.g_a._limbs, "787968B4FA32C3FD2417842E73BBFEFF2F3C848B6831D7E0EC65228B3937E498", params::BITS / 32);

  // TODO: measure memory copy time separately
  gettimeofday(&timecheck, NULL);
  t_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&d_instances, sizeof(instance_t) * instance_count));
  CUDA_CHECK(cudaMalloc((void **)&d_results, sizeof(int32_t) * instance_count));
  CUDA_CHECK(cudaMemcpy(d_instances, instances, sizeof(instance_t) * instance_count, cudaMemcpyHostToDevice));

  CUDA_CHECK(cgbn_error_report_alloc(&report));

  kernel_sig_verify<params><<<(instance_count + IPB - 1) / IPB, TPB>>>(report, d_instances, instance_count, sm2, d_results);

  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

#ifdef DEBUG
  CUDA_CHECK(cudaMemcpy(instances, d_instances, sizeof(instance_t) * instance_count, cudaMemcpyDeviceToHost));
#endif

  CUDA_CHECK(cudaMemcpy(results, d_results, sizeof(int32_t) * instance_count, cudaMemcpyDeviceToHost));

  gettimeofday(&timecheck, NULL);
  t_end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
  printf("Wall time: %lf s\n", (double)(t_end - t_start) / 1000);

  gsv_t<params>::verify_results(instances, instance_count, results);

  free(instances);
  free(results);
  CUDA_CHECK(cudaFree(d_instances));
  CUDA_CHECK(cudaFree(d_results));
  CUDA_CHECK(cgbn_error_report_free(report));
}

#ifdef DEBUG
#define INSTANCES 1
#else
#define INSTANCES 1280000
#endif

int main() {
  // TODO: parameter tuning
  typedef gsv_params_t<16, 512> params;  // threads per instance, instance size

  test_sig_verify<params>(INSTANCES);

  return 0;
}
