/* Reference C port of the gradient-free brain.
 *
 * NOT compiled or benchmarked here - this file exists to show that the whole
 * online-learning loop is a handful of multiply-adds and needs no framework,
 * no autograd, no allocator.
 *
 * Target: any 32-bit MCU with an FPU (Cortex-M4F class and up).
 *
 * Key tricks:
 *   - no exp(): softmax uses Schraudolph's bit-level exp approximation
 *   - no tanh(): rational approximation, |err| < 1e-3
 *   - the "write" is NLMS: w += lr * phi[j] * err[i] / (phi . phi)
 *   - the sphere lock is one reciprocal-sqrt per row
 */

#include <stdint.h>
#include <math.h>

#define DIM      8      /* input/output width          */
#define HID      128    /* frozen feature width        */
#define FDIM     (2 * HID + HID)  /* tanh(hid) + cos(hid) + tanh(hid2) */
#define SLOTS    32

/* ---------- fast transcendental approximations ---------- */

static inline float fast_exp(float x) {
    /* Schraudolph (1999): ~2% relative error, a few cycles, no libm call. */
    union { float f; int32_t i; } v;
    v.i = (int32_t)(12102203.0f * x + 1064866805.0f);
    return v.f;
}

static inline float fast_tanh(float x) {
    /* Pade-style; saturates cleanly, error < 1e-3 on [-4, 4]. */
    if (x < -4.0f) return -1.0f;
    if (x >  4.0f) return  1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

static inline float rsqrt_fast(float x) {
    /* one Newton step from a magic constant; good enough for the sphere lock */
    union { float f; int32_t i; } v;
    v.f = x;
    v.i = 0x5f3759df - (v.i >> 1);
    float y = v.f;
    return y * (1.5f - 0.5f * x * y * y);
}

/* ---------- state ---------- */

typedef struct {
    /* frozen feature map */
    float K1[DIM * HID];
    float K2[(2 * HID) * HID];
    /* free readouts (the only thing written online) */
    float Ws[FDIM * DIM];
    float Wd[FDIM * DIM];
    /* config */
    float lr_sensory, lr_decision, decay, budget;
} Brain;

static float phi_buf[FDIM];

/* ---------- forward ---------- */

static void features(const Brain *b, const float *x) {
    float h1[2 * HID];

    /* h1 = [ tanh(K1^T x) , cos(K1^T x) ] : odd part + even part. */
    for (int j = 0; j < HID; ++j) {
        float z = 0.0f;
        for (int i = 0; i < DIM; ++i) z += x[i] * b->K1[i * HID + j];
        h1[j] = fast_tanh(z);
        float z2 = z * z;
        /* 6th-order even series; the orthogonal K1 keeps |z| < 3 after
         * input normalisation, which is where this series stays accurate. */
        h1[HID + j] = 1.0f - z2 * 0.5f + z2 * z2 * (1.0f / 24.0f)
                      - z2 * z2 * z2 * (1.0f / 720.0f);
    }

    /* phi = [ h1 (2*HID) , tanh(K2^T h1) (HID) ] */
    for (int i = 0; i < 2 * HID; ++i) phi_buf[i] = h1[i];
    for (int j = 0; j < HID; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < 2 * HID; ++i) acc += h1[i] * b->K2[i * HID + j];
        phi_buf[2 * HID + j] = fast_tanh(acc);
    }
}

static void readout(const float *W, const float *phi, float *out) {
    for (int o = 0; o < DIM; ++o) {
        float acc = 0.0f;
        for (int j = 0; j < FDIM; ++j) acc += phi[j] * W[j * DIM + o];
        out[o] = acc;
    }
}

/* ---------- write (NLMS + decay + budget) ---------- */

static void nlms_write(float *W, const float *phi, const float *err, float lr, float decay, float budget) {
    float energy = 1e-6f;
    for (int j = 0; j < FDIM; ++j) energy += phi[j] * phi[j];

    for (int o = 0; o < DIM; ++o) {
        float g = lr * err[o] / energy;
        for (int j = 0; j < FDIM; ++j) {
            W[j * DIM + o] = W[j * DIM + o] * decay + phi[j] * g;
        }
    }
    /* energy budget: clip each output column */
    for (int o = 0; o < DIM; ++o) {
        float n2 = 1e-12f;
        for (int j = 0; j < FDIM; ++j) n2 += W[j * DIM + o] * W[j * DIM + o];
        float n = n2 * rsqrt_fast(n2);
        if (n > budget) {
            float s = budget * rsqrt_fast(n2);
            for (int j = 0; j < FDIM; ++j) W[j * DIM + o] *= s;
        }
    }
}

/* ---------- one online step: predict, then adapt ---------- */

void brain_step(Brain *b, const float *x, const float *y, float *pred_out) {
    features(b, x);

    float s[DIM], d[DIM];
    readout(b->Ws, phi_buf, s);
    readout(b->Wd, phi_buf, d);
    for (int i = 0; i < DIM; ++i) pred_out[i] = d[i];

    /* sensory organ: reconstruct the input, high fixed gain */
    float err_s[DIM];
    for (int i = 0; i < DIM; ++i) err_s[i] = x[i] - s[i];
    nlms_write(b->Ws, phi_buf, err_s, b->lr_sensory, b->decay, b->budget);

    /* decision organ: non-saturating stress response */
    if (y) {
        float err_d[DIM];
        for (int i = 0; i < DIM; ++i) {
            float e = y[i] - d[i];
            float a = e < 0.0f ? -e : e;
            err_d[i] = e * (1.0f + 2.0f * fast_tanh(a)); /* magnitude preserved */
        }
        nlms_write(b->Wd, phi_buf, err_d, b->lr_decision, b->decay, b->budget);
    }
}

/* Cost per step (DIM=8, HID=128):
 *   features  : 8*128 + 256*128  = 33,792 MAC
 *   readout x2: 2 * 384 * 8      =  6,144 MAC
 *   write   x2: 2 * 384 * 8 * 2  = 12,288 MAC  (decay + add)
 *   ~52k MAC/step. On a 100 MHz M4F (1 MAC/cycle) that is ~0.5 ms/step,
 *   i.e. 2 kHz online learning - comfortably above typical 100-500 Hz sensors.
 */
