#include "strict_fp32_contract.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <intrin.h>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using transfer_fp32::Geometry;
using transfer_fp32::Scalar;

#define CUDA_CHECK(call)                                                                  \
  do {                                                                                    \
    const cudaError_t error__ = (call);                                                    \
    if (error__ != cudaSuccess)                                                            \
      throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(error__));      \
  } while (0)

#define CUBLAS_CHECK(call)                                                                \
  do {                                                                                    \
    const cublasStatus_t status__ = (call);                                                \
    if (status__ != CUBLAS_STATUS_SUCCESS)                                                 \
      throw std::runtime_error("cuBLAS status " + std::to_string(int(status__)));          \
  } while (0)

namespace {

constexpr int J1 = transfer_fp32::kDifferentiatedJetComponents;
#if defined(TRANSFER_VORTICITY_RAYLEIGH)
// Vorticity-Rayleigh uses omega=curl(div A) and grad(omega), so A is needed
// through third derivatives and no unused derivative channel is propagated.
constexpr int J = transfer_fp32::kJet3Components;
#elif defined(TRANSFER_VECTOR_LIBRARY) || defined(TRANSFER_VELOCITY_RAYLEIGH)
// Full-grid reconstruction needs A through second derivatives to form velocity
// and vorticity. The velocity-Rayleigh objective likewise needs A through
// second derivatives to form u=div(A) and grad(u). Neither path propagates
// unused third derivatives.
constexpr int J = transfer_fp32::kJet2Components;
#else
// The potential Dirichlet Rayleigh quotient uses only A and grad(A).
constexpr int J = J1;
#endif
constexpr int H = transfer_fp32::kHiddenWidth;
constexpr int L = transfer_fp32::kHiddenLayers;
constexpr int IN = transfer_fp32::kRawInputs;
constexpr int OUT = transfer_fp32::kVectorComponents;
constexpr int P = transfer_fp32::kParametersPerMode;

#if defined(TRANSFER_VELOCITY_RAYLEIGH)
constexpr bool kVelocityRayleigh = true;
#else
constexpr bool kVelocityRayleigh = false;
#endif
#if defined(TRANSFER_VORTICITY_RAYLEIGH)
constexpr bool kVorticityRayleigh = true;
#else
constexpr bool kVorticityRayleigh = false;
#endif

constexpr const char* objective_name() {
  if constexpr (kVorticityRayleigh) return "vorticity gradient Rayleigh";
  if constexpr (kVelocityRayleigh) return "velocity gradient Rayleigh";
  return "potential Dirichlet Rayleigh";
}
constexpr const char* objective_slug() {
  if constexpr (kVorticityRayleigh) return "vorticity-gradient-Rayleigh";
  if constexpr (kVelocityRayleigh) return "velocity-gradient-Rayleigh";
  return "potential-Dirichlet-Rayleigh";
}
constexpr const char* rayleigh_numerator() {
  if constexpr (kVorticityRayleigh) return "integral ||grad(curl(div A))||^2";
  if constexpr (kVelocityRayleigh) return "integral ||grad(div A)||^2";
  return "integral ||grad A||^2";
}
constexpr const char* rayleigh_denominator() {
  if constexpr (kVorticityRayleigh) return "integral ||curl(div A)||^2";
  if constexpr (kVelocityRayleigh) return "integral ||div A||^2";
  return "integral ||A||^2";
}
constexpr const char* field_name() {
  if constexpr (kVorticityRayleigh) return "vorticity";
  if constexpr (kVelocityRayleigh) return "velocity";
  return "potential";
}

constexpr Scalar kOuterHalf = 0.49f;
constexpr Scalar kCenterMin = 0.25f;
constexpr Scalar kCenterMax = 0.75f;
constexpr Scalar kRadiusMin = 0.10f;
constexpr Scalar kRadiusMax = 0.30f;
constexpr Scalar kSoftminTau = 0.02f;
constexpr Scalar kOuterDelta = 0.18f;
constexpr Scalar kObstacleDelta = 0.10f;
constexpr Scalar kMinimumOverlap = 0.025f;
constexpr Scalar kOuterClearance = 0.03f;
constexpr Scalar kFirstOmega = 6.0f;
constexpr Scalar kHiddenOmega = 36.0f;
constexpr Scalar kBaseLearningRate = 2.8e-4f;
constexpr Scalar kMinLearningRate = 1.0e-6f;
constexpr Scalar kGradClip = 8.0f;
constexpr Scalar kMassFloor = 1.0e-12f;
// This threshold only detects an exactly unresolved MGS direction.  It is
// intentionally below the Rayleigh denominator floor: after two-pass MGS a
// legitimate high-rank residual can be much smaller than its raw unit mass,
// and it is normalized immediately before being stored.
constexpr Scalar kPeelingMassFloor = 1.0e-20f;
constexpr std::uint64_t kDefaultSeed = 20260823ULL;

struct Options {
  int modes = 1024;
  int epochs = 2000;
  int points = 9192;
  int validation_points = 4096;
  int validation_geometries = 8;
  int validate_every = 25;
  int checkpoint_every = 200;
  int stop_after = 0;
  int lr_schedule_start = 1;
  int lr_schedule_end = 0;
  std::uint64_t seed = kDefaultSeed;
  std::uint64_t fixed_geometry_token = 0ULL;
  bool fixed_geometry = false;
  fs::path fixed_geometry_file;
  fs::path geometry_dataset_file;
  fs::path result = "results/eigenfluid_transfer_native_fp32_k1024_p9192_e2000";
  fs::path resume;
  bool self_test = false;
};

template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { resize(count); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_) cudaFree(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }
  ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
  void resize(std::size_t count) {
    if (count == size_) return;
    if (ptr_) CUDA_CHECK(cudaFree(ptr_));
    ptr_ = nullptr;
    size_ = count;
    if (count) CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
  }
  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  std::size_t size() const { return size_; }
 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

struct U128 { std::uint64_t lo, hi; };

U128 multiply_low_128(U128 a, U128 b) {
  std::uint64_t product_hi = 0;
  const std::uint64_t product_lo = _umul128(a.lo, b.lo, &product_hi);
  product_hi += a.lo * b.hi;
  product_hi += a.hi * b.lo;
  return {product_lo, product_hi};
}

U128 add_128(U128 a, U128 b) {
  const std::uint64_t lo = a.lo + b.lo;
  return {lo, a.hi + b.hi + std::uint64_t(lo < a.lo)};
}

std::uint64_t splitmix64(std::uint64_t& value) {
  value += 0x9e3779b97f4a7c15ULL;
  std::uint64_t z = value;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

class Pcg64 {
 public:
  explicit Pcg64(std::uint64_t seed) {
    std::uint64_t s = seed;
    state_ = {splitmix64(s), splitmix64(s)};
    increment_ = {splitmix64(s) | 1ULL, splitmix64(s)};
    next_u64();
  }
  std::uint64_t next_u64() {
    const U128 old = state_;
    constexpr U128 multiplier{0x4385df649fccf645ULL, 0x2360ed051fc65da4ULL};
    state_ = add_128(multiply_low_128(old, multiplier), increment_);
    const std::uint64_t value = old.hi ^ old.lo;
    const unsigned rotation = unsigned(old.hi >> 58);
    return _rotr64(value, rotation);
  }
  Scalar uniform() { return Scalar(next_u64() >> 40) * 0x1.0p-24f; }
 private:
  U128 state_{};
  U128 increment_{};
};

bool connected(const Geometry& geometry) {
  bool edge[3][3]{};
  for (int a = 0; a < 3; ++a) for (int b = a + 1; b < 3; ++b) {
    const Scalar dx = geometry.sphere[a][0] - geometry.sphere[b][0];
    const Scalar dy = geometry.sphere[a][1] - geometry.sphere[b][1];
    const Scalar dz = geometry.sphere[a][2] - geometry.sphere[b][2];
    const Scalar distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    edge[a][b] = edge[b][a] = geometry.sphere[a][3] + geometry.sphere[b][3] - distance >= kMinimumOverlap;
  }
  bool seen[3]{true, false, false};
  for (int pass = 0; pass < 3; ++pass)
    for (int a = 0; a < 3; ++a) if (seen[a])
      for (int b = 0; b < 3; ++b) if (edge[a][b]) seen[b] = true;
  return seen[0] && seen[1] && seen[2];
}

bool inside_rounded_cube(const Geometry& geometry) {
  constexpr int directions = 128;
  constexpr Scalar pi = 3.14159265358979323846f;
  const Scalar available_extent = kOuterHalf - kOuterClearance;
  for (int s = 0; s < 3; ++s) for (int i = 0; i < directions; ++i) {
    const Scalar z = 1.0f - 2.0f * (Scalar(i) + 0.5f) / Scalar(directions);
    const Scalar radius = std::sqrt(std::max(0.0f, 1.0f - z * z));
    const Scalar angle = pi * (3.0f - std::sqrt(5.0f)) * Scalar(i);
    const Scalar q[3] = {radius * std::cos(angle), radius * std::sin(angle), z};
    Scalar sum = 0.0f;
    for (int d = 0; d < 3; ++d) {
      const Scalar x = std::abs((geometry.sphere[s][d] + geometry.sphere[s][3] * q[d] - 0.5f) / available_extent);
      const Scalar x2 = x * x;
      sum += x2 * x2 * x2 * x2;
    }
    if (sum >= 1.0f) return false;
  }
  return true;
}

Geometry geometry_from_token(std::uint64_t seed, std::uint64_t token) {
  Pcg64 rng(seed + 104729ULL * token);
  for (int attempt = 0; attempt < 100000; ++attempt) {
    Geometry g{};
    for (int s = 0; s < 3; ++s) {
      for (int d = 0; d < 3; ++d)
        g.sphere[s][d] = kCenterMin + (kCenterMax - kCenterMin) * rng.uniform();
      g.sphere[s][3] = kRadiusMin + (kRadiusMax - kRadiusMin) * rng.uniform();
    }
    if (connected(g) && inside_rounded_cube(g)) return g;
  }
  throw std::runtime_error("could not draw a connected transfer geometry");
}

Geometry geometry_from_file(const fs::path& path) {
  Geometry geometry{};
  std::ifstream stream(path,std::ios::binary);
  stream.read(reinterpret_cast<char*>(geometry.sphere),sizeof(geometry.sphere));
  if(!stream)throw std::runtime_error("cannot read fixed geometry file: "+path.string());
  return geometry;
}

std::vector<Geometry> geometry_dataset_from_file(const fs::path& path, int expected_records) {
  if (expected_records < 1) throw std::runtime_error("geometry dataset must contain records");
  const std::uintmax_t expected_bytes = std::uintmax_t(expected_records) * sizeof(Geometry);
  if (!fs::exists(path) || fs::file_size(path) != expected_bytes)
    throw std::runtime_error("geometry dataset size mismatch: " + path.string());
  std::vector<Geometry> geometries;
  geometries.resize(std::size_t(expected_records));
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(geometries.data()), std::streamsize(expected_bytes));
  if (!stream) throw std::runtime_error("cannot read geometry dataset: " + path.string());
  for (int index = 0; index < expected_records; ++index) {
    if (!connected(geometries[std::size_t(index)]) || !inside_rounded_cube(geometries[std::size_t(index)]))
      throw std::runtime_error("invalid geometry dataset record " + std::to_string(index));
  }
  return geometries;
}

Scalar outer_level(Scalar x, Scalar y, Scalar z) {
  const Scalar q[3] = {(x - 0.5f) / kOuterHalf, (y - 0.5f) / kOuterHalf, (z - 0.5f) / kOuterHalf};
  Scalar sum = 0.0f;
  for (Scalar a : q) { const Scalar a2 = a * a; sum += a2 * a2 * a2 * a2; }
  return 1.0f - sum;
}

Scalar obstacle_level(Scalar x, Scalar y, Scalar z, const Geometry& g) {
  Scalar distances[3];
  Scalar minimum = 1.0e30f;
  for (int s = 0; s < 3; ++s) {
    const Scalar dx = x - g.sphere[s][0], dy = y - g.sphere[s][1], dz = z - g.sphere[s][2];
    distances[s] = std::sqrt(dx * dx + dy * dy + dz * dz) - g.sphere[s][3];
    minimum = std::min(minimum, distances[s]);
  }
  Scalar sum = 0.0f;
  for (Scalar distance : distances) sum += std::exp(-(distance - minimum) / kSoftminTau);
  return minimum - kSoftminTau * std::log(sum);
}

bool is_fluid(Scalar x, Scalar y, Scalar z, const Geometry& g) {
  return outer_level(x, y, z) > 0.0f && obstacle_level(x, y, z, g) > 0.0f;
}

std::vector<Scalar> sample_fluid_points(int count, const Geometry& g, std::uint64_t seed) {
  Pcg64 rng(seed);
  std::vector<Scalar> points;
  points.reserve(std::size_t(count) * 3);
  while (int(points.size() / 3) < count) {
    const Scalar x = rng.uniform(), y = rng.uniform(), z = rng.uniform();
    if (is_fluid(x, y, z, g)) { points.push_back(x); points.push_back(y); points.push_back(z); }
  }
  return points;
}

Scalar approximate_volume(const Geometry& g, std::uint64_t seed, int samples) {
  Pcg64 rng(seed);
  int accepted = 0;
  for (int i = 0; i < samples; ++i)
    accepted += is_fluid(rng.uniform(), rng.uniform(), rng.uniform(), g) ? 1 : 0;
  return Scalar(accepted) / Scalar(samples);
}

__device__ __constant__ int kMx[transfer_fp32::kJet3Components] = {0,1,0,0,2,1,1,0,0,0,3,2,2,1,1,1,0,0,0,0};
__device__ __constant__ int kMy[transfer_fp32::kJet3Components] = {0,0,1,0,0,1,0,2,1,0,0,1,0,2,1,0,3,2,1,0};
__device__ __constant__ int kMz[transfer_fp32::kJet3Components] = {0,0,0,1,0,0,1,0,1,2,0,0,1,0,1,2,0,1,2,3};

struct Taylor3 { Scalar c[J]; };

__device__ int multi_index(int x, int y, int z) {
  for (int i = 0; i < J; ++i) if (kMx[i] == x && kMy[i] == y && kMz[i] == z) return i;
  return -1;
}

__device__ Taylor3 jet_constant(Scalar value) {
  Taylor3 r{}; r.c[0] = value; return r;
}

__device__ Taylor3 jet_coordinate(Scalar value, int derivative) {
  Taylor3 r{}; r.c[0] = value; r.c[1 + derivative] = 1.0f; return r;
}

__device__ Taylor3 jet_add(Taylor3 a, Taylor3 b) {
  Taylor3 r; for (int i = 0; i < J; ++i) r.c[i] = a.c[i] + b.c[i]; return r;
}

__device__ Taylor3 jet_sub(Taylor3 a, Taylor3 b) {
  Taylor3 r; for (int i = 0; i < J; ++i) r.c[i] = a.c[i] - b.c[i]; return r;
}

__device__ Taylor3 jet_scale(Taylor3 a, Scalar scale) {
  Taylor3 r; for (int i = 0; i < J; ++i) r.c[i] = a.c[i] * scale; return r;
}

__device__ Taylor3 jet_mul(Taylor3 a, Taylor3 b) {
  Taylor3 r{};
  constexpr int maximum_degree = J == J1 ? 1 : (J == transfer_fp32::kJet2Components ? 2 : 3);
  for (int i = 0; i < J; ++i) for (int j = 0; j < J; ++j) {
    const int degree = kMx[i] + kMy[i] + kMz[i] + kMx[j] + kMy[j] + kMz[j];
    if (degree > maximum_degree) continue;
    const int k = multi_index(kMx[i] + kMx[j], kMy[i] + kMy[j], kMz[i] + kMz[j]);
    if (k >= 0) r.c[k] += a.c[i] * b.c[j];
  }
  return r;
}

__device__ Taylor3 jet_unary(Taylor3 x, Scalar f0, Scalar f1, Scalar half_f2, Scalar sixth_f3) {
  Taylor3 delta = x; delta.c[0] = 0.0f;
  Taylor3 r = jet_constant(f0);
  r = jet_add(r, jet_scale(delta, f1));
  if constexpr (J >= transfer_fp32::kJet2Components) {
    const Taylor3 delta2 = jet_mul(delta, delta);
    r = jet_add(r, jet_scale(delta2, half_f2));
    if constexpr (J >= transfer_fp32::kJet3Components) {
      const Taylor3 delta3 = jet_mul(delta2, delta);
      r = jet_add(r, jet_scale(delta3, sixth_f3));
    }
  }
  return r;
}

__device__ Taylor3 jet_exp(Taylor3 x) {
  const Scalar f = expf(x.c[0]); return jet_unary(x, f, f, 0.5f * f, f / 6.0f);
}

__device__ Taylor3 jet_log(Taylor3 x) {
  const Scalar a = x.c[0], inv = 1.0f / a;
  return jet_unary(x, logf(a), inv, -0.5f * inv * inv, inv * inv * inv / 3.0f);
}

__device__ Taylor3 jet_inverse(Taylor3 x) {
  const Scalar inv = 1.0f / x.c[0];
  return jet_unary(x, inv, -inv * inv, inv * inv * inv, -inv * inv * inv * inv);
}

__device__ Taylor3 jet_sqrt(Taylor3 x) {
  const Scalar root = sqrtf(fmaxf(x.c[0], 1.0e-16f));
  const Scalar inv = 1.0f / root;
  return jet_unary(x, root, 0.5f * inv, -0.125f * inv * inv * inv,
                   0.0625f * inv * inv * inv * inv * inv);
}

__device__ Taylor3 jet_div(Taylor3 a, Taylor3 b) { return jet_mul(a, jet_inverse(b)); }

__device__ Taylor3 envelope_jet(Scalar px, Scalar py, Scalar pz, const Geometry& g) {
  Taylor3 xyz[3] = {jet_coordinate(px, 0), jet_coordinate(py, 1), jet_coordinate(pz, 2)};
  Taylor3 outer_sum = jet_constant(0.0f);
  for (int d = 0; d < 3; ++d) {
    Taylor3 q = jet_scale(jet_sub(xyz[d], jet_constant(0.5f)), 1.0f / kOuterHalf);
    Taylor3 q2 = jet_mul(q, q), q4 = jet_mul(q2, q2);
    outer_sum = jet_add(outer_sum, jet_mul(q4, q4));
  }
  Taylor3 outer = jet_sub(jet_constant(1.0f), outer_sum);
  Taylor3 distances[3];
  int minimum_index = 0;
  for (int s = 0; s < 3; ++s) {
    Taylor3 radius2 = jet_constant(0.0f);
    for (int d = 0; d < 3; ++d) {
      Taylor3 delta = jet_sub(xyz[d], jet_constant(g.sphere[s][d]));
      radius2 = jet_add(radius2, jet_mul(delta, delta));
    }
    distances[s] = jet_sub(jet_sqrt(radius2), jet_constant(g.sphere[s][3]));
    if (distances[s].c[0] < distances[minimum_index].c[0]) minimum_index = s;
  }
  const Taylor3 minimum_distance = distances[minimum_index];
  Taylor3 exponential_sum = jet_constant(0.0f);
  for (int s = 0; s < 3; ++s) {
    const Taylor3 shifted = jet_sub(distances[s], minimum_distance);
    exponential_sum = jet_add(exponential_sum, jet_exp(jet_scale(shifted, -1.0f / kSoftminTau)));
  }
  Taylor3 obstacle = jet_sub(minimum_distance, jet_scale(jet_log(exponential_sum), kSoftminTau));
  Taylor3 outer_factor = jet_div(outer, jet_add(jet_constant(kOuterDelta), outer));
  Taylor3 obstacle_factor = jet_div(obstacle, jet_add(jet_constant(kObstacleDelta), obstacle));
  const Taylor3 combined = jet_mul(outer_factor, obstacle_factor);
  return jet_mul(combined, combined);
}

__device__ Scalar actual_factor(int index) {
  if (index == 4 || index == 7 || index == 9 || index == 11 || index == 12 ||
      index == 13 || index == 15 || index == 17 || index == 18) return 2.0f;
  if (index == 10 || index == 16 || index == 19) return 6.0f;
  return 1.0f;
}

__global__ void prepare_inputs_kernel(const Scalar* points, int count, Geometry geometry,
                                      Scalar* features, Scalar* envelope) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= count) return;
  const Scalar x = points[3 * n], y = points[3 * n + 1], z = points[3 * n + 2];
  for (int j = 0; j < J; ++j) for (int input = 0; input < IN; ++input)
    features[j * IN * count + input + IN * n] = 0.0f;
  features[0 * IN * count + 0 + IN * n] = 2.0f * x - 1.0f;
  features[0 * IN * count + 1 + IN * n] = 2.0f * y - 1.0f;
  features[0 * IN * count + 2 + IN * n] = 2.0f * z - 1.0f;
  features[1 * IN * count + 0 + IN * n] = 2.0f;
  features[2 * IN * count + 1 + IN * n] = 2.0f;
  features[3 * IN * count + 2 + IN * n] = 2.0f;
  for (int sphere = 0; sphere < 3; ++sphere) {
    for (int d = 0; d < 3; ++d) {
      const Scalar u = (geometry.sphere[sphere][d] - kCenterMin) / (kCenterMax - kCenterMin);
      features[(3 + 4 * sphere + d) + IN * n] = 2.0f * u - 1.0f;
    }
    const Scalar u = (geometry.sphere[sphere][3] - kRadiusMin) / (kRadiusMax - kRadiusMin);
    features[(6 + 4 * sphere) + IN * n] = 2.0f * u - 1.0f;
  }
  const Taylor3 e = envelope_jet(x, y, z, geometry);
  for (int j = 0; j < J; ++j) envelope[j * count + n] = e.c[j] * actual_factor(j);
}

__global__ void affine_kernel(Scalar* values, const Scalar* bias, int output, int count,
                              Scalar omega, int jet_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = jet_count * output * count;
  if (index >= total) return;
  const int jet = index / (output * count);
  const int local = index - jet * output * count;
  const int neuron = local % output;
  values[index] = omega * (values[index] + (jet == 0 ? bias[neuron] : 0.0f));
}

__global__ void sine_jet_kernel(const Scalar* input, Scalar* output, int neurons) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= neurons) return;
  Scalar x[J];
  for (int j = 0; j < J; ++j) x[j] = input[j * neurons + n];
  const Scalar s = sinf(x[0]), c = cosf(x[0]);
  output[0 * neurons + n] = s;
  output[1 * neurons + n] = c * x[1];
  output[2 * neurons + n] = c * x[2];
  output[3 * neurons + n] = c * x[3];
  if constexpr (J >= transfer_fp32::kJet2Components) {
    output[4 * neurons + n] = c * x[4] - s * x[1] * x[1];
    output[5 * neurons + n] = c * x[5] - s * x[1] * x[2];
    output[6 * neurons + n] = c * x[6] - s * x[1] * x[3];
    output[7 * neurons + n] = c * x[7] - s * x[2] * x[2];
    output[8 * neurons + n] = c * x[8] - s * x[2] * x[3];
    output[9 * neurons + n] = c * x[9] - s * x[3] * x[3];
  }
  if constexpr (J >= transfer_fp32::kJet3Components) {
    output[10 * neurons + n] = c*x[10] - 3.0f*s*x[4]*x[1] - c*x[1]*x[1]*x[1];
    output[11 * neurons + n] = c*x[11] - s*(x[4]*x[2]+2.0f*x[5]*x[1]) - c*x[1]*x[1]*x[2];
    output[12 * neurons + n] = c*x[12] - s*(x[4]*x[3]+2.0f*x[6]*x[1]) - c*x[1]*x[1]*x[3];
    output[13 * neurons + n] = c*x[13] - s*(x[7]*x[1]+2.0f*x[5]*x[2]) - c*x[1]*x[2]*x[2];
    output[14 * neurons + n] = c*x[14] - s*(x[5]*x[3]+x[6]*x[2]+x[8]*x[1]) - c*x[1]*x[2]*x[3];
    output[15 * neurons + n] = c*x[15] - s*(x[9]*x[1]+2.0f*x[6]*x[3]) - c*x[1]*x[3]*x[3];
    output[16 * neurons + n] = c*x[16] - 3.0f*s*x[7]*x[2] - c*x[2]*x[2]*x[2];
    output[17 * neurons + n] = c*x[17] - s*(x[7]*x[3]+2.0f*x[8]*x[2]) - c*x[2]*x[2]*x[3];
    output[18 * neurons + n] = c*x[18] - s*(x[9]*x[2]+2.0f*x[8]*x[3]) - c*x[2]*x[3]*x[3];
    output[19 * neurons + n] = c*x[19] - 3.0f*s*x[9]*x[3] - c*x[3]*x[3]*x[3];
  }
}

__global__ void build_vector_field_kernel(const Scalar* raw, const Scalar* envelope,
                                          int count, Scalar* composed_jet3,
                                          Scalar* potential, Scalar* gradient) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count * OUT) return;
  const int n = index / OUT, component = index % OUT;
  Scalar product[J]{};
  for (int left = 0; left < J; ++left) for (int right = 0; right < J; ++right) {
    if (kMx[left]+kMy[left]+kMz[left]+kMx[right]+kMy[right]+kMz[right] > 3) continue;
    const int target = multi_index(kMx[left]+kMx[right], kMy[left]+kMy[right], kMz[left]+kMz[right]);
    if (target >= 0) {
      const Scalar e = envelope[left * count + n] / actual_factor(left);
      const Scalar r = raw[right * OUT * count + component + OUT * n] / actual_factor(right);
      product[target] += e * r;
    }
  }
  for (int jet = 0; jet < J; ++jet)
    composed_jet3[jet * OUT * count + index] = product[jet] * actual_factor(jet);
  potential[index] = product[0];
  for (int d = 0; d < 3; ++d) gradient[index * 3 + d] = product[1 + d];
}

// Convert the complete antisymmetric potential (A12,A13,A23) and its analytic
// Jet2 derivatives into the complete divergence-free velocity u=div(A) and
// its analytic first derivatives.  The output layouts deliberately match the
// potential/gradient FieldWorkspace layouts so all causal peeling and MGS code
// operates in the velocity mass inner product when TRANSFER_VELOCITY_RAYLEIGH
// is enabled.
__global__ void build_velocity_field_kernel(const Scalar* composed, int count,
                                            Scalar* velocity, Scalar* velocity_gradient) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count * OUT) return;
  const int n = index / OUT, component = index % OUT;
  auto a = [&](int jet, int c) -> Scalar {
    return composed[std::size_t(jet) * OUT * count + c + OUT * n];
  };
  Scalar u = 0.0f, g0 = 0.0f, g1 = 0.0f, g2 = 0.0f;
  if (component == 0) {
    u  =  a(2,0) + a(3,1);
    g0 =  a(5,0) + a(6,1);
    g1 =  a(7,0) + a(8,1);
    g2 =  a(8,0) + a(9,1);
  } else if (component == 1) {
    u  = -a(1,0) + a(3,2);
    g0 = -a(4,0) + a(6,2);
    g1 = -a(5,0) + a(8,2);
    g2 = -a(6,0) + a(9,2);
  } else {
    u  = -a(1,1) - a(2,2);
    g0 = -a(4,1) - a(5,2);
    g1 = -a(5,1) - a(7,2);
    g2 = -a(6,1) - a(8,2);
  }
  velocity[index] = u;
  velocity_gradient[index * 3 + 0] = g0;
  velocity_gradient[index * 3 + 1] = g1;
  velocity_gradient[index * 3 + 2] = g2;
}

// Build omega=curl(div A) and grad(omega) directly from the analytic Jet3 of
// the complete antisymmetric tensor potential A=(A12,A13,A23).
__global__ void build_vorticity_field_kernel(const Scalar* composed, int count,
                                             Scalar* vorticity, Scalar* vorticity_gradient) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count * OUT) return;
  const int n = index / OUT, component = index % OUT;
  auto a = [&](int jet, int c) -> Scalar {
    return composed[std::size_t(jet) * OUT * count + c + OUT * n];
  };
  Scalar w = 0.0f, g0 = 0.0f, g1 = 0.0f, g2 = 0.0f;
  if (component == 0) {
    w  =  a(6,0) - a(5,1) - a(7,2) - a(9,2);
    g0 =  a(12,0) - a(11,1) - a(13,2) - a(15,2);
    g1 =  a(14,0) - a(13,1) - a(16,2) - a(18,2);
    g2 =  a(15,0) - a(14,1) - a(17,2) - a(19,2);
  } else if (component == 1) {
    w  =  a(8,0) + a(9,1) + a(4,1) + a(5,2);
    g0 =  a(14,0) + a(15,1) + a(10,1) + a(11,2);
    g1 =  a(17,0) + a(18,1) + a(11,1) + a(13,2);
    g2 =  a(18,0) + a(19,1) + a(12,1) + a(14,2);
  } else {
    w  = -a(4,0) - a(7,0) + a(6,2) - a(8,1);
    g0 = -a(10,0) - a(13,0) + a(12,2) - a(14,1);
    g1 = -a(11,0) - a(16,0) + a(14,2) - a(17,1);
    g2 = -a(12,0) - a(17,0) + a(15,2) - a(18,1);
  }
  vorticity[index] = w;
  vorticity_gradient[index * 3 + 0] = g0;
  vorticity_gradient[index * 3 + 1] = g1;
  vorticity_gradient[index * 3 + 2] = g2;
}

__global__ void field_to_raw_adjoint_kernel(const Scalar* potential_adjoint,
                                             const Scalar* gradient_adjoint,
                                             const Scalar* envelope, int count,
                                             Scalar* raw_adjoint) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count * OUT) return;
  const int n = index / OUT;
  Scalar value = envelope[n] * potential_adjoint[index];
  for (int d = 0; d < 3; ++d)
    value += envelope[(1 + d) * count + n] * gradient_adjoint[index * 3 + d];
  raw_adjoint[index] = value;
  for (int d = 0; d < 3; ++d)
    raw_adjoint[(1 + d) * OUT * count + index] = envelope[n] * gradient_adjoint[index * 3 + d];
}

__global__ void velocity_to_composed_adjoint_kernel(const Scalar* velocity_adjoint,
                                                     const Scalar* gradient_adjoint,
                                                     int count, Scalar* composed_adjoint) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= count) return;
  Scalar local[J * OUT]{};
  auto add = [&](int jet, int component, Scalar value) {
    local[jet * OUT + component] += value;
  };
  const Scalar ax = velocity_adjoint[OUT*n+0];
  const Scalar ay = velocity_adjoint[OUT*n+1];
  const Scalar az = velocity_adjoint[OUT*n+2];
  add(2,0, ax); add(3,1, ax);
  add(1,0,-ay); add(3,2, ay);
  add(1,1,-az); add(2,2,-az);

  const Scalar g00=gradient_adjoint[(OUT*n+0)*3+0];
  const Scalar g01=gradient_adjoint[(OUT*n+0)*3+1];
  const Scalar g02=gradient_adjoint[(OUT*n+0)*3+2];
  const Scalar g10=gradient_adjoint[(OUT*n+1)*3+0];
  const Scalar g11=gradient_adjoint[(OUT*n+1)*3+1];
  const Scalar g12=gradient_adjoint[(OUT*n+1)*3+2];
  const Scalar g20=gradient_adjoint[(OUT*n+2)*3+0];
  const Scalar g21=gradient_adjoint[(OUT*n+2)*3+1];
  const Scalar g22=gradient_adjoint[(OUT*n+2)*3+2];
  add(5,0, g00); add(6,1, g00);
  add(7,0, g01); add(8,1, g01);
  add(8,0, g02); add(9,1, g02);
  add(4,0,-g10); add(6,2, g10);
  add(5,0,-g11); add(8,2, g11);
  add(6,0,-g12); add(9,2, g12);
  add(4,1,-g20); add(5,2,-g20);
  add(5,1,-g21); add(7,2,-g21);
  add(6,1,-g22); add(8,2,-g22);

  for (int jet=0; jet<J; ++jet) for (int component=0; component<OUT; ++component)
    composed_adjoint[std::size_t(jet)*OUT*count + component + OUT*n] = local[jet*OUT+component];
}

__global__ void vorticity_to_composed_adjoint_kernel(const Scalar* vorticity_adjoint,
                                                      const Scalar* gradient_adjoint,
                                                      int count, Scalar* composed_adjoint) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= count) return;
  Scalar local[J * OUT]{};
  auto add = [&](int jet, int component, Scalar value) {
    local[jet * OUT + component] += value;
  };
  const Scalar wx = vorticity_adjoint[OUT*n+0];
  const Scalar wy = vorticity_adjoint[OUT*n+1];
  const Scalar wz = vorticity_adjoint[OUT*n+2];
  add(6,0, wx); add(5,1,-wx); add(7,2,-wx); add(9,2,-wx);
  add(8,0, wy); add(9,1, wy); add(4,1, wy); add(5,2, wy);
  add(4,0,-wz); add(7,0,-wz); add(6,2, wz); add(8,1,-wz);

  const Scalar g00=gradient_adjoint[(OUT*n+0)*3+0];
  const Scalar g01=gradient_adjoint[(OUT*n+0)*3+1];
  const Scalar g02=gradient_adjoint[(OUT*n+0)*3+2];
  const Scalar g10=gradient_adjoint[(OUT*n+1)*3+0];
  const Scalar g11=gradient_adjoint[(OUT*n+1)*3+1];
  const Scalar g12=gradient_adjoint[(OUT*n+1)*3+2];
  const Scalar g20=gradient_adjoint[(OUT*n+2)*3+0];
  const Scalar g21=gradient_adjoint[(OUT*n+2)*3+1];
  const Scalar g22=gradient_adjoint[(OUT*n+2)*3+2];
  add(12,0, g00); add(11,1,-g00); add(13,2,-g00); add(15,2,-g00);
  add(14,0, g01); add(13,1,-g01); add(16,2,-g01); add(18,2,-g01);
  add(15,0, g02); add(14,1,-g02); add(17,2,-g02); add(19,2,-g02);
  add(14,0, g10); add(15,1, g10); add(10,1, g10); add(11,2, g10);
  add(17,0, g11); add(18,1, g11); add(11,1, g11); add(13,2, g11);
  add(18,0, g12); add(19,1, g12); add(12,1, g12); add(14,2, g12);
  add(10,0,-g20); add(13,0,-g20); add(12,2, g20); add(14,1,-g20);
  add(11,0,-g21); add(16,0,-g21); add(14,2, g21); add(17,1,-g21);
  add(12,0,-g22); add(17,0,-g22); add(15,2, g22); add(18,1,-g22);

  for (int jet=0; jet<J; ++jet) for (int component=0; component<OUT; ++component)
    composed_adjoint[std::size_t(jet)*OUT*count + component + OUT*n] = local[jet*OUT+component];
}

// Adjoint of the analytic product A=e^2 A_raw in actual-derivative Jet
// coordinates.  actual_factor converts between Taylor coefficients and
// physical derivatives, including the diagonal second-derivative factorials.
__global__ void composed_to_raw_adjoint_kernel(const Scalar* composed_adjoint,
                                               const Scalar* envelope, int count,
                                               Scalar* raw_adjoint) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count * OUT) return;
  const int n = index / OUT;
  for (int right=0; right<J; ++right) {
    Scalar value = 0.0f;
    for (int left=0; left<J; ++left) {
      const int target = multi_index(kMx[left]+kMx[right], kMy[left]+kMy[right], kMz[left]+kMz[right]);
      if (target < 0) continue;
      const Scalar coefficient = actual_factor(target) /
          (actual_factor(left) * actual_factor(right));
      value += composed_adjoint[std::size_t(target)*OUT*count + index] *
               envelope[std::size_t(left)*count+n] * coefficient;
    }
    raw_adjoint[std::size_t(right)*OUT*count + index] = value;
  }
}

__global__ void scale_kernel(Scalar* values, int count, Scalar scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) values[i] *= scale;
}

__global__ void loss_adjoint_kernel(const Scalar* potential, const Scalar* gradient,
                                    int potential_count, Scalar weight, Scalar mass,
                                    Scalar stiffness, Scalar* potential_adjoint,
                                    Scalar* gradient_adjoint) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < potential_count) {
    potential_adjoint[i] = -2.0f * weight * stiffness / (mass * mass) * potential[i];
  }
  const int gradient_count = 3 * potential_count;
  for (int j = i; j < gradient_count; j += gridDim.x * blockDim.x)
    gradient_adjoint[j] = 2.0f * weight / mass * gradient[j];
}

__global__ void sine_backward_jet_kernel(const Scalar* pre, const Scalar* output_adjoint,
                                         Scalar* pre_adjoint, int neurons) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= neurons) return;
  Scalar x[J], a[J], d[J]{};
  for (int j=0; j<J; ++j) {
    x[j] = pre[std::size_t(j)*neurons+n];
    a[j] = output_adjoint[std::size_t(j)*neurons+n];
  }
  const Scalar s=sinf(x[0]), c=cosf(x[0]);
  d[0] = a[0]*c - s*(a[1]*x[1]+a[2]*x[2]+a[3]*x[3]);
  d[1] = a[1]*c; d[2]=a[2]*c; d[3]=a[3]*c;
  if constexpr (J >= transfer_fp32::kJet2Components) {
    d[0] += a[4]*(-s*x[4]-c*x[1]*x[1]);
    d[0] += a[5]*(-s*x[5]-c*x[1]*x[2]);
    d[0] += a[6]*(-s*x[6]-c*x[1]*x[3]);
    d[0] += a[7]*(-s*x[7]-c*x[2]*x[2]);
    d[0] += a[8]*(-s*x[8]-c*x[2]*x[3]);
    d[0] += a[9]*(-s*x[9]-c*x[3]*x[3]);
    d[1] += -s*(2.0f*a[4]*x[1]+a[5]*x[2]+a[6]*x[3]);
    d[2] += -s*(a[5]*x[1]+2.0f*a[7]*x[2]+a[8]*x[3]);
    d[3] += -s*(a[6]*x[1]+a[8]*x[2]+2.0f*a[9]*x[3]);
    for (int j=4;j<transfer_fp32::kJet2Components;++j) d[j] += a[j]*c;
  }
  if constexpr (J >= transfer_fp32::kJet3Components) {
    auto third = [&](int out, Scalar sine_sum, Scalar cosine_product,
                     Scalar d1, int i1, Scalar d2, int i2,
                     Scalar d3, int i3, Scalar d4, int i4) {
      const Scalar q = a[out];
      d[0] += q * (-s*x[out] - c*sine_sum + s*cosine_product);
      d[out] += q*c;
      d[i1] += q*d1; d[i2] += q*d2;
      if (i3 >= 0) d[i3] += q*d3;
      if (i4 >= 0) d[i4] += q*d4;
    };
    third(10, 3.0f*x[4]*x[1], x[1]*x[1]*x[1],
          -3.0f*s*x[1],4, -3.0f*s*x[4]-3.0f*c*x[1]*x[1],1, 0,-1,0,-1);
    third(11, x[4]*x[2]+2.0f*x[5]*x[1], x[1]*x[1]*x[2],
          -s*x[2],4, -s*x[4]-c*x[1]*x[1],2,
          -2.0f*s*x[1],5, -2.0f*s*x[5]-2.0f*c*x[1]*x[2],1);
    third(12, x[4]*x[3]+2.0f*x[6]*x[1], x[1]*x[1]*x[3],
          -s*x[3],4, -s*x[4]-c*x[1]*x[1],3,
          -2.0f*s*x[1],6, -2.0f*s*x[6]-2.0f*c*x[1]*x[3],1);
    third(13, x[7]*x[1]+2.0f*x[5]*x[2], x[1]*x[2]*x[2],
          -s*x[1],7, -s*x[7]-c*x[2]*x[2],1,
          -2.0f*s*x[2],5, -2.0f*s*x[5]-2.0f*c*x[1]*x[2],2);
    third(14, x[5]*x[3]+x[6]*x[2]+x[8]*x[1], x[1]*x[2]*x[3],
          -s*x[3],5, -s*x[2],6, -s*x[1],8, 0,-1);
    d[3] += a[14]*(-s*x[5]-c*x[1]*x[2]);
    d[2] += a[14]*(-s*x[6]-c*x[1]*x[3]);
    d[1] += a[14]*(-s*x[8]-c*x[2]*x[3]);
    third(15, x[9]*x[1]+2.0f*x[6]*x[3], x[1]*x[3]*x[3],
          -s*x[1],9, -s*x[9]-c*x[3]*x[3],1,
          -2.0f*s*x[3],6, -2.0f*s*x[6]-2.0f*c*x[1]*x[3],3);
    third(16, 3.0f*x[7]*x[2], x[2]*x[2]*x[2],
          -3.0f*s*x[2],7, -3.0f*s*x[7]-3.0f*c*x[2]*x[2],2, 0,-1,0,-1);
    third(17, x[7]*x[3]+2.0f*x[8]*x[2], x[2]*x[2]*x[3],
          -s*x[3],7, -s*x[7]-c*x[2]*x[2],3,
          -2.0f*s*x[2],8, -2.0f*s*x[8]-2.0f*c*x[2]*x[3],2);
    third(18, x[9]*x[2]+2.0f*x[8]*x[3], x[2]*x[3]*x[3],
          -s*x[2],9, -s*x[9]-c*x[3]*x[3],2,
          -2.0f*s*x[3],8, -2.0f*s*x[8]-2.0f*c*x[2]*x[3],3);
    third(19, 3.0f*x[9]*x[3], x[3]*x[3]*x[3],
          -3.0f*s*x[3],9, -3.0f*s*x[9]-3.0f*c*x[3]*x[3],3, 0,-1,0,-1);
  }
  for (int j=0; j<J; ++j) pre_adjoint[std::size_t(j)*neurons+n] = d[j];
}

__global__ void bias_gradient_kernel(const Scalar* linear_adjoint, int output, int count,
                                     Scalar* bias_gradient) {
  const int neuron = blockIdx.x;
  if (neuron >= output) return;
  Scalar sum = 0.0f;
  for (int n = threadIdx.x; n < count; n += blockDim.x)
    sum += linear_adjoint[neuron + output * n];
  __shared__ Scalar partial[256];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) bias_gradient[neuron] = partial[0];
}

__global__ void adam_kernel(Scalar* parameter, Scalar* first, Scalar* second,
                            const Scalar* gradient, int count, Scalar learning_rate,
                            Scalar beta1_power, Scalar beta2_power) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  constexpr Scalar beta1 = 0.9f, beta2 = 0.999f, epsilon = 1.0e-8f;
  const Scalar g = gradient[i];
  first[i] = beta1 * first[i] + (1.0f - beta1) * g;
  second[i] = beta2 * second[i] + (1.0f - beta2) * g * g;
  const Scalar mhat = first[i] / (1.0f - beta1_power);
  const Scalar vhat = second[i] / (1.0f - beta2_power);
  parameter[i] -= learning_rate * mhat / (sqrtf(vhat) + epsilon);
}

__global__ void gram_error_kernel(const Scalar* gram, int modes, Scalar* maximum, Scalar* minimum_diagonal) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= modes * modes) return;
  const int row = index % modes, column = index / modes;
  const Scalar target = row == column ? 1.0f : 0.0f;
  atomicMax(reinterpret_cast<int*>(maximum), __float_as_int(fabsf(gram[index] - target)));
  if (row == column) atomicMin(reinterpret_cast<int*>(minimum_diagonal), __float_as_int(gram[index]));
}

struct NetworkWorkspace {
  int count;
  DeviceBuffer<Scalar> features;
  DeviceBuffer<Scalar> envelope;
  std::array<DeviceBuffer<Scalar>, L> pre;
  std::array<DeviceBuffer<Scalar>, L> activation;
  DeviceBuffer<Scalar> raw;
  DeviceBuffer<Scalar> composed_potential;
  DeviceBuffer<Scalar> composed_adjoint;
  DeviceBuffer<Scalar> adjoint_a;
  DeviceBuffer<Scalar> adjoint_b;
  DeviceBuffer<Scalar> raw_adjoint;
  DeviceBuffer<Scalar> parameter_gradient;
  explicit NetworkWorkspace(int n) : count(n), features(std::size_t(J)*IN*n), envelope(std::size_t(J)*n),
      raw(std::size_t(J)*OUT*n), composed_potential(std::size_t(J)*OUT*n),
      composed_adjoint(std::size_t(J)*OUT*n),
      adjoint_a(std::size_t(J)*H*n), adjoint_b(std::size_t(J)*H*n),
      raw_adjoint(std::size_t(J)*OUT*n), parameter_gradient(P) {
    for (int layer = 0; layer < L; ++layer) {
      pre[layer].resize(std::size_t(J)*H*n);
      activation[layer].resize(std::size_t(J)*H*n);
    }
  }
};

struct FieldWorkspace {
  int count, modes;
  int potential_rows, gradient_rows;
  DeviceBuffer<Scalar> points, potential, gradient, potential_adjoint, gradient_adjoint;
  DeviceBuffer<Scalar> basis_potential, basis_gradient, coefficient, coefficient_aux, gram;
  explicit FieldWorkspace(int n, int k) : count(n), modes(k), potential_rows(n*OUT), gradient_rows(n*OUT*3),
      points(std::size_t(n)*3), potential(potential_rows), gradient(gradient_rows),
      potential_adjoint(potential_rows), gradient_adjoint(gradient_rows),
      basis_potential(std::size_t(potential_rows)*k), basis_gradient(std::size_t(gradient_rows)*k),
      coefficient(k), coefficient_aux(k), gram(std::size_t(k)*k) {}
};

class NetworkEvaluator {
 public:
  NetworkEvaluator(cublasHandle_t handle, int count) : handle_(handle), ws_(count) {}
  NetworkWorkspace& workspace() { return ws_; }

  void prepare(const Scalar* device_points, Geometry geometry) {
    const int blocks = (ws_.count + 127) / 128;
    prepare_inputs_kernel<<<blocks,128>>>(device_points, ws_.count, geometry, ws_.features.data(), ws_.envelope.data());
    CUDA_CHECK(cudaGetLastError());
  }

  const Scalar* forward(const Scalar* parameter) {
    const Scalar one = 1.0f, zero = 0.0f;
    const Scalar* input = ws_.features.data();
    for (int layer = 0; layer < L; ++layer) {
      const auto layout = transfer_fp32::kLayers[layer];
      CUBLAS_CHECK(cublasSgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
          layout.output, ws_.count, layout.input, &one,
          parameter + layout.weight_offset, layout.output, 0,
          input, layout.input, std::int64_t(layout.input) * ws_.count,
          &zero, ws_.pre[layer].data(), layout.output, std::int64_t(layout.output) * ws_.count, J));
      const int values = J * layout.output * ws_.count;
      affine_kernel<<<(values+255)/256,256>>>(ws_.pre[layer].data(), parameter + layout.bias_offset,
                                             layout.output, ws_.count,
                                             layer == 0 ? kFirstOmega : kHiddenOmega, J);
      sine_jet_kernel<<<(layout.output*ws_.count+255)/256,256>>>(
          ws_.pre[layer].data(), ws_.activation[layer].data(), layout.output * ws_.count);
      input = ws_.activation[layer].data();
    }
    const auto output = transfer_fp32::kLayers[L];
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
        OUT, ws_.count, H, &one, parameter + output.weight_offset, OUT, 0,
        input, H, std::int64_t(H)*ws_.count, &zero, ws_.raw.data(), OUT,
        std::int64_t(OUT)*ws_.count, J));
    affine_kernel<<<(J*OUT*ws_.count+255)/256,256>>>(ws_.raw.data(), parameter + output.bias_offset,
                                                     OUT, ws_.count, 1.0f, J);
    CUDA_CHECK(cudaGetLastError());
    return ws_.raw.data();
  }

  void backward(const Scalar* parameter, Scalar* parameter_gradient) {
    CUDA_CHECK(cudaMemset(parameter_gradient, 0, P * sizeof(Scalar)));
    const Scalar one = 1.0f, zero = 0.0f;
    const auto output = transfer_fp32::kLayers[L];
    accumulate_weight_gradient(ws_.raw_adjoint.data(), ws_.activation[L-1].data(),
                               OUT, H, parameter_gradient + output.weight_offset);
    bias_gradient_kernel<<<OUT,256>>>(ws_.raw_adjoint.data(), OUT, ws_.count,
                                     parameter_gradient + output.bias_offset);
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        H, ws_.count, OUT, &one, parameter + output.weight_offset, OUT, 0,
        ws_.raw_adjoint.data(), OUT, std::int64_t(OUT)*ws_.count, &zero,
        ws_.adjoint_a.data(), H, std::int64_t(H)*ws_.count, J));

    Scalar* activation_adjoint = ws_.adjoint_a.data();
    Scalar* previous_adjoint = ws_.adjoint_b.data();
    for (int layer = L - 1; layer >= 0; --layer) {
      const auto layout = transfer_fp32::kLayers[layer];
      sine_backward_jet_kernel<<<(H*ws_.count+255)/256,256>>>(
          ws_.pre[layer].data(), activation_adjoint, previous_adjoint, H * ws_.count);
      const Scalar omega = layer == 0 ? kFirstOmega : kHiddenOmega;
      scale_kernel<<<(J*H*ws_.count+255)/256,256>>>(previous_adjoint, J*H*ws_.count, omega);
      const Scalar* layer_input = layer == 0 ? ws_.features.data() : ws_.activation[layer-1].data();
      accumulate_weight_gradient(previous_adjoint, layer_input, H, layout.input,
                                 parameter_gradient + layout.weight_offset);
      bias_gradient_kernel<<<H,256>>>(previous_adjoint, H, ws_.count,
                                     parameter_gradient + layout.bias_offset);
      if (layer > 0) {
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            layout.input, ws_.count, H, &one, parameter + layout.weight_offset, H, 0,
            previous_adjoint, H, std::int64_t(H)*ws_.count, &zero,
            activation_adjoint, layout.input, std::int64_t(layout.input)*ws_.count, J));
      }
    }
    CUDA_CHECK(cudaGetLastError());
  }

 private:
  void accumulate_weight_gradient(const Scalar* output_adjoint, const Scalar* input,
                                  int output, int input_count, Scalar* gradient) {
    const Scalar one = 1.0f, zero = 0.0f;
    for (int jet = 0; jet < J; ++jet) {
      const Scalar beta = jet == 0 ? zero : one;
      CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, output, input_count,
          ws_.count, &one, output_adjoint + std::size_t(jet)*output*ws_.count, output,
          input + std::size_t(jet)*input_count*ws_.count, input_count,
          &beta, gradient, output));
    }
  }
  cublasHandle_t handle_;
  NetworkWorkspace ws_;
};

struct Metrics {
  Scalar mean_rayleigh = 0.0f;
  Scalar min_rayleigh = 0.0f;
  Scalar max_rayleigh = 0.0f;
  Scalar orthogonality = 0.0f;
  Scalar minimum_diagonal = 0.0f;
};

class Trainer {
 public:
  Trainer(const Options& options) : options_(options), parameters_(std::size_t(options.modes)*P),
      adam_first_(std::size_t(options.modes)*P), adam_second_(std::size_t(options.modes)*P) {
    CUBLAS_CHECK(cublasCreate(&handle_));
    CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH));
    initialize_parameters();
  }
  ~Trainer() { if (handle_) cublasDestroy(handle_); }

  Metrics step(const std::vector<Scalar>& host_points, const Geometry& geometry,
               Scalar volume, bool update, int optimizer_step) {
    const int count = int(host_points.size() / 3);
    FieldWorkspace field(count, options_.modes);
    NetworkEvaluator network(handle_, count);
    CUDA_CHECK(cudaMemcpy(field.points.data(), host_points.data(), host_points.size()*sizeof(Scalar), cudaMemcpyHostToDevice));
    network.prepare(field.points.data(), geometry);
    const Scalar weight = volume / Scalar(count);
    std::vector<std::pair<Scalar,int>> sorting;
    sorting.reserve(options_.modes);
    for (int mode = 0; mode < options_.modes; ++mode) {
      network.forward(parameters_.data() + std::size_t(mode)*P);
      build_field(network, field);
      const Scalar mass = dot(field.potential.data(), field.potential.data(), field.potential_rows) * weight;
      const Scalar stiffness = dot(field.gradient.data(), field.gradient.data(), field.gradient_rows) * weight;
      if (!std::isfinite(mass) || !std::isfinite(stiffness)) {
        std::vector<Scalar> hp(field.potential_rows), he(std::size_t(J)*count), hr(std::size_t(J)*OUT*count);
        CUDA_CHECK(cudaMemcpy(hp.data(), field.potential.data(), hp.size()*sizeof(Scalar), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(he.data(), network.workspace().envelope.data(), he.size()*sizeof(Scalar), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hr.data(), network.workspace().raw.data(), hr.size()*sizeof(Scalar), cudaMemcpyDeviceToHost));
        int bad = -1;
        for (int i = 0; i < field.potential_rows; ++i) if (!std::isfinite(hp[i])) { bad=i; break; }
        std::ostringstream message;
        message << std::setprecision(9) << "non-finite sort field mode=" << (mode+1)
                << " mass=" << mass << " stiffness=" << stiffness << " bad_field_index=" << bad;
        if (bad >= 0) {
          const int n=bad/OUT, c=bad%OUT;
          message << " point=" << n << " component=" << c << " envelope0=" << he[n]
                  << " raw0=" << hr[c+OUT*n] << " envelope3=" << he[3*count+n]
                  << " raw3=" << hr[3*OUT*count+c+OUT*n]
                  << " xyz=" << host_points[3*n] << "," << host_points[3*n+1] << "," << host_points[3*n+2]
                  << " host_outer=" << outer_level(host_points[3*n],host_points[3*n+1],host_points[3*n+2])
                  << " host_obstacle=" << obstacle_level(host_points[3*n],host_points[3*n+1],host_points[3*n+2],geometry);
          for (int s=0;s<3;++s) message << " sphere" << s << "=" << geometry.sphere[s][0] << ","
              << geometry.sphere[s][1] << "," << geometry.sphere[s][2] << "," << geometry.sphere[s][3];
        }
        throw std::runtime_error(message.str());
      }
      sorting.emplace_back(stiffness / std::max(mass, kMassFloor), mode);
    }
    std::stable_sort(sorting.begin(), sorting.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<Scalar> losses;
    losses.reserve(options_.modes);
    for (int rank = 0; rank < options_.modes; ++rank) {
      const int mode = sorting[rank].second;
      network.forward(parameters_.data() + std::size_t(mode)*P);
      build_field(network, field);
      const Scalar raw_mass = weight * dot(field.potential.data(), field.potential.data(), field.potential_rows);
      if (!(raw_mass > 0.0f) || !std::isfinite(raw_mass)) {
        std::ostringstream message;
        message << std::setprecision(9) << "invalid raw vector mass mode=" << (mode + 1)
                << " causal_rank=" << (rank + 1) << " mass=" << raw_mass
                << " sort_key=" << sorting[rank].first;
        throw std::runtime_error(message.str());
      }
      const Scalar candidate_scale = 1.0f / std::sqrt(raw_mass);
      scale(field.potential.data(), field.potential_rows, candidate_scale);
      scale(field.gradient.data(), field.gradient_rows, candidate_scale);
      for (int pass = 0; pass < 2; ++pass) project_forward(field, rank, weight);
      const Scalar mass = weight * dot(field.potential.data(), field.potential.data(), field.potential_rows);
      const Scalar stiffness = weight * dot(field.gradient.data(), field.gradient.data(), field.gradient_rows);
      if (!(mass > kPeelingMassFloor) || !std::isfinite(stiffness)) {
        std::ostringstream message;
        message << std::setprecision(9) << "unresolved peeled vector mode=" << (mode+1)
                << " causal_rank=" << (rank+1) << " raw_mass=" << raw_mass
                << " peeled_mass=" << mass << " stiffness=" << stiffness
                << " sort_key=" << sorting[rank].first << " volume=" << volume
                << " points=" << count;
        throw std::runtime_error(message.str());
      }
      const Scalar loss = stiffness / mass;
      losses.push_back(loss);
      if (update) {
        loss_adjoint_kernel<<<std::max(1,(field.gradient_rows+255)/256),256>>>(
            field.potential.data(), field.gradient.data(), field.potential_rows, weight, mass,
            stiffness, field.potential_adjoint.data(), field.gradient_adjoint.data());
        for (int pass = 0; pass < 2; ++pass) project_reverse(field, rank, weight);
        scale(field.potential_adjoint.data(), field.potential_rows, candidate_scale);
        scale(field.gradient_adjoint.data(), field.gradient_rows, candidate_scale);
        if constexpr (kVorticityRayleigh) {
          vorticity_to_composed_adjoint_kernel<<<(count+127)/128,128>>>(
              field.potential_adjoint.data(), field.gradient_adjoint.data(), count,
              network.workspace().composed_adjoint.data());
          composed_to_raw_adjoint_kernel<<<(field.potential_rows+255)/256,256>>>(
              network.workspace().composed_adjoint.data(), network.workspace().envelope.data(),
              count, network.workspace().raw_adjoint.data());
        } else if constexpr (kVelocityRayleigh) {
          velocity_to_composed_adjoint_kernel<<<(count+127)/128,128>>>(
              field.potential_adjoint.data(), field.gradient_adjoint.data(), count,
              network.workspace().composed_adjoint.data());
          composed_to_raw_adjoint_kernel<<<(field.potential_rows+255)/256,256>>>(
              network.workspace().composed_adjoint.data(), network.workspace().envelope.data(),
              count, network.workspace().raw_adjoint.data());
        } else {
          field_to_raw_adjoint_kernel<<<(field.potential_rows+255)/256,256>>>(
              field.potential_adjoint.data(), field.gradient_adjoint.data(), network.workspace().envelope.data(),
              count, network.workspace().raw_adjoint.data());
        }
        Scalar* gradient = network.workspace().parameter_gradient.data();
        network.backward(parameters_.data() + std::size_t(mode)*P, gradient);
        clip_and_update(mode, gradient, optimizer_step);
      }
      const Scalar normalizer = 1.0f / std::sqrt(mass);
      scale(field.potential.data(), field.potential_rows, normalizer);
      scale(field.gradient.data(), field.gradient_rows, normalizer);
      CUDA_CHECK(cudaMemcpy(field.basis_potential.data() + std::size_t(rank)*field.potential_rows,
                            field.potential.data(), field.potential_rows*sizeof(Scalar), cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(field.basis_gradient.data() + std::size_t(rank)*field.gradient_rows,
                            field.gradient.data(), field.gradient_rows*sizeof(Scalar), cudaMemcpyDeviceToDevice));
    }
    Metrics metrics;
    metrics.mean_rayleigh = std::accumulate(losses.begin(), losses.end(), 0.0f) / Scalar(losses.size());
    metrics.min_rayleigh = *std::min_element(losses.begin(), losses.end());
    metrics.max_rayleigh = *std::max_element(losses.begin(), losses.end());
    compute_gram_metrics(field, weight, metrics);
    CUDA_CHECK(cudaDeviceSynchronize());
    return metrics;
  }

  void save(const fs::path& path, int epoch) {
    struct Header {
      char magic[8]; std::uint32_t version, epoch, modes, parameters_per_mode;
      std::uint32_t raw_inputs, vector_components, jet_components, scalar_bytes;
    } header{{'T','V','F','P','3','2','V','2'},2U,std::uint32_t(epoch),std::uint32_t(options_.modes),
             std::uint32_t(P),std::uint32_t(IN),std::uint32_t(OUT),std::uint32_t(J),4U};
    const std::size_t total = std::size_t(options_.modes) * P;
    std::vector<Scalar> parameter(total), first(total), second(total);
    CUDA_CHECK(cudaMemcpy(parameter.data(), parameters_.data(), total*sizeof(Scalar), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(first.data(), adam_first_.data(), total*sizeof(Scalar), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(second.data(), adam_second_.data(), total*sizeof(Scalar), cudaMemcpyDeviceToHost));
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    stream.write(reinterpret_cast<const char*>(parameter.data()), std::streamsize(total*sizeof(Scalar)));
    stream.write(reinterpret_cast<const char*>(first.data()), std::streamsize(total*sizeof(Scalar)));
    stream.write(reinterpret_cast<const char*>(second.data()), std::streamsize(total*sizeof(Scalar)));
    stream.close();
    if (!stream) throw std::runtime_error("checkpoint write failed");
    if (fs::exists(path)) throw std::runtime_error("refusing to overwrite immutable checkpoint: " + path.string());
    fs::rename(temporary, path);
  }

  int load(const fs::path& path) {
    struct Header {
      char magic[8]; std::uint32_t version, epoch, modes, parameters_per_mode;
      std::uint32_t raw_inputs, vector_components, jet_components, scalar_bytes;
    } header{};
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (std::memcmp(header.magic, "TVFP32V2", 8) || header.version != 2 ||
        int(header.modes) != options_.modes || int(header.parameters_per_mode) != P ||
        int(header.raw_inputs) != IN || int(header.vector_components) != OUT ||
        (int(header.jet_components) != transfer_fp32::kDifferentiatedJetComponents &&
         int(header.jet_components) != transfer_fp32::kJet2Components &&
         int(header.jet_components) != transfer_fp32::kJet3Components) ||
        header.scalar_bytes != 4)
      throw std::runtime_error("checkpoint violates strict vector FP32 contract");
    const std::size_t total = std::size_t(options_.modes)*P;
    std::vector<Scalar> parameter(total), first(total), second(total);
    stream.read(reinterpret_cast<char*>(parameter.data()), std::streamsize(total*sizeof(Scalar)));
    stream.read(reinterpret_cast<char*>(first.data()), std::streamsize(total*sizeof(Scalar)));
    stream.read(reinterpret_cast<char*>(second.data()), std::streamsize(total*sizeof(Scalar)));
    if (!stream) throw std::runtime_error("truncated checkpoint");
    CUDA_CHECK(cudaMemcpy(parameters_.data(), parameter.data(), total*sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(adam_first_.data(), first.data(), total*sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(adam_second_.data(), second.data(), total*sizeof(Scalar), cudaMemcpyHostToDevice));
    return int(header.epoch);
  }

 private:
  void initialize_parameters() {
    const std::size_t total = std::size_t(options_.modes)*P;
    std::vector<Scalar> host(total, 0.0f);
    for (int mode = 0; mode < options_.modes; ++mode) {
      Pcg64 rng(options_.seed + 7919ULL * std::uint64_t(mode + 1));
      Scalar* base = host.data() + std::size_t(mode)*P;
      for (int layer = 0; layer <= L; ++layer) {
        const auto layout = transfer_fp32::kLayers[layer];
        const Scalar bound = layer == 0 ? 1.0f / Scalar(layout.input)
                                        : std::sqrt(6.0f / Scalar(layout.input)) / kHiddenOmega;
        for (int i = 0; i < layout.input * layout.output; ++i)
          base[layout.weight_offset + i] = (2.0f*rng.uniform()-1.0f)*bound;
        for (int i = 0; i < layout.output; ++i)
          base[layout.bias_offset + i] = (2.0f*rng.uniform()-1.0f)*bound;
      }
    }
    CUDA_CHECK(cudaMemcpy(parameters_.data(), host.data(), total*sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(adam_first_.data(), 0, total*sizeof(Scalar)));
    CUDA_CHECK(cudaMemset(adam_second_.data(), 0, total*sizeof(Scalar)));
  }

  void build_field(NetworkEvaluator& network, FieldWorkspace& field) {
    build_vector_field_kernel<<<(field.potential_rows+255)/256,256>>>(
        network.workspace().raw.data(), network.workspace().envelope.data(), field.count,
        network.workspace().composed_potential.data(), field.potential.data(), field.gradient.data());
    if constexpr (kVorticityRayleigh) {
      build_vorticity_field_kernel<<<(field.potential_rows+255)/256,256>>>(
          network.workspace().composed_potential.data(), field.count,
          field.potential.data(), field.gradient.data());
    } else if constexpr (kVelocityRayleigh) {
      build_velocity_field_kernel<<<(field.potential_rows+255)/256,256>>>(
          network.workspace().composed_potential.data(), field.count,
          field.potential.data(), field.gradient.data());
    }
    CUDA_CHECK(cudaGetLastError());
  }
  Scalar dot(const Scalar* a, const Scalar* b, int count) {
    Scalar value = 0.0f; CUBLAS_CHECK(cublasSdot(handle_, count, a, 1, b, 1, &value)); return value;
  }
  void scale(Scalar* values, int count, Scalar factor) {
    scale_kernel<<<(count+255)/256,256>>>(values, count, factor);
  }
  void project_forward(FieldWorkspace& f, int rank, Scalar weight) {
    if (!rank) return;
    const Scalar zero=0.0f, minus=-1.0f, one=1.0f;
    CUBLAS_CHECK(cublasSgemv(handle_, CUBLAS_OP_T, f.potential_rows, rank, &weight,
        f.basis_potential.data(), f.potential_rows, f.potential.data(), 1, &zero, f.coefficient.data(), 1));
    CUBLAS_CHECK(cublasSgemv(handle_, CUBLAS_OP_N, f.potential_rows, rank, &minus,
        f.basis_potential.data(), f.potential_rows, f.coefficient.data(), 1, &one, f.potential.data(), 1));
    CUBLAS_CHECK(cublasSgemv(handle_, CUBLAS_OP_N, f.gradient_rows, rank, &minus,
        f.basis_gradient.data(), f.gradient_rows, f.coefficient.data(), 1, &one, f.gradient.data(), 1));
  }
  void project_reverse(FieldWorkspace& f, int rank, Scalar weight) {
    if (!rank) return;
    const Scalar zero=0.0f, minus=-1.0f, one=1.0f;
    CUBLAS_CHECK(cublasSgemv(handle_, CUBLAS_OP_T, f.potential_rows, rank, &minus,
        f.basis_potential.data(), f.potential_rows, f.potential_adjoint.data(), 1, &zero, f.coefficient_aux.data(), 1));
    CUBLAS_CHECK(cublasSgemv(handle_, CUBLAS_OP_T, f.gradient_rows, rank, &minus,
        f.basis_gradient.data(), f.gradient_rows, f.gradient_adjoint.data(), 1, &one, f.coefficient_aux.data(), 1));
    CUBLAS_CHECK(cublasSgemv(handle_, CUBLAS_OP_N, f.potential_rows, rank, &weight,
        f.basis_potential.data(), f.potential_rows, f.coefficient_aux.data(), 1, &one, f.potential_adjoint.data(), 1));
  }
  void clip_and_update(int mode, Scalar* gradient, int step) {
    const Scalar norm = std::sqrt(dot(gradient, gradient, P));
    if (!std::isfinite(norm)) throw std::runtime_error("non-finite FP32 parameter gradient");
    if (norm > kGradClip) scale(gradient, P, kGradClip / norm);
    const Scalar beta1_power = std::pow(0.9f, Scalar(step));
    const Scalar beta2_power = std::pow(0.999f, Scalar(step));
    adam_kernel<<<(P+255)/256,256>>>(parameters_.data()+std::size_t(mode)*P,
        adam_first_.data()+std::size_t(mode)*P, adam_second_.data()+std::size_t(mode)*P,
        gradient, P, current_learning_rate_, beta1_power, beta2_power);
  }
  void compute_gram_metrics(FieldWorkspace& f, Scalar weight, Metrics& metrics) {
    const Scalar zero=0.0f;
    CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, options_.modes, options_.modes,
        f.potential_rows, &weight, f.basis_potential.data(), f.potential_rows,
        f.basis_potential.data(), f.potential_rows, &zero, f.gram.data(), options_.modes));
    DeviceBuffer<Scalar> maximum(1), minimum(1);
    Scalar hmax=0.0f, hmin=1.0e30f;
    CUDA_CHECK(cudaMemcpy(maximum.data(), &hmax, sizeof(Scalar), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(minimum.data(), &hmin, sizeof(Scalar), cudaMemcpyHostToDevice));
    gram_error_kernel<<<(options_.modes*options_.modes+255)/256,256>>>(f.gram.data(), options_.modes,
                                                                      maximum.data(), minimum.data());
    CUDA_CHECK(cudaMemcpy(&metrics.orthogonality, maximum.data(), sizeof(Scalar), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&metrics.minimum_diagonal, minimum.data(), sizeof(Scalar), cudaMemcpyDeviceToHost));
  }
 public:
  void set_learning_rate(Scalar value) { current_learning_rate_ = value; }
  Scalar* parameters_data() { return parameters_.data(); }
  cublasHandle_t cublas_handle() const { return handle_; }
 private:
  Options options_;
  cublasHandle_t handle_ = nullptr;
  DeviceBuffer<Scalar> parameters_, adam_first_, adam_second_;
  Scalar current_learning_rate_ = kBaseLearningRate;
};

Options parse_options(int argc, char** argv) {
  Options o;
  auto value = [&](int& i) -> std::string { if (++i >= argc) throw std::runtime_error("missing argument value"); return argv[i]; };
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--modes") o.modes = std::stoi(value(i));
    else if (key == "--epochs") o.epochs = std::stoi(value(i));
    else if (key == "--points") o.points = std::stoi(value(i));
    else if (key == "--validation-points") o.validation_points = std::stoi(value(i));
    else if (key == "--validation-geometries") o.validation_geometries = std::stoi(value(i));
    else if (key == "--validate-every") o.validate_every = std::stoi(value(i));
    else if (key == "--checkpoint-every") o.checkpoint_every = std::stoi(value(i));
    else if (key == "--stop-after") o.stop_after = std::stoi(value(i));
    else if (key == "--lr-schedule-start") o.lr_schedule_start = std::stoi(value(i));
    else if (key == "--lr-schedule-end") o.lr_schedule_end = std::stoi(value(i));
    else if (key == "--seed") o.seed = std::stoull(value(i));
    else if (key == "--fixed-geometry-token") {
      o.fixed_geometry_token = std::stoull(value(i));
      o.fixed_geometry = true;
    }
    else if (key == "--fixed-geometry-file") {
      o.fixed_geometry_file = value(i);
      o.fixed_geometry = true;
    }
    else if (key == "--geometry-dataset-file") o.geometry_dataset_file = value(i);
    else if (key == "--result") o.result = value(i);
    else if (key == "--resume") o.resume = value(i);
    else if (key == "--self-test") o.self_test = true;
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (o.self_test) {
    o.modes=4; o.epochs=1; o.points=256; o.validation_points=128;
    o.validation_geometries=1; o.validate_every=1; o.checkpoint_every=1;
  }
  if (o.modes < 1 || o.points < o.modes || o.validation_points < o.modes)
    throw std::runtime_error("quadrature must contain at least one point per vector mode");
  if (o.fixed_geometry && !o.geometry_dataset_file.empty())
    throw std::runtime_error("fixed geometry and changing geometry dataset are mutually exclusive");
  const int schedule_end = o.lr_schedule_end > 0 ? o.lr_schedule_end : o.epochs;
  if (o.lr_schedule_start < 1 || schedule_end < o.lr_schedule_start || schedule_end > o.epochs)
    throw std::runtime_error("invalid learning-rate schedule interval");
  return o;
}

Scalar learning_rate(const Options& o, int epoch) {
  const int schedule_end = o.lr_schedule_end > 0 ? o.lr_schedule_end : o.epochs;
  if (schedule_end <= o.lr_schedule_start) return kMinLearningRate;
  constexpr Scalar pi=3.14159265358979323846f;
  const Scalar unclamped = Scalar(epoch-o.lr_schedule_start)/Scalar(schedule_end-o.lr_schedule_start);
  const Scalar progress = std::max(0.0f, std::min(1.0f, unclamped));
  return kMinLearningRate+0.5f*(kBaseLearningRate-kMinLearningRate)*(1.0f+std::cos(pi*progress));
}

void write_protocol(const Options& o) {
  fs::create_directories(o.result / "data");
  fs::create_directories(o.result / "checkpoints");
  std::ofstream s(o.result / "protocol.json", std::ios::trunc);
  s << "{\n"
    << "  \"implementation\": \"workspace-native CUDA\",\n"
    << "  \"modes\": " << o.modes << ",\n"
    << "  \"raw_input_features\": 15,\n"
    << "  \"raw_input\": \"centered xyz plus normalized three-sphere xyzr\",\n"
    << "  \"vector_output\": [\"A12\", \"A13\", \"A23\"],\n"
    << "  \"scalar\": \"FP32\",\n"
    << "  \"scalar_bytes\": 4,\n"
    << "  \"tf32\": false,\n"
    << "  \"pytorch\": false,\n"
    << "  \"training_grid\": null,\n"
    << "  \"geometry_schedule\": \""
    << (o.fixed_geometry ? "fixed" : (!o.geometry_dataset_file.empty()
        ? "pre-generated changing xyzr: one training geometry per epoch plus held-out validation geometries"
        : "one deterministic geometry per epoch")) << "\",\n"
    << "  \"fixed_geometry_token\": " << (o.fixed_geometry ? o.fixed_geometry_token : 0ULL) << ",\n"
    << "  \"explicit_fixed_geometry_file\": " << (o.fixed_geometry_file.empty()?"false":"true") << ",\n"
    << "  \"explicit_geometry_dataset_file\": " << (o.geometry_dataset_file.empty()?"false":"true") << ",\n"
    << "  \"geometry_dataset_file\": \"" << o.geometry_dataset_file.generic_string() << "\",\n"
    << "  \"training_geometries\": " << (o.fixed_geometry ? 1 : o.epochs) << ",\n"
    << "  \"held_out_validation_geometries\": " << o.validation_geometries << ",\n"
    << "  \"geometry_parameters_change_each_epoch\": " << (o.fixed_geometry?"false":"true") << ",\n"
    << "  \"learning_rate_schedule\": \"cosine restart\",\n"
    << "  \"learning_rate_schedule_start_epoch\": " << o.lr_schedule_start << ",\n"
    << "  \"learning_rate_schedule_end_epoch\": " << (o.lr_schedule_end > 0 ? o.lr_schedule_end : o.epochs) << ",\n"
    << "  \"derivative_objective\": \""
    << objective_name() << "\",\n"
    << "  \"rayleigh_numerator\": \"" << rayleigh_numerator() << "\",\n"
    << "  \"rayleigh_denominator\": \"" << rayleigh_denominator() << "\",\n"
    << "  \"potential_derivative_order\": " << (kVorticityRayleigh ? 3 : (kVelocityRayleigh ? 2 : 1)) << ",\n"
    << "  \"velocity_derivative_order\": " << (kVorticityRayleigh ? 2 : (kVelocityRayleigh ? 1 : 0)) << ",\n"
    << "  \"vorticity_derivative_order\": " << (kVorticityRayleigh ? 1 : 0) << ",\n"
    << "  \"active_jet_components\": " << J << ",\n"
    << "  \"jet3\": " << (kVorticityRayleigh ? "true" : "false") << ",\n"
    << "  \"envelope\": \"squared transfer envelope; second-order boundary zero\",\n"
    << "  \"sorting\": \"stable raw "
    << (kVorticityRayleigh ? "vorticity-gradient" : (kVelocityRayleigh ? "velocity-gradient" : "potential-Dirichlet")) << " Rayleigh\",\n"
    << "  \"peeling\": \"two-pass "
    << (kVorticityRayleigh ? "vorticity-mass" : (kVelocityRayleigh ? "velocity-mass" : "potential-mass")) << " MGS\",\n"
    << "  \"peeling_residual_floor\": " << kPeelingMassFloor << ",\n"
    << "  \"complete_tensor_modes\": " << o.modes << ",\n"
    << "  \"independent_coefficients\": " << o.modes << ",\n"
    << "  \"one_q_per_complete_A\": true\n"
    << "}\n";
}

void append_history(const fs::path& path, int epoch, const Metrics& train, const Metrics& validation,
                    bool has_validation, Scalar lr, Scalar seconds) {
  std::ofstream s(path, std::ios::app);
  s << std::setprecision(9) << "{\"epoch\":" << epoch << ",\"seconds\":" << seconds
    << ",\"learning_rate\":" << lr << ",\"train_mean_rayleigh\":" << train.mean_rayleigh
    << ",\"train_min_rayleigh\":" << train.min_rayleigh << ",\"train_max_rayleigh\":" << train.max_rayleigh
    << ",\"train_vector_orthogonality_max_abs\":" << train.orthogonality;
  if (has_validation) s << ",\"validation_mean_rayleigh\":" << validation.mean_rayleigh
                        << ",\"validation_vector_orthogonality_max_abs\":" << validation.orthogonality;
  s << "}\n";
}

Scalar read_active_seconds(const fs::path& path) {
  Scalar value = 0.0f;
  std::ifstream stream(path);
  if (stream) stream >> value;
  return value;
}

void write_active_seconds(const fs::path& path, Scalar value) {
  const fs::path temporary = path.string() + ".tmp";
  std::ofstream stream(temporary, std::ios::trunc);
  stream << std::setprecision(9) << value << "\n";
  stream.close();
  if (fs::exists(path)) fs::remove(path);
  fs::rename(temporary, path);
}

}  // namespace

#ifndef TRANSFER_VECTOR_LIBRARY
int main(int argc, char** argv) {
  try {
    Options options = parse_options(argc, argv);
    int device = 0; CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp properties{}; CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major < 8) throw std::runtime_error("strict FP32 trainer requires an Ampere-or-newer CUDA GPU");
    write_protocol(options);
    const std::vector<Geometry> geometry_dataset = options.geometry_dataset_file.empty()
        ? std::vector<Geometry>{}
        : geometry_dataset_from_file(options.geometry_dataset_file,
                                     options.epochs + options.validation_geometries);
    Trainer trainer(options);
    int start_epoch = 1;
    if (!options.resume.empty()) start_epoch = trainer.load(options.resume) + 1;
    std::ofstream log(options.result / "process.stdout.log", std::ios::app);
    log << "start gpu=" << properties.name << " modes=" << options.modes << " points=" << options.points
        << " epochs=" << options.epochs
        << " lr_schedule=" << options.lr_schedule_start << "-"
        << (options.lr_schedule_end > 0 ? options.lr_schedule_end : options.epochs)
        << " raw_inputs=15 vector_outputs=3 derivative_order=" << (kVorticityRayleigh ? 3 : (kVelocityRayleigh ? 2 : 1))
        << " active_jet=" << J
        << " objective=" << objective_slug()
        << " scalar=FP32 tf32=off\n";
    log.flush();
    Metrics final_validation{};
    const int end_epoch = options.stop_after > 0 ? std::min(options.stop_after, options.epochs) : options.epochs;
    const fs::path active_time_path = options.result / "data/active_training_seconds.txt";
    Scalar active_training_seconds = read_active_seconds(active_time_path);
    int completed_epoch = start_epoch - 1;
    for (int epoch = start_epoch; epoch <= end_epoch; ++epoch) {
      const auto started = std::chrono::steady_clock::now();
      const Scalar lr = learning_rate(options, epoch);
      trainer.set_learning_rate(lr);
      const std::uint64_t training_token = options.fixed_geometry
          ? options.fixed_geometry_token : std::uint64_t(epoch);
      const Geometry geometry = !geometry_dataset.empty()
          ? geometry_dataset.at(std::size_t(epoch - 1))
          : (options.fixed_geometry_file.empty()
              ? geometry_from_token(options.seed,training_token)
              : geometry_from_file(options.fixed_geometry_file));
      const auto points = sample_fluid_points(options.points, geometry, options.seed + 100000ULL + epoch);
      const Scalar volume = approximate_volume(geometry, options.seed + 3000003ULL + epoch,
                                               options.self_test ? 4096 : 65536);
      const Metrics train = trainer.step(points, geometry, volume, true, epoch);
      const bool validate = epoch % options.validate_every == 0 || epoch == options.epochs;
      Metrics validation{};
      if (validate) {
        Metrics sum{};
        sum.min_rayleigh = 1.0e30f;
        for (int index = 0; index < options.validation_geometries; ++index) {
          const std::uint64_t token = options.fixed_geometry
              ? options.fixed_geometry_token : 10000000ULL + std::uint64_t(index);
          const Geometry vg = !geometry_dataset.empty()
              ? geometry_dataset.at(std::size_t(options.epochs + index))
              : (options.fixed_geometry_file.empty()
                  ? geometry_from_token(options.seed,token)
                  : geometry_from_file(options.fixed_geometry_file));
          const auto vp = sample_fluid_points(options.validation_points, vg, options.seed + 9000000ULL + index);
          const Scalar vv = approximate_volume(vg, options.seed + 13000000ULL + index,
                                               options.self_test ? 4096 : 65536);
          const Metrics m = trainer.step(vp, vg, vv, false, epoch);
          sum.mean_rayleigh += m.mean_rayleigh;
          sum.min_rayleigh = std::min(sum.min_rayleigh, m.min_rayleigh);
          sum.max_rayleigh = std::max(sum.max_rayleigh, m.max_rayleigh);
          sum.orthogonality = std::max(sum.orthogonality, m.orthogonality);
          sum.minimum_diagonal = index ? std::min(sum.minimum_diagonal, m.minimum_diagonal) : m.minimum_diagonal;
        }
        validation = sum;
        validation.mean_rayleigh /= Scalar(options.validation_geometries);
        final_validation = validation;
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      const Scalar seconds = std::chrono::duration<Scalar>(std::chrono::steady_clock::now()-started).count();
      active_training_seconds += seconds;
      write_active_seconds(active_time_path, active_training_seconds);
      completed_epoch = epoch;
      append_history(options.result / "history.jsonl", epoch, train, validation, validate, lr, seconds);
      log << std::setprecision(7) << "epoch=" << epoch << "/" << options.epochs << " sec=" << seconds
          << " loss=" << train.mean_rayleigh << " orth=" << train.orthogonality;
      if (validate) log << " val=" << validation.mean_rayleigh << " val_orth=" << validation.orthogonality;
      log << " lr=" << lr << "\n"; log.flush();
      if (epoch % options.checkpoint_every == 0 || epoch == end_epoch) {
        const fs::path immutable = options.result / "checkpoints" /
            (std::string("epoch_") + (epoch < 1000 ? (epoch < 100 ? (epoch < 10 ? "000" : "00") : "0") : "") +
             std::to_string(epoch) + ".bin");
        trainer.save(immutable, epoch);
        fs::copy_file(immutable, options.result / "checkpoints/latest.bin", fs::copy_options::overwrite_existing);
      }
    }
    std::ofstream completion(options.result / "completion.json", std::ios::trunc);
    const bool complete = completed_epoch == options.epochs;
    completion << std::setprecision(9) << "{\n  \"complete\": " << (complete ? "true" : "false")
               << ",\n  \"status\": \"" << (complete ? "complete" : "paused_for_epoch_0050_review")
               << "\",\n  \"epochs_requested\": " << options.epochs
               << ",\n  \"epochs_completed\": " << completed_epoch << ",\n  \"modes\": " << options.modes
               << ",\n  \"points_per_step\": " << options.points
               << ",\n  \"validation_points\": " << options.validation_points
               << ",\n  \"geometry_schedule\": \""
               << (options.fixed_geometry ? "fixed" : (!options.geometry_dataset_file.empty()
                   ? "pre-generated changing xyzr: one training geometry per epoch plus held-out validation geometries"
                   : "one deterministic geometry per epoch")) << "\""
               << ",\n  \"fixed_geometry_token\": " << (options.fixed_geometry ? options.fixed_geometry_token : 0ULL)
               << ",\n  \"explicit_fixed_geometry_file\": " << (options.fixed_geometry_file.empty()?"false":"true")
               << ",\n  \"explicit_geometry_dataset_file\": " << (options.geometry_dataset_file.empty()?"false":"true")
               << ",\n  \"geometry_dataset_file\": \"" << options.geometry_dataset_file.generic_string() << "\""
               << ",\n  \"training_geometries\": " << (options.fixed_geometry ? 1 : options.epochs)
               << ",\n  \"held_out_validation_geometries\": " << options.validation_geometries
               << ",\n  \"geometry_parameters_change_each_epoch\": " << (options.fixed_geometry?"false":"true")
               << ",\n  \"learning_rate_schedule\": \"cosine restart\""
               << ",\n  \"learning_rate_schedule_start_epoch\": " << options.lr_schedule_start
               << ",\n  \"learning_rate_schedule_end_epoch\": " << (options.lr_schedule_end > 0 ? options.lr_schedule_end : options.epochs)
               << ",\n  \"raw_input_features\": 15,\n  \"vector_components\": 3,\n"
               << "  \"derivative_objective\": \""
               << objective_name() << "\",\n"
               << "  \"rayleigh_numerator\": \"" << rayleigh_numerator() << "\",\n"
               << "  \"rayleigh_denominator\": \"" << rayleigh_denominator() << "\",\n"
               << "  \"potential_derivative_order\": " << (kVorticityRayleigh ? 3 : (kVelocityRayleigh ? 2 : 1))
               << ",\n  \"velocity_derivative_order\": " << (kVorticityRayleigh ? 2 : (kVelocityRayleigh ? 1 : 0))
               << ",\n  \"vorticity_derivative_order\": " << (kVorticityRayleigh ? 1 : 0)
               << ",\n  \"active_jet_components\": " << J
               << ",\n  \"jet3\": " << (kVorticityRayleigh ? "true" : "false") << ",\n"
               << "  \"complete_tensor_modes\": " << options.modes
               << ",\n  \"independent_coefficients\": " << options.modes
               << ",\n  \"one_q_per_complete_A\": true,\n"
               << "  \"scalar\": \"FP32\",\n  \"tf32\": false,\n"
               << "  \"active_training_seconds\": " << active_training_seconds
               << ",\n  \"pause_and_review_seconds_included\": false,\n"
               << "  \"validation_mean_rayleigh\": " << final_validation.mean_rayleigh << "\n}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << std::endl;
    return 1;
  }
}
#endif
