#pragma once

#include <cstdint>
#include <type_traits>

namespace transfer_fp32 {

using Scalar = float;

inline constexpr int kSpatialDimensions = 3;
inline constexpr int kSphereCount = 3;
inline constexpr int kGeometryParameters = 4 * kSphereCount;
inline constexpr int kRawInputs = kSpatialDimensions + kGeometryParameters;
inline constexpr int kVectorComponents = 3;
inline constexpr int kJet3Components = 20;
inline constexpr int kJet2Components = 10;
inline constexpr int kDifferentiatedJetComponents = 4;
inline constexpr int kHiddenWidth = 32;
inline constexpr int kHiddenLayers = 4;

static_assert(TRANSFER_STRICT_FP32 == 1, "Strict FP32 build flag is mandatory");
static_assert(std::is_same_v<Scalar, float>, "The training scalar must be float");
static_assert(sizeof(Scalar) == 4, "The training scalar must be IEEE-style FP32");
static_assert(kRawInputs == 15, "Transfer contract is raw xyz plus 12 geometry values");
static_assert(kVectorComponents == 3, "Each mode must output A12, A13 and A23 together");
static_assert(kJet3Components == 20, "A 3D symmetric Jet3 has 20 components");
static_assert(kJet2Components == 10, "A 3D symmetric Jet2 has 10 components");

struct Geometry {
  Scalar sphere[kSphereCount][4];
};

struct LayerLayout {
  int input;
  int output;
  std::uint64_t weight_offset;
  std::uint64_t bias_offset;
};

inline constexpr LayerLayout kLayers[kHiddenLayers + 1] = {
    {kRawInputs, kHiddenWidth, 0, kRawInputs * kHiddenWidth},
    {kHiddenWidth, kHiddenWidth, 512, 1536},
    {kHiddenWidth, kHiddenWidth, 1568, 2592},
    {kHiddenWidth, kHiddenWidth, 2624, 3648},
    {kHiddenWidth, kVectorComponents, 3680, 3776},
};

inline constexpr int kParametersPerMode = 3779;
static_assert(kLayers[4].bias_offset + kVectorComponents == kParametersPerMode);

}  // namespace transfer_fp32
