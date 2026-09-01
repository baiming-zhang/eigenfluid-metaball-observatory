cmake_minimum_required(VERSION 3.20)
project(transfer_vector_fp32 LANGUAGES CXX CUDA)

find_package(CUDAToolkit 11.8 REQUIRED)

add_executable(transfer_vector_fp32 src/transfer_vector_fp32.cu)
target_include_directories(transfer_vector_fp32 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(transfer_vector_fp32 PRIVATE CUDA::cudart CUDA::cublas)
target_compile_features(transfer_vector_fp32 PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(transfer_vector_fp32 PRIVATE TRANSFER_STRICT_FP32=1)
set_target_properties(transfer_vector_fp32 PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION OFF
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)
target_compile_options(transfer_vector_fp32 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ftz=false --prec-div=true --prec-sqrt=true --fmad=true>
  $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:strict>
)

# Independent velocity-Rayleigh trainer.  It shares the exact architecture,
# geometry and strict-FP32 contract with transfer_vector_fp32, but compiles the
# analytic Jet2 velocity objective into a distinct executable so an active
# potential-Rayleigh run can never be overwritten or silently changed.
add_executable(transfer_vector_velocity_rayleigh_fp32 src/transfer_vector_fp32.cu)
target_include_directories(transfer_vector_velocity_rayleigh_fp32 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(transfer_vector_velocity_rayleigh_fp32 PRIVATE CUDA::cudart CUDA::cublas)
target_compile_features(transfer_vector_velocity_rayleigh_fp32 PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(transfer_vector_velocity_rayleigh_fp32 PRIVATE
  TRANSFER_STRICT_FP32=1
  TRANSFER_VELOCITY_RAYLEIGH=1
)
set_target_properties(transfer_vector_velocity_rayleigh_fp32 PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION OFF
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)
target_compile_options(transfer_vector_velocity_rayleigh_fp32 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ftz=false --prec-div=true --prec-sqrt=true --fmad=true>
  $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:strict>
)

# Independent vorticity-Rayleigh trainer. It uses the same complete tensor
# modes and geometry transfer schedule, but evaluates omega=curl(div A) and
# grad(omega) with the full analytic Jet3.
add_executable(transfer_vector_vorticity_rayleigh_fp32 src/transfer_vector_fp32.cu)
target_include_directories(transfer_vector_vorticity_rayleigh_fp32 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(transfer_vector_vorticity_rayleigh_fp32 PRIVATE CUDA::cudart CUDA::cublas)
target_compile_features(transfer_vector_vorticity_rayleigh_fp32 PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(transfer_vector_vorticity_rayleigh_fp32 PRIVATE
  TRANSFER_STRICT_FP32=1
  TRANSFER_VORTICITY_RAYLEIGH=1
)
set_target_properties(transfer_vector_vorticity_rayleigh_fp32 PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION OFF
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)
target_compile_options(transfer_vector_vorticity_rayleigh_fp32 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ftz=false --prec-div=true --prec-sqrt=true --fmad=true>
  $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:strict>
)

# Deterministic, non-trained complete-vector Hodge-Galerkin baseline.  Every
# trial is already a full A12/A13/A23 tensor potential; no scalar-times-three
# polarization lifting is used.
add_executable(hodge_vector_analytic_k1024 src/hodge_vector_analytic_k1024.cu)
target_include_directories(hodge_vector_analytic_k1024 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hodge_vector_analytic_k1024 PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
target_compile_features(hodge_vector_analytic_k1024 PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(hodge_vector_analytic_k1024 PRIVATE TRANSFER_STRICT_FP32=1)
set_target_properties(hodge_vector_analytic_k1024 PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION OFF
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)
target_compile_options(hodge_vector_analytic_k1024 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ftz=false --prec-div=true --prec-sqrt=true --fmad=true>
  $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:strict>
)

# Oversampled analytic complete-vector Hodge-Galerkin baseline: assemble 3072
# geometry-dependent trial coordinates globally and retain the lowest 1024
# complete eigenmodes, each with one external q coefficient.
add_executable(hodge_vector_analytic_rr3072_k1024 src/hodge_vector_analytic_k1024.cu)
target_include_directories(hodge_vector_analytic_rr3072_k1024 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hodge_vector_analytic_rr3072_k1024 PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
target_compile_features(hodge_vector_analytic_rr3072_k1024 PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(hodge_vector_analytic_rr3072_k1024 PRIVATE
  TRANSFER_STRICT_FP32=1
  HODGE_ANALYTIC_TRIALS=3072
)
set_target_properties(hodge_vector_analytic_rr3072_k1024 PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION OFF
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)
target_compile_options(hodge_vector_analytic_rr3072_k1024 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ftz=false --prec-div=true --prec-sqrt=true --fmad=true>
  $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:strict>
)

# Same analytic Hodge construction and 3072 trials, retaining 2048 modes.
add_executable(hodge_vector_analytic_rr3072_k2048 src/hodge_vector_analytic_k1024.cu)
target_include_directories(hodge_vector_analytic_rr3072_k2048 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hodge_vector_analytic_rr3072_k2048 PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
target_compile_features(hodge_vector_analytic_rr3072_k2048 PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(hodge_vector_analytic_rr3072_k2048 PRIVATE
  TRANSFER_STRICT_FP32=1
  HODGE_ANALYTIC_TRIALS=3072
  HODGE_ANALYTIC_MODES=2048
)
set_target_properties(hodge_vector_analytic_rr3072_k2048 PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION OFF
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)
target_compile_options(hodge_vector_analytic_rr3072_k2048 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ftz=false --prec-div=true --prec-sqrt=true --fmad=true>
  $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:strict>
)

add_executable(epoch50_review src/epoch50_review.cu)
target_include_directories(epoch50_review PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(epoch50_review PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
target_compile_features(epoch50_review PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(epoch50_review PRIVATE TRANSFER_STRICT_FP32=1)
set_target_properties(epoch50_review PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION OFF
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)
target_compile_options(epoch50_review PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ftz=false --prec-div=true --prec-sqrt=true --fmad=true>
  $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:strict>
)

add_executable(reconstruct_three_dns src/reconstruct_three_dns.cu)
target_include_directories(reconstruct_three_dns PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(reconstruct_three_dns PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
target_compile_features(reconstruct_three_dns PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(reconstruct_three_dns PRIVATE TRANSFER_STRICT_FP32=1)
set_target_properties(reconstruct_three_dns PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION OFF
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)
target_compile_options(reconstruct_three_dns PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ftz=false --prec-div=true --prec-sqrt=true --fmad=true>
  $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:strict>
)

add_executable(reconstruct_boundary_gt src/reconstruct_three_dns.cu)
target_include_directories(reconstruct_boundary_gt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(reconstruct_boundary_gt PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
target_compile_features(reconstruct_boundary_gt PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(reconstruct_boundary_gt PRIVATE
  TRANSFER_STRICT_FP32=1
  TRANSFER_SINGLE_BOUNDARY_GT=1
)
set_target_properties(reconstruct_boundary_gt PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION OFF
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)
target_compile_options(reconstruct_boundary_gt PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ftz=false --prec-div=true --prec-sqrt=true --fmad=true>
  $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:strict>
)

add_executable(reconstruct_boundary_gt_velocity_rayleigh src/reconstruct_three_dns.cu)
target_include_directories(reconstruct_boundary_gt_velocity_rayleigh PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(reconstruct_boundary_gt_velocity_rayleigh PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
target_compile_features(reconstruct_boundary_gt_velocity_rayleigh PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(reconstruct_boundary_gt_velocity_rayleigh PRIVATE
  TRANSFER_STRICT_FP32=1
  TRANSFER_SINGLE_BOUNDARY_GT=1
  TRANSFER_VELOCITY_RAYLEIGH=1
)
set_target_properties(reconstruct_boundary_gt_velocity_rayleigh PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION OFF
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)
target_compile_options(reconstruct_boundary_gt_velocity_rayleigh PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ftz=false --prec-div=true --prec-sqrt=true --fmad=true>
  $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:strict>
)

function(add_boundary_reconstructor_k2048 target)
  add_executable(${target} src/reconstruct_three_dns.cu)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
  target_compile_features(${target} PRIVATE cxx_std_17 cuda_std_17)
  target_compile_definitions(${target} PRIVATE
    TRANSFER_STRICT_FP32=1
    TRANSFER_SINGLE_BOUNDARY_GT=1
    RECONSTRUCT_MODES=2048
    ${ARGN}
  )
  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES 89
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
  )
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ftz=false --prec-div=true --prec-sqrt=true --fmad=true>
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:strict>
  )
endfunction()

add_boundary_reconstructor_k2048(reconstruct_boundary_gt_k2048 TRANSFER_VECTOR_LIBRARY=1)
add_boundary_reconstructor_k2048(reconstruct_boundary_gt_velocity_rayleigh_k2048 TRANSFER_VELOCITY_RAYLEIGH=1)
add_boundary_reconstructor_k2048(reconstruct_boundary_gt_vorticity_rayleigh_k2048 TRANSFER_VORTICITY_RAYLEIGH=1)
