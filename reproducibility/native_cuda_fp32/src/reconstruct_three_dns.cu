#define TRANSFER_VECTOR_LIBRARY 1
#include "transfer_vector_fp32.cu"

#include <cusolverDn.h>

namespace {

#define CUSOLVER_CHECK(call)                                                              \
  do {                                                                                    \
    const cusolverStatus_t status__ = (call);                                              \
    if (status__ != CUSOLVER_STATUS_SUCCESS)                                               \
      throw std::runtime_error("cuSOLVER status " + std::to_string(int(status__)));       \
  } while (0)

constexpr int kGrid = 128;
#ifndef RECONSTRUCT_MODES
#define RECONSTRUCT_MODES 1024
#endif
constexpr int kReconstructModes = RECONSTRUCT_MODES;
#if defined(TRANSFER_SINGLE_BOUNDARY_GT)
constexpr int kCases = 1;
#else
constexpr int kCases = 3;
#endif
constexpr int kChunkPoints = 262144;
constexpr Scalar kRidgeFactor = 1.0e-6f;

#if defined(TRANSFER_SINGLE_BOUNDARY_GT)
const char* kCaseIds[kCases] = {"boundary_compatible_gt"};
#else
const char* kCaseIds[kCases] = {
    "case_01_independent_dns", "case_02_large_eddy", "case_03_natural_large_eddy"};
#endif

__device__ void velocity_vorticity_from_jet(const Scalar* a, int stride, Scalar* velocity,
                                             Scalar* vorticity) {
  auto A = [&](int component, int jet) { return a[jet * OUT * stride + component]; };
  velocity[0] = A(0,2) + A(1,3);
  velocity[1] = -A(0,1) + A(2,3);
  velocity[2] = -A(1,1) - A(2,2);
  vorticity[0] = -A(1,5) - A(2,7) + A(0,6) - A(2,9);
  vorticity[1] = A(0,8) + A(1,9) + A(1,4) + A(2,5);
  vorticity[2] = -A(0,4) + A(2,6) - A(0,7) - A(1,8);
}

__global__ void jet_to_flow_kernel(const Scalar* composed, int count, Scalar* velocity,
                                   Scalar* vorticity) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= count) return;
  Scalar local[J * OUT];
  for (int jet = 0; jet < J; ++jet)
    for (int component = 0; component < OUT; ++component)
      local[jet * OUT + component] = composed[jet * OUT * count + component + OUT * n];
  velocity_vorticity_from_jet(local, 1, velocity + 3*n, vorticity + 3*n);
}

__global__ void add_diagonal_kernel(Scalar* matrix, int size, Scalar ridge) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) matrix[i + size*i] += ridge;
}

template <class T>
std::vector<T> read_raw(const fs::path& path, std::size_t count) {
  if (!fs::exists(path) || fs::file_size(path) != count * sizeof(T))
    throw std::runtime_error("unexpected raw file size: " + path.string());
  std::vector<T> values(count);
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
  if (!stream) throw std::runtime_error("raw file read failed: " + path.string());
  return values;
}

template <class T>
void write_raw(const fs::path& path, const std::vector<T>& values) {
  const fs::path temporary = path.string() + ".tmp";
  std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(values.data()),
               std::streamsize(values.size() * sizeof(T)));
  stream.close();
  if (!stream) throw std::runtime_error("raw file write failed: " + path.string());
  if (fs::exists(path)) fs::remove(path);
  fs::rename(temporary, path);
}

std::string chunk_name(int chunk, const char* field) {
  std::ostringstream name;
  name << "chunk_" << std::setw(2) << std::setfill('0') << chunk << "_" << field << ".f32";
  return name.str();
}

void write_progress(const fs::path& output, int complete_chunks, int chunks,
                    std::uint64_t fluid_nodes, bool basis_complete, bool reconstruction_complete) {
  std::ofstream stream(output / "progress.json", std::ios::trunc);
  stream << "{\n  \"complete_chunks\": " << complete_chunks
         << ",\n  \"chunks\": " << chunks
         << ",\n  \"fluid_nodes\": " << fluid_nodes
         << ",\n  \"modes\": " << kReconstructModes << ",\n  \"grid\": [128, 128, 128],\n"
         << "  \"basis_complete\": " << (basis_complete ? "true" : "false")
         << ",\n  \"reconstruction_complete\": "
         << (reconstruction_complete ? "true" : "false") << "\n}\n";
}

struct Errors {
  Scalar velocity_error_sq = 0.0f;
  Scalar velocity_norm_sq = 0.0f;
  Scalar vorticity_error_sq = 0.0f;
  Scalar vorticity_norm_sq = 0.0f;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4)
      throw std::runtime_error("usage: reconstruct_three_dns checkpoint.bin prepared_dir output_dir");
    const fs::path checkpoint = argv[1];
    const fs::path prepared = argv[2];
    const fs::path output = argv[3];
    const fs::path basis_dir = output / "basis_chunks";
    fs::create_directories(basis_dir);

    const auto geometry_values = read_raw<Scalar>(prepared / "geometry_xyzr.f32", 12);
    Geometry geometry{};
    std::copy(geometry_values.begin(), geometry_values.end(), &geometry.sphere[0][0]);
    const std::uint64_t voxels = std::uint64_t(kGrid) * kGrid * kGrid;
    if (!fs::exists(prepared / "fluid_indices.u32") ||
        fs::file_size(prepared / "fluid_indices.u32") % sizeof(std::uint32_t))
      throw std::runtime_error("invalid fluid index file");
    const std::size_t fluid_nodes = fs::file_size(prepared / "fluid_indices.u32") / sizeof(std::uint32_t);
    const auto indices = read_raw<std::uint32_t>(prepared / "fluid_indices.u32", fluid_nodes);
    for (std::uint32_t index : indices)
      if (index >= voxels) throw std::runtime_error("fluid index out of range");
    std::array<std::vector<Scalar>, kCases> target_velocity, target_vorticity;
    for (int c = 0; c < kCases; ++c) {
      target_velocity[c] = read_raw<Scalar>(prepared / (std::string(kCaseIds[c]) + "_velocity.f32"), fluid_nodes * 3);
      target_vorticity[c] = read_raw<Scalar>(prepared / (std::string(kCaseIds[c]) + "_vorticity.f32"), fluid_nodes * 3);
    }

    Options options;
    options.modes = kReconstructModes;
    Trainer trainer(options);
    const int epoch = trainer.load(checkpoint);
    if (epoch <= 0)
      throw std::runtime_error("reconstruction requires a positive-epoch TVFP32V2 checkpoint");
    cublasHandle_t handle = trainer.cublas_handle();
    const int modes = options.modes;
    DeviceBuffer<Scalar> gram(std::size_t(modes) * modes), rhs(std::size_t(modes) * kCases);
    CUDA_CHECK(cudaMemset(gram.data(), 0, gram.size() * sizeof(Scalar)));
    CUDA_CHECK(cudaMemset(rhs.data(), 0, rhs.size() * sizeof(Scalar)));
    const int chunks = int((fluid_nodes + kChunkPoints - 1) / kChunkPoints);
    write_progress(output, 0, chunks, fluid_nodes, false, false);

    const Scalar one = 1.0f, zero = 0.0f;
    for (int chunk = 0; chunk < chunks; ++chunk) {
      const std::size_t begin = std::size_t(chunk) * kChunkPoints;
      const int active = int(std::min<std::size_t>(kChunkPoints, fluid_nodes - begin));
      const int rows = active * 3;
      const std::size_t basis_values = std::size_t(rows) * modes;
      const fs::path velocity_path = basis_dir / chunk_name(chunk, "velocity");
      const fs::path vorticity_path = basis_dir / chunk_name(chunk, "vorticity");
      const bool cached = fs::exists(velocity_path) && fs::exists(vorticity_path) &&
          fs::file_size(velocity_path) == basis_values * sizeof(Scalar) &&
          fs::file_size(vorticity_path) == basis_values * sizeof(Scalar);
      DeviceBuffer<Scalar> basis_velocity(basis_values);
      if (cached) {
        std::cout << "reuse basis chunk " << (chunk + 1) << "/" << chunks << std::endl;
        const auto host = read_raw<Scalar>(velocity_path, basis_values);
        CUDA_CHECK(cudaMemcpy(basis_velocity.data(), host.data(), host.size() * sizeof(Scalar), cudaMemcpyHostToDevice));
      } else {
        std::cout << "evaluate basis chunk " << (chunk + 1) << "/" << chunks
                  << " points=" << active << std::endl;
        DeviceBuffer<Scalar> basis_vorticity(basis_values);
        {
          std::vector<Scalar> points(std::size_t(active) * 3);
          for (int n = 0; n < active; ++n) {
            const std::uint32_t flat = indices[begin + n];
            const int x = flat % kGrid;
            const int y = (flat / kGrid) % kGrid;
            const int z = flat / (kGrid * kGrid);
            // Every GT and mask in this comparison is cell centred.  Using
            // node coordinates here would evaluate a different geometry.
            points[3*n] = (Scalar(x) + 0.5f) / Scalar(kGrid);
            points[3*n+1] = (Scalar(y) + 0.5f) / Scalar(kGrid);
            points[3*n+2] = (Scalar(z) + 0.5f) / Scalar(kGrid);
          }
          FieldWorkspace field(active, 1);
          NetworkEvaluator network(handle, active);
          DeviceBuffer<Scalar> raw_velocity(rows), raw_vorticity(rows);
          CUDA_CHECK(cudaMemcpy(field.points.data(), points.data(), points.size() * sizeof(Scalar), cudaMemcpyHostToDevice));
          network.prepare(field.points.data(), geometry);
          for (int mode = 0; mode < modes; ++mode) {
            network.forward(trainer.parameters_data() + std::size_t(mode) * P);
            build_vector_field_kernel<<<(field.potential_rows + 255) / 256, 256>>>(
                network.workspace().raw.data(), network.workspace().envelope.data(), active,
                network.workspace().composed_potential.data(), field.potential.data(), field.gradient.data());
            jet_to_flow_kernel<<<(active + 255) / 256, 256>>>(
                network.workspace().composed_potential.data(), active, raw_velocity.data(), raw_vorticity.data());
            CUDA_CHECK(cudaMemcpy(basis_velocity.data() + std::size_t(mode) * rows, raw_velocity.data(),
                                  rows * sizeof(Scalar), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(basis_vorticity.data() + std::size_t(mode) * rows, raw_vorticity.data(),
                                  rows * sizeof(Scalar), cudaMemcpyDeviceToDevice));
            if ((mode + 1) % 128 == 0)
              std::cout << "  mode " << (mode + 1) << "/" << modes << std::endl;
          }
          CUDA_CHECK(cudaDeviceSynchronize());
        }
        std::vector<Scalar> host(basis_values);
        CUDA_CHECK(cudaMemcpy(host.data(), basis_velocity.data(), host.size() * sizeof(Scalar), cudaMemcpyDeviceToHost));
        write_raw(velocity_path, host);
        CUDA_CHECK(cudaMemcpy(host.data(), basis_vorticity.data(), host.size() * sizeof(Scalar), cudaMemcpyDeviceToHost));
        write_raw(vorticity_path, host);
      }

      const Scalar beta = chunk == 0 ? zero : one;
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, modes, modes, rows,
          &one, basis_velocity.data(), rows, basis_velocity.data(), rows, &beta, gram.data(), modes));
      std::vector<Scalar> host_targets(std::size_t(rows) * kCases);
      for (int c = 0; c < kCases; ++c)
        std::copy_n(target_velocity[c].data() + 3 * begin, rows,
                    host_targets.data() + std::size_t(c) * rows);
      DeviceBuffer<Scalar> device_targets(host_targets.size());
      CUDA_CHECK(cudaMemcpy(device_targets.data(), host_targets.data(),
                            host_targets.size() * sizeof(Scalar), cudaMemcpyHostToDevice));
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, modes, kCases, rows,
          &one, basis_velocity.data(), rows, device_targets.data(), rows, &beta, rhs.data(), modes));
      CUDA_CHECK(cudaDeviceSynchronize());
      write_progress(output, chunk + 1, chunks, fluid_nodes, chunk + 1 == chunks, false);
    }

    std::vector<Scalar> host_gram(std::size_t(modes) * modes);
    CUDA_CHECK(cudaMemcpy(host_gram.data(), gram.data(), host_gram.size() * sizeof(Scalar),
                          cudaMemcpyDeviceToHost));
    write_raw(output / "full128_velocity_gram_f32.bin", host_gram);

    Scalar trace = 0.0f;
    CUBLAS_CHECK(cublasSasum(handle, modes, gram.data(), modes + 1, &trace));
    const Scalar ridge = kRidgeFactor * trace / Scalar(modes);
    add_diagonal_kernel<<<(modes + 255) / 256, 256>>>(gram.data(), modes, ridge);
    cusolverDnHandle_t solver = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));
    int workspace_size = 0;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, modes,
                                               gram.data(), modes, &workspace_size));
    DeviceBuffer<Scalar> solver_workspace(workspace_size);
    DeviceBuffer<int> info(1);
    CUSOLVER_CHECK(cusolverDnSpotrf(solver, CUBLAS_FILL_MODE_LOWER, modes, gram.data(), modes,
                                    solver_workspace.data(), workspace_size, info.data()));
    int host_info = 0;
    CUDA_CHECK(cudaMemcpy(&host_info, info.data(), sizeof(int), cudaMemcpyDeviceToHost));
    if (host_info != 0)
      throw std::runtime_error("full-128 velocity Cholesky failed at pivot " + std::to_string(host_info));
    CUSOLVER_CHECK(cusolverDnSpotrs(solver, CUBLAS_FILL_MODE_LOWER, modes, kCases,
                                    gram.data(), modes, rhs.data(), modes, info.data()));
    CUSOLVER_CHECK(cusolverDnDestroy(solver));
    std::vector<Scalar> coefficients(std::size_t(modes) * kCases);
    CUDA_CHECK(cudaMemcpy(coefficients.data(), rhs.data(), coefficients.size() * sizeof(Scalar), cudaMemcpyDeviceToHost));
    for (int c = 0; c < kCases; ++c) {
      std::vector<Scalar> values(modes);
      std::copy_n(coefficients.data() + std::size_t(c) * modes, modes, values.data());
      write_raw(output / (std::string(kCaseIds[c]) + "_coefficients.f32"), values);
    }

    std::array<std::vector<Scalar>, kCases> reconstructed_velocity, reconstructed_vorticity;
    for (int c = 0; c < kCases; ++c) {
      reconstructed_velocity[c].assign(voxels * 3, 0.0f);
      reconstructed_vorticity[c].assign(voxels * 3, 0.0f);
    }
    std::array<Errors, kCases> errors{};
    for (int chunk = 0; chunk < chunks; ++chunk) {
      const std::size_t begin = std::size_t(chunk) * kChunkPoints;
      const int active = int(std::min<std::size_t>(kChunkPoints, fluid_nodes - begin));
      const int rows = active * 3;
      const std::size_t basis_values = std::size_t(rows) * modes;
      DeviceBuffer<Scalar> basis(basis_values), prediction(rows), difference(rows), target(rows);
      std::vector<Scalar> host_prediction(rows);
      for (int field_index = 0; field_index < 2; ++field_index) {
        const bool velocity_field = field_index == 0;
        const fs::path path = basis_dir / chunk_name(chunk, velocity_field ? "velocity" : "vorticity");
        const auto host_basis = read_raw<Scalar>(path, basis_values);
        CUDA_CHECK(cudaMemcpy(basis.data(), host_basis.data(), host_basis.size() * sizeof(Scalar), cudaMemcpyHostToDevice));
        for (int c = 0; c < kCases; ++c) {
          CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, rows, modes, &one, basis.data(), rows,
                                   rhs.data() + std::size_t(c) * modes, 1, &zero, prediction.data(), 1));
          CUDA_CHECK(cudaMemcpy(host_prediction.data(), prediction.data(), rows * sizeof(Scalar), cudaMemcpyDeviceToHost));
          auto& full = velocity_field ? reconstructed_velocity[c] : reconstructed_vorticity[c];
          for (int n = 0; n < active; ++n)
            std::copy_n(host_prediction.data() + 3*n, 3,
                        full.data() + std::size_t(indices[begin+n]) * 3);
          const auto& host_target = velocity_field ? target_velocity[c] : target_vorticity[c];
          CUDA_CHECK(cudaMemcpy(target.data(), host_target.data() + 3*begin,
                                rows * sizeof(Scalar), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(difference.data(), prediction.data(),
                                rows * sizeof(Scalar), cudaMemcpyDeviceToDevice));
          const Scalar minus = -1.0f;
          CUBLAS_CHECK(cublasSaxpy(handle, rows, &minus, target.data(), 1, difference.data(), 1));
          Scalar error_sq = 0.0f, norm_sq = 0.0f;
          CUBLAS_CHECK(cublasSdot(handle, rows, difference.data(), 1, difference.data(), 1, &error_sq));
          CUBLAS_CHECK(cublasSdot(handle, rows, target.data(), 1, target.data(), 1, &norm_sq));
          if (velocity_field) {
            errors[c].velocity_error_sq += error_sq;
            errors[c].velocity_norm_sq += norm_sq;
          } else {
            errors[c].vorticity_error_sq += error_sq;
            errors[c].vorticity_norm_sq += norm_sq;
          }
        }
      }
      std::cout << "reconstruct chunk " << (chunk + 1) << "/" << chunks << std::endl;
    }
    for (int c = 0; c < kCases; ++c) {
      write_raw(output / (std::string(kCaseIds[c]) + "_reconstruction_velocity_128.f32"),
                reconstructed_velocity[c]);
      write_raw(output / (std::string(kCaseIds[c]) + "_reconstruction_vorticity_128.f32"),
                reconstructed_vorticity[c]);
    }

    std::ofstream metrics(output / "metrics.json", std::ios::trunc);
    metrics << std::setprecision(10)
            << "{\n  \"complete\": true,\n  \"checkpoint_epoch\": " << epoch << ",\n"
             << "  \"checkpoint_format\": \"TVFP32V2\",\n  \"modes\": " << modes << ",\n"
             << "  \"training_objective\": \""
             << (kVorticityRayleigh ? "vorticity gradient Rayleigh" :
                 (kVelocityRayleigh ? "velocity gradient Rayleigh" : "potential Dirichlet Rayleigh")) << "\",\n"
             << "  \"grid\": [128, 128, 128],\n"
             << "  \"coordinates\": \"cell centers (i+0.5)/128\",\n"
             << "  \"potential_derivative_order\": " << (kVorticityRayleigh ? 3 : (kVelocityRayleigh ? 2 : 1))
             << ",\n  \"active_jet_components\": " << J
             << ",\n  \"jet3\": " << (kVorticityRayleigh ? "true" : "false") << ",\n"
             << "  \"projection\": \"full fluid-node velocity least squares\",\n"
            << "  \"fluid_nodes\": " << fluid_nodes
            << ",\n  \"ridge_factor\": " << kRidgeFactor
            << ",\n  \"ridge\": " << ridge << ",\n  \"cases\": [\n";
    for (int c = 0; c < kCases; ++c) {
      const Scalar velocity_error = std::sqrt(errors[c].velocity_error_sq / errors[c].velocity_norm_sq);
      const Scalar vorticity_error = std::sqrt(errors[c].vorticity_error_sq / errors[c].vorticity_norm_sq);
      metrics << "    {\"id\": \"" << kCaseIds[c]
              << "\", \"velocity_relative_l2\": " << velocity_error
              << ", \"vorticity_relative_l2\": " << vorticity_error << "}"
              << (c + 1 < kCases ? "," : "") << "\n";
    }
    metrics << "  ]\n}\n";
    write_progress(output, chunks, chunks, fluid_nodes, true, true);
    std::cout << "complete full-128 K=" << modes << " reconstruction" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << std::endl;
    return 1;
  }
}
