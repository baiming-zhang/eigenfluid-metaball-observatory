"""Build and reconstruct the strict 2048-complete-A FD-Lap basis.

Mode semantics are deliberately explicit:

* 2048 distinct Dirichlet scalar shapes are used.
* Shape k defines one complete antisymmetric potential A_k=e*phi_k*c_k.
* c_k=(c12,c13,c23) is fixed by a local 3x3 velocity/curl Rayleigh problem.
* Only one external coefficient q_k is exposed for each complete A_k.

Thus K=2048 means exactly 2048 complete A modes and 2048 independent q
coefficients.  The three tensor polarizations are internal components, not
additional modes or degrees of freedom.  Projection and common FD-curl error
evaluation use every fluid node of the 128^3 grid without regularization,
subsampling, or TF32.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
import scipy.linalg as sla
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCALAR_ROOT = RESULTS / "laplacian_128_scalar_k2048_from_verified_k1024"
OLD_CELL_CACHE = (
    RESULTS
    / "laplacian_128_scalar_k1024"
    / "basis_full128_one_to_one"
    / "laplacian_scalar_cell_center_128.npy"
)
BASIS_ROOT = RESULTS / "laplacian_strict_complete_A2048_basis_full128"
GT_ROOT = RESULTS / "boundary_compatible_double_zero_gt_128_v2"
OUTPUT_ROOT = RESULTS / "fd_lap_strict_matched_e_phiD_complete_A2048_k2048_full128"

CORE_SCRIPT = ROOT / "scripts" / "reconstruct_three_dns_laplacian_double_zero_A1024_DOF3072.py"
SHARED_SCRIPT = ROOT / "scripts" / "reconstruct_three_dns_laplacian_scalar_k1024_k2048.py"
CURL_SCRIPT = ROOT / "scripts" / "generate_envelope_matched_gt_first_frame_128.py"
VIZ_SCRIPT = ROOT / "scripts" / "finalize_strict_fd_lap_vs_eigenfluid_k1024.py"

GRID = 128
VOXELS = GRID**3
FLUID_NODES = 1_599_754
COMPLETE_A_COUNT = 2048
POLARIZATIONS_PER_A = 3
K_SMALL = 1024
K_LARGE = 2048
WEIGHT = (1.0 / GRID) ** 3

BLUE = "#3B6FB6"
LIGHT_BLUE = "#7CAACB"
RED = "#C83E4D"
ORANGE = "#E07B39"
DARK = "#1E293B"
GRAY = "#64748B"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load_module("complete_A2048_core", CORE_SCRIPT)
shared = load_module("complete_A2048_shared", SHARED_SCRIPT)
curl_helper = load_module("complete_A2048_curl", CURL_SCRIPT)
viz = load_module("complete_A2048_viz", VIZ_SCRIPT)
helper = core.helper

# The reused cache builders intentionally read these globals at call time.
core.A_GROUPS = COMPLETE_A_COUNT
core.SCALAR_DOFS = COMPLETE_A_COUNT * POLARIZATIONS_PER_A
shared.K_LARGE = COMPLETE_A_COUNT
shared.BASE_SCALAR_MODES = K_SMALL


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def replace_with_retry(temporary: Path, path: Path) -> None:
    for attempt in range(60):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 59:
                raise
            time.sleep(0.25)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    replace_with_retry(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    replace_with_retry(temporary, path)


def open_or_create(path: Path, shape: tuple[int, ...]) -> np.memmap:
    if path.exists():
        cached = np.load(path, mmap_mode="r+")
        if cached.shape != shape or cached.dtype != np.float32:
            raise ValueError(f"unexpected cache {path}: {cached.shape} {cached.dtype}")
        return cached
    return open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def ranges(size: int, blocks: int) -> list[tuple[int, int]]:
    return [(size * i // blocks, size * (i + 1) // blocks) for i in range(blocks)]


def trim_windows_working_set() -> None:
    if os.name == "nt":
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.EmptyWorkingSet(handle)


def single_zero_envelope(e_squared_path: Path, output_path: Path) -> np.memmap:
    e_squared = np.load(e_squared_path, mmap_mode="r")
    if e_squared.shape != (GRID, GRID, GRID):
        raise ValueError(f"unexpected GT envelope {e_squared.shape}")
    if output_path.exists():
        output = np.load(output_path, mmap_mode="r")
        if output.shape != e_squared.shape or output.dtype != np.float32:
            raise ValueError(f"unexpected single-zero envelope {output.shape}")
    else:
        output = open_memmap(output_path, mode="w+", dtype=np.float32, shape=e_squared.shape)
        np.sqrt(np.asarray(e_squared), out=output)
        output.flush()
    mismatch = float(np.max(np.abs(np.asarray(output) ** 2 - np.asarray(e_squared))))
    if mismatch > 2.0e-6:
        raise RuntimeError(f"single-zero envelope square mismatch {mismatch:.3e}")
    return np.load(output_path, mmap_mode="r")


def build_cell_scalar_cache_gpu(
    cfg: Any,
    geometry: np.ndarray,
    endpoint_path: Path,
    old_cache_path: Path,
    output_path: Path,
    progress_path: Path,
    mode_batch: int = 32,
) -> np.memmap:
    """Trilinearly sample endpoint modes at cell centers in CUDA batches.

    This is the same endpoint-to-cell-center map as scipy.map_coordinates with
    order=1, zero padding and prefilter disabled.  Batching changes only the
    execution schedule and avoids one strided 1.55M-entry read plus one
    128^3 interpolation launch per scalar mode.
    """
    if not torch.cuda.is_available():
        log("CUDA unavailable for cell interpolation; use the scalar CPU cache builder")
        return shared.build_cell_scalar_cache(
            cfg, geometry, endpoint_path, old_cache_path, output_path, progress_path
        )

    endpoint_grid = shared.base.build_grid(cfg, GRID, geometry)
    endpoint = np.load(endpoint_path, mmap_mode="r")
    if endpoint.shape != (len(endpoint_grid["points"]), COMPLETE_A_COUNT):
        raise ValueError(f"unexpected endpoint scalar cache {endpoint.shape}")
    old_cache = np.load(old_cache_path, mmap_mode="r")
    if old_cache.shape != (VOXELS, K_SMALL):
        raise ValueError(f"unexpected old cell scalar cache {old_cache.shape}")
    if output_path.exists():
        output = np.load(output_path, mmap_mode="r+")
        if output.shape != (VOXELS, COMPLETE_A_COUNT) or output.dtype != np.float32:
            raise ValueError(f"unexpected cell scalar output {output.shape} {output.dtype}")
    else:
        output = open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(VOXELS, COMPLETE_A_COUNT),
        )

    next_mode = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            log("reuse complete 2048-scalar cell-center cache")
            return np.load(output_path, mmap_mode="r")
        next_mode = int(progress.get("next_scalar_mode", 0))
    if next_mode < K_SMALL:
        log("copy exact cell-centered scalar modes 1-1024")
        for start in range(0, VOXELS, 131_072):
            stop = min(start + 131_072, VOXELS)
            output[start:stop, :K_SMALL] = old_cache[start:stop]
        output.flush()
        next_mode = K_SMALL
        atomic_json(
            progress_path,
            {
                "complete": False,
                "next_scalar_mode": next_mode,
                "scalar_modes": COMPLETE_A_COUNT,
                "grid": [GRID, GRID, GRID],
                "backend": "CUDA trilinear FP32",
            },
        )

    # The final eigenspectrum is stored node-major.  Reading 32 columns from
    # that 12.7 GiB C-order memmap causes millions of small page faults per
    # interpolation batch.  Transpose only the new modes 1025..2048 once by
    # contiguous node rows, then every CUDA batch becomes a sequential read.
    endpoint_mode_major_path = (
        output_path.parent / "endpoint_modes_1025_2048_mode_major.npy"
    )
    endpoint_mode_major_progress_path = (
        output_path.parent / "endpoint_mode_major_progress.json"
    )
    endpoint_mode_major_shape = (COMPLETE_A_COUNT - K_SMALL, endpoint.shape[0])
    if endpoint_mode_major_path.exists():
        endpoint_mode_major = np.load(endpoint_mode_major_path, mmap_mode="r+")
        if (
            endpoint_mode_major.shape != endpoint_mode_major_shape
            or endpoint_mode_major.dtype != np.float32
        ):
            raise ValueError(
                f"unexpected endpoint mode-major cache {endpoint_mode_major.shape} "
                f"{endpoint_mode_major.dtype}"
            )
    else:
        endpoint_mode_major = open_memmap(
            endpoint_mode_major_path,
            mode="w+",
            dtype=np.float32,
            shape=endpoint_mode_major_shape,
        )
    next_endpoint_row = 0
    if endpoint_mode_major_progress_path.exists():
        endpoint_progress = json.loads(
            endpoint_mode_major_progress_path.read_text(encoding="utf-8")
        )
        if endpoint_progress.get("complete"):
            next_endpoint_row = endpoint.shape[0]
        else:
            next_endpoint_row = int(endpoint_progress.get("next_endpoint_row", 0))
    if next_endpoint_row < endpoint.shape[0]:
        log(
            "build sequential mode-major endpoint cache for modes 1025-2048; "
            f"resume row {next_endpoint_row}/{endpoint.shape[0]}"
        )
        row_chunk = 8192
        for row_start in range(next_endpoint_row, endpoint.shape[0], row_chunk):
            row_stop = min(row_start + row_chunk, endpoint.shape[0])
            contiguous_rows = np.asarray(
                endpoint[row_start:row_stop, K_SMALL:COMPLETE_A_COUNT],
                dtype=np.float32,
            )
            endpoint_mode_major[:, row_start:row_stop] = contiguous_rows.T
            if row_stop % 131_072 < row_chunk or row_stop == endpoint.shape[0]:
                endpoint_mode_major.flush()
                atomic_json(
                    endpoint_mode_major_progress_path,
                    {
                        "complete": row_stop == endpoint.shape[0],
                        "next_endpoint_row": row_stop,
                        "endpoint_rows": endpoint.shape[0],
                        "scalar_modes": COMPLETE_A_COUNT - K_SMALL,
                        "source_mode_start_one_based": K_SMALL + 1,
                        "layout": "mode-major C-order",
                    },
                )
                log(
                    f"endpoint mode-major rows {row_stop}/{endpoint.shape[0]}"
                )
    del endpoint_mode_major
    endpoint_mode_major = np.load(endpoint_mode_major_path, mmap_mode="r")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    normalized = 2.0 * (torch.arange(GRID, dtype=torch.float32) + 0.5) / GRID - 1.0
    zz, yy, xx = torch.meshgrid(normalized, normalized, normalized, indexing="ij")
    sample_grid = torch.stack((xx, yy, zz), dim=-1).unsqueeze(0).to(device)
    endpoint_mask = np.asarray(endpoint_grid["mask"], dtype=bool)
    shape = tuple(int(value) for value in endpoint_grid["shape"])
    if shape != (GRID, GRID, GRID):
        raise ValueError(f"unexpected endpoint grid shape {shape}")

    log(
        f"CUDA FP32 trilinear cell interpolation resumes at mode {next_mode + 1}; "
        f"batch={mode_batch}"
    )
    with torch.inference_mode():
        for start in range(next_mode, COMPLETE_A_COUNT, mode_batch):
            stop = min(start + mode_batch, COMPLETE_A_COUNT)
            width = stop - start
            endpoint_block = np.asarray(
                endpoint_mode_major[start - K_SMALL : stop - K_SMALL],
                dtype=np.float32,
            )
            volumes = np.zeros((width,) + shape, dtype=np.float32)
            volumes[:, endpoint_mask] = endpoint_block
            volume_gpu = torch.from_numpy(volumes[:, None]).to(device)
            sampled_gpu = torch.nn.functional.grid_sample(
                volume_gpu,
                sample_grid.expand(width, -1, -1, -1, -1),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = (
                sampled_gpu[:, 0]
                .reshape(width, VOXELS)
                .transpose(0, 1)
                .contiguous()
                .cpu()
                .numpy()
            )
            output[:, start:stop] = sampled
            # np.memmap.flush() on this 17.18 GB node-major file scans the
            # full mapping on Windows.  Flushing every 32-mode CUDA batch made
            # a few-second interpolation wait roughly six minutes.  Commit a
            # resumable checkpoint every 256 modes instead; an interruption
            # can lose at most one deterministic 256-mode group, which is
            # safely recomputed from the immutable endpoint cache.
            checkpoint = ((stop - next_mode) % 256 == 0) or stop == COMPLETE_A_COUNT
            if checkpoint:
                output.flush()
                atomic_json(
                    progress_path,
                    {
                        "complete": stop == COMPLETE_A_COUNT,
                        "next_scalar_mode": stop,
                        "scalar_modes": COMPLETE_A_COUNT,
                        "grid": [GRID, GRID, GRID],
                        "endpoint_interpolation": "trilinear endpoint-to-cell-center",
                        "backend": "CUDA trilinear FP32",
                        "tf32": False,
                        "mode_batch": mode_batch,
                        "checkpoint_modes": 256,
                    },
                )
                log(
                    f"cell-centered scalar modes {stop}/{COMPLETE_A_COUNT} "
                    "(CUDA, checkpoint committed)"
                )
            else:
                log(
                    f"cell-centered scalar modes {stop}/{COMPLETE_A_COUNT} "
                    "(CUDA, staged)"
                )
            del endpoint_block, volumes, volume_gpu, sampled_gpu, sampled
    return np.load(output_path, mmap_mode="r")


def local_mass(gradient: np.ndarray) -> np.ndarray:
    dx = np.asarray(gradient[:, 0], dtype=np.float64)
    dy = np.asarray(gradient[:, 1], dtype=np.float64)
    dz = np.asarray(gradient[:, 2], dtype=np.float64)
    xx = float(dx @ dx)
    yy = float(dy @ dy)
    zz = float(dz @ dz)
    xy = float(dx @ dy)
    xz = float(dx @ dz)
    yz = float(dy @ dz)
    return WEIGHT * np.asarray(
        [[yy + xx, yz, -xz], [yz, zz + xx, xy], [-xz, xy, zz + yy]],
        dtype=np.float64,
    )


def local_stiffness(w12: np.ndarray, w13: np.ndarray, w23: np.ndarray) -> np.ndarray:
    values = np.stack((w12, w13, w23), axis=2)
    return WEIGHT * np.einsum(
        "nci,ncj->ij", values, values, dtype=np.float64, optimize=True
    )


def fix_sign(vector: np.ndarray) -> np.ndarray:
    pivot = int(np.argmax(np.abs(vector)))
    return -vector if vector[pivot] < 0.0 else vector


def build_complete_modes(
    gradient_path: Path,
    mask: np.ndarray,
    flat_indices: np.ndarray,
    velocity_path: Path,
    polarization_path: Path,
    progress_path: Path,
    scalar_batch: int,
) -> tuple[np.memmap, np.ndarray, np.ndarray, dict[str, Any]]:
    velocity = open_or_create(velocity_path, (COMPLETE_A_COUNT, FLUID_NODES, 3))
    polarizations = np.zeros((COMPLETE_A_COUNT, 3), dtype=np.float64)
    rayleigh = np.full(COMPLETE_A_COUNT, np.nan, dtype=np.float64)
    residuals = np.full(COMPLETE_A_COUNT, np.nan, dtype=np.float64)
    next_mode = 0
    if polarization_path.exists():
        with np.load(polarization_path, allow_pickle=False) as cached:
            polarizations = np.asarray(cached["polarizations"], dtype=np.float64)
            rayleigh = np.asarray(cached["rayleigh"], dtype=np.float64)
            residuals = np.asarray(cached["relative_residuals"], dtype=np.float64)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            log("reuse complete 2048 complete-A velocity basis and internal polarizations")
            audit = progress["audit"]
            return np.load(velocity_path, mmap_mode="r"), polarizations, rayleigh, audit
        next_mode = int(progress.get("next_mode", 0))

    gradient = np.load(gradient_path, mmap_mode="r")
    if gradient.shape != (COMPLETE_A_COUNT, FLUID_NODES, 3):
        raise ValueError(f"unexpected gradient basis {gradient.shape}")
    spacing = 1.0 / GRID
    for start in range(next_mode, COMPLETE_A_COUNT, scalar_batch):
        stop = min(start + scalar_batch, COMPLETE_A_COUNT)
        width = stop - start
        gradient_block = np.asarray(gradient[start:stop], dtype=np.float32)
        volume = np.zeros((VOXELS, 3, width), dtype=np.float32)
        volume[flat_indices] = gradient_block.transpose(1, 2, 0)
        volume = volume.reshape(mask.shape + (3, width))
        dx, dy, dz = volume[..., 0, :], volume[..., 1, :], volume[..., 2, :]
        dx_x = np.gradient(dx, spacing, axis=2, edge_order=2)
        dx_y = np.gradient(dx, spacing, axis=1, edge_order=2)
        dx_z = np.gradient(dx, spacing, axis=0, edge_order=2)
        dy_x = np.gradient(dy, spacing, axis=2, edge_order=2)
        dy_y = np.gradient(dy, spacing, axis=1, edge_order=2)
        dy_z = np.gradient(dy, spacing, axis=0, edge_order=2)
        dz_x = np.gradient(dz, spacing, axis=2, edge_order=2)
        dz_y = np.gradient(dz, spacing, axis=1, edge_order=2)
        dz_z = np.gradient(dz, spacing, axis=0, edge_order=2)

        w12 = np.empty((width, FLUID_NODES, 3), dtype=np.float32)
        w12[:, :, 0] = dx_z[mask].T
        w12[:, :, 1] = dy_z[mask].T
        w12[:, :, 2] = (-(dx_x + dy_y))[mask].T
        w13 = np.empty_like(w12)
        w13[:, :, 0] = (-dx_y)[mask].T
        w13[:, :, 1] = (dz_z + dx_x)[mask].T
        w13[:, :, 2] = (-dz_y)[mask].T
        w23 = np.empty_like(w12)
        w23[:, :, 0] = (-(dy_y + dz_z))[mask].T
        w23[:, :, 1] = dy_x[mask].T
        w23[:, :, 2] = dz_x[mask].T

        for local, mode in enumerate(range(start, stop)):
            mass = local_mass(gradient_block[local])
            stiffness = local_stiffness(w12[local], w13[local], w23[local])
            eigenvalues, eigenvectors = sla.eigh(
                stiffness, mass, driver="gvd", check_finite=False
            )
            coefficient = fix_sign(np.asarray(eigenvectors[:, 0], dtype=np.float64))
            left = stiffness @ coefficient
            right = (mass @ coefficient) * eigenvalues[0]
            residual = float(
                np.linalg.norm(left - right)
                / max(np.linalg.norm(left), np.linalg.norm(right), 1.0e-300)
            )
            polarizations[mode] = coefficient
            rayleigh[mode] = float(eigenvalues[0])
            residuals[mode] = residual

            dx_f, dy_f, dz_f = (
                gradient_block[local, :, 0],
                gradient_block[local, :, 1],
                gradient_block[local, :, 2],
            )
            c12, c13, c23 = coefficient.astype(np.float32)
            velocity[mode, :, 0] = c12 * dy_f + c13 * dz_f
            velocity[mode, :, 1] = -c12 * dx_f + c23 * dz_f
            velocity[mode, :, 2] = -c13 * dx_f - c23 * dy_f

        checkpoint = ((stop - next_mode) % 128 == 0) or stop == COMPLETE_A_COUNT
        if checkpoint:
            velocity.flush()
            audit = {
                "local_generalized_eigen_relative_residual_max": float(
                    np.nanmax(residuals[:stop])
                ),
                "local_rayleigh_min": float(np.nanmin(rayleigh[:stop])),
                "local_rayleigh_max": float(np.nanmax(rayleigh[:stop])),
                "internal_branch": "lowest eigenvector of each independent 3x3 polarization problem",
                "internal_polarization_coefficients_per_A": 3,
            }
            atomic_npz(
                polarization_path,
                polarizations=polarizations,
                rayleigh=rayleigh,
                relative_residuals=residuals,
            )
            atomic_json(
                progress_path,
                {
                    "complete": stop == COMPLETE_A_COUNT,
                    "next_mode": stop,
                    "complete_A_count": COMPLETE_A_COUNT,
                    "independent_coefficients": COMPLETE_A_COUNT,
                    "one_q_per_complete_A": True,
                    "polarizations_are_internal": True,
                    "common_fd_curl": "128^3 second-order",
                    "checkpoint_modes": 128,
                    "audit": audit,
                },
            )
            log(
                f"fixed internal polarization and complete velocity A modes "
                f"{stop}/{COMPLETE_A_COUNT} (checkpoint committed)"
            )
        elif (stop - next_mode) % 32 == 0:
            log(
                f"fixed internal polarization and complete velocity A modes "
                f"{stop}/{COMPLETE_A_COUNT} (staged)"
            )
        del (
            w23,
            w13,
            w12,
            dz_z,
            dz_y,
            dz_x,
            dy_z,
            dy_y,
            dy_x,
            dx_z,
            dx_y,
            dx_x,
            dz,
            dy,
            dx,
            volume,
            gradient_block,
        )
        trim_windows_working_set()
    return np.load(velocity_path, mmap_mode="r"), polarizations, rayleigh, audit


def transpose_complete_basis(
    mode_major_path: Path,
    output_path: Path,
    progress_path: Path,
    row_chunk: int,
) -> np.memmap:
    source = np.load(mode_major_path, mmap_mode="r")
    if source.shape != (COMPLETE_A_COUNT, FLUID_NODES, 3):
        raise ValueError(f"unexpected complete-A mode-major basis {source.shape}")
    output = open_or_create(output_path, (FLUID_NODES, 3, COMPLETE_A_COUNT))
    next_row = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            log("reuse complete spatial-major complete-A basis")
            return np.load(output_path, mmap_mode="r")
        next_row = int(progress.get("next_row", 0))
    for start in range(next_row, FLUID_NODES, row_chunk):
        stop = min(start + row_chunk, FLUID_NODES)
        output[start:stop] = np.asarray(source[:, start:stop, :]).transpose(1, 2, 0)
        checkpoint_rows = row_chunk * 128
        checkpoint = ((stop - next_row) % checkpoint_rows == 0) or stop == FLUID_NODES
        if checkpoint:
            output.flush()
            atomic_json(
                progress_path,
                {
                    "complete": stop == FLUID_NODES,
                    "next_row": stop,
                    "complete_A_count": COMPLETE_A_COUNT,
                    "layout": "fluid-node x vector-component x complete-A",
                    "checkpoint_rows": checkpoint_rows,
                },
            )
            log(
                f"complete-A spatial-major rows {stop}/{FLUID_NODES} "
                "(checkpoint committed)"
            )
            trim_windows_working_set()
        elif (stop - next_row) % (row_chunk * 16) == 0:
            log(f"complete-A spatial-major rows {stop}/{FLUID_NODES} (staged)")
    return np.load(output_path, mmap_mode="r")


def assemble_complete_mass(
    basis_path: Path,
    target: np.ndarray,
    output_path: Path,
    progress_path: Path,
    spatial_blocks: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if output_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            with np.load(output_path, allow_pickle=False) as cached:
                return (
                    np.asarray(cached["mass"], dtype=np.float64),
                    np.asarray(cached["correlations"], dtype=np.float64),
                    json.loads(str(cached["audit_json"].item())),
                )
    basis = np.load(basis_path, mmap_mode="r")
    if basis.shape != (FLUID_NODES, 3, COMPLETE_A_COUNT):
        raise ValueError(f"unexpected complete-A spatial basis {basis.shape}")
    if target.shape != (FLUID_NODES, 3):
        raise ValueError(f"unexpected target {target.shape}")
    mass = np.zeros((COMPLETE_A_COUNT, COMPLETE_A_COUNT), dtype=np.float64)
    correlations = np.zeros(COMPLETE_A_COUNT, dtype=np.float64)
    partial_path = output_path.with_name(output_path.stem + "_partial.npz")
    next_block = 0
    if partial_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        next_block = int(progress.get("next_block", 0))
        with np.load(partial_path, allow_pickle=False) as cached:
            mass = np.asarray(cached["mass"], dtype=np.float64)
            correlations = np.asarray(cached["correlations"], dtype=np.float64)
    if not torch.cuda.is_available():
        raise RuntimeError("full 128^3 complete-A mass assembly requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    block_list = ranges(FLUID_NODES, spatial_blocks)
    for block_index, (start, stop) in enumerate(block_list):
        if block_index < next_block:
            continue
        gib = (stop - start) * 3 * COMPLETE_A_COUNT * 4 / 1024**3
        log(
            f"complete-A mass block {block_index + 1}/{spatial_blocks}: "
            f"basis={gib:.2f} GiB, all fluid nodes"
        )
        values = torch.tensor(
            np.asarray(basis[start:stop]), device="cuda", dtype=torch.float32
        ).reshape(-1, COMPLETE_A_COUNT)
        target_tensor = torch.tensor(
            np.asarray(target[start:stop]), device="cuda", dtype=torch.float32
        ).reshape(-1)
        mass += (WEIGHT * (values.T @ values)).double().cpu().numpy()
        correlations += (WEIGHT * (values.T @ target_tensor)).double().cpu().numpy()
        del target_tensor, values
        torch.cuda.empty_cache()
        atomic_npz(partial_path, mass=mass, correlations=correlations)
        atomic_json(
            progress_path,
            {
                "complete": block_index + 1 == spatial_blocks,
                "next_block": block_index + 1,
                "spatial_blocks": spatial_blocks,
                "complete_A_count": COMPLETE_A_COUNT,
                "independent_coefficients": COMPLETE_A_COUNT,
                "all_fluid_nodes": True,
                "subsampling": False,
                "tf32": False,
            },
        )
    mass = 0.5 * (mass + mass.T)
    eigenvalues = sla.eigvalsh(mass, check_finite=False, driver="evd")
    threshold = max(float(eigenvalues[-1]), 1.0e-300) * 1.0e-10
    audit = {
        "mass_min_eigenvalue": float(eigenvalues[0]),
        "mass_max_eigenvalue": float(eigenvalues[-1]),
        "mass_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "mass_rank_1e_minus_10": int(np.count_nonzero(eigenvalues > threshold)),
        "mass_dimension": COMPLETE_A_COUNT,
    }
    if audit["mass_rank_1e_minus_10"] != COMPLETE_A_COUNT or eigenvalues[0] <= 0.0:
        raise RuntimeError(f"complete-A mass matrix is not positive definite/full rank: {audit}")
    atomic_npz(
        output_path,
        mass=mass,
        correlations=correlations,
        audit_json=np.asarray(json.dumps(audit)),
    )
    return mass, correlations, audit


def solve_projection(mass: np.ndarray, correlations: np.ndarray, k: int) -> tuple[np.ndarray, dict[str, Any]]:
    local_mass = mass[:k, :k]
    local_correlations = correlations[:k]
    q = sla.solve(local_mass, local_correlations, assume_a="pos", check_finite=False)
    residual = float(
        np.linalg.norm(local_mass @ q - local_correlations)
        / max(np.linalg.norm(local_correlations), 1.0e-300)
    )
    return q, {
        "complete_A_count": k,
        "independent_coefficients": k,
        "one_q_per_complete_A": True,
        "normal_equation_relative_residual": residual,
        "regularization": None,
    }


def reconstruct_prefixes(
    basis_path: Path,
    q1024: np.ndarray,
    q2048: np.ndarray,
    flat_indices: np.ndarray,
    output_root: Path,
    spatial_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    path1024 = output_root / "velocity_complete_A1024_128.npy"
    path2048 = output_root / "velocity_complete_A2048_128.npy"
    if path1024.exists() and path2048.exists():
        a = np.load(path1024, mmap_mode="r")
        b = np.load(path2048, mmap_mode="r")
        if a.shape == b.shape == (GRID, GRID, GRID, 3):
            log("reuse complete K1024/K2048 velocity reconstructions")
            return np.asarray(a), np.asarray(b)
    basis = np.load(basis_path, mmap_mode="r")
    output1024 = np.zeros((VOXELS, 3), dtype=np.float32)
    output2048 = np.zeros((VOXELS, 3), dtype=np.float32)
    q1024_gpu = torch.tensor(q1024.astype(np.float32), device="cuda")
    q2048_gpu = torch.tensor(q2048.astype(np.float32), device="cuda")
    for block_index, (start, stop) in enumerate(ranges(FLUID_NODES, spatial_blocks)):
        values = torch.tensor(
            np.asarray(basis[start:stop]), device="cuda", dtype=torch.float32
        )
        u1024 = torch.einsum("nck,k->nc", values[:, :, :K_SMALL], q1024_gpu)
        u2048 = torch.einsum("nck,k->nc", values, q2048_gpu)
        output1024[flat_indices[start:stop]] = u1024.cpu().numpy()
        output2048[flat_indices[start:stop]] = u2048.cpu().numpy()
        del u2048, u1024, values
        torch.cuda.empty_cache()
        log(f"K1024/K2048 reconstruction block {block_index + 1}/{spatial_blocks}")
    velocity1024 = output1024.reshape((GRID, GRID, GRID, 3))
    velocity2048 = output2048.reshape((GRID, GRID, GRID, 3))
    np.save(path1024, velocity1024)
    np.save(path2048, velocity2048)
    return velocity1024, velocity2048


def relative_l2(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    ref = np.asarray(reference[mask], dtype=np.float64)
    got = np.asarray(estimate[mask], dtype=np.float64)
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def render_comparison(
    mask: np.ndarray,
    gt_vorticity: np.ndarray,
    vorticity1024: np.ndarray,
    vorticity2048: np.ndarray,
    level: float,
    errors1024: tuple[float, float],
    errors2048: tuple[float, float],
    png_path: Path,
    pdf_path: Path,
) -> None:
    fields = [
        gt_vorticity,
        vorticity1024,
        vorticity2048,
        vorticity1024 - gt_vorticity,
        vorticity2048 - gt_vorticity,
    ]
    meshes = [viz.surface(viz.magnitude(field, mask), level) for field in fields]
    titles = [
        "Matched GT",
        "FD-Lap: 1024 complete A",
        "FD-Lap: 2048 complete A",
        "1024-mode error",
        "2048-mode error",
    ]
    colors = [BLUE, RED, ORANGE, LIGHT_BLUE, RED]
    figure = plt.figure(figsize=(14.0, 7.1), facecolor="white")
    grid = figure.add_gridspec(2, 3, left=0.015, right=0.985, bottom=0.04, top=0.86)
    for mesh, title, color, position in zip(
        meshes, titles, colors, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    ):
        axis = figure.add_subplot(grid[position], projection="3d")
        viz.draw_surface(axis, mesh, color)
        axis.set_title(title, color=DARK, fontsize=10, pad=2)
    note = figure.add_subplot(grid[1, 2])
    note.axis("off")
    note.text(0.02, 0.95, "Complete-A mode comparison", color=DARK, weight="bold", fontsize=12, va="top")
    note.text(
        0.02,
        0.78,
        "One mode = one complete A and one q\n"
        "Three polarizations fixed internally\n"
        "Full 128 x 128 x 128 fluid grid\n"
        "No subsampling, ridge, or TF32",
        color=GRAY,
        fontsize=9.5,
        va="top",
        linespacing=1.45,
    )
    note.text(
        0.02,
        0.43,
        f"K=1024: velocity {errors1024[0]:.2%}\n"
        f"             vorticity {errors1024[1]:.2%}\n\n"
        f"K=2048: velocity {errors2048[0]:.2%}\n"
        f"             vorticity {errors2048[1]:.2%}",
        color=DARK,
        fontsize=10,
        va="top",
        linespacing=1.35,
    )
    figure.suptitle(
        "Strict FD-Lap complete-A reconstruction at a common absolute isovorticity",
        color=DARK,
        fontsize=14,
        weight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.91,
        f"All panels use |omega| = {level:.4f} (GT q96.5) and the same second-order FD curl.",
        ha="center",
        color=GRAY,
        fontsize=9.5,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=240, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--scalar-root", type=Path, default=SCALAR_ROOT)
    parser.add_argument("--basis-root", type=Path, default=BASIS_ROOT)
    parser.add_argument("--gt-root", type=Path, default=GT_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--scalar-batch", type=int, default=8)
    parser.add_argument("--transpose-row-chunk", type=int, default=4096)
    parser.add_argument("--mass-spatial-blocks", type=int, default=2)
    parser.add_argument("--reconstruction-spatial-blocks", type=int, default=2)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    scalar_root = args.scalar_root.resolve()
    basis_root = args.basis_root.resolve()
    gt_root = args.gt_root.resolve()
    output_root = args.output.resolve()
    basis_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    atomic_json(
        output_root / "protocol.json",
        {
            "status": "running",
            "mode_semantics": "one mode equals one complete A_k and one external q_k",
            "complete_A_count": COMPLETE_A_COUNT,
            "independent_coefficients": COMPLETE_A_COUNT,
            "one_q_per_complete_A": True,
            "polarizations_per_A": POLARIZATIONS_PER_A,
            "polarizations_are_internal": True,
            "polarization_selection": "lowest local 3x3 generalized velocity/curl Rayleigh eigenvector",
            "source_scalar_shapes": COMPLETE_A_COUNT,
            "source_scalar_shapes_distinct": True,
            "potential": "A_k=e phi_D,k c_k ~ d^2",
            "grid": [GRID] * 3,
            "fluid_nodes": FLUID_NODES,
            "all_fluid_nodes": True,
            "subsampling": False,
            "regularization": None,
            "tf32": False,
        },
    )

    metadata_path = scalar_root / "laplacian_scalar_metadata_128.json"
    scalar_path = scalar_root / "laplacian_scalar_eigenvectors_128.npy"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("complete") or int(metadata.get("scalar_modes", 0)) != COMPLETE_A_COUNT:
        raise RuntimeError("the 2048 distinct high-resolution scalar spectrum is not complete")
    residual_max = float(metadata.get("relative_residual_max", np.nan))
    if not np.isfinite(residual_max):
        batch_residuals = [
            float(batch["relative_residual_max"])
            for batch in metadata.get("batches", [])
            if "relative_residual_max" in batch
        ]
        if not batch_residuals:
            raise RuntimeError("completed scalar metadata has no residual audit")
        residual_max = max(batch_residuals)
    if residual_max > 1.0e-3:
        raise RuntimeError(f"scalar residual target violated: {residual_max:.3e}")

    cfg, geometry, mask, _, _ = core.previous.load_context(workspace)
    _, flat_indices = helper.cell_center_points(mask)
    if len(flat_indices) != FLUID_NODES:
        raise ValueError(f"unexpected fluid-node count {len(flat_indices)}")

    cell_cache = basis_root / "laplacian_scalar_cell_center_2048.npy"
    build_cell_scalar_cache_gpu(
        cfg,
        geometry,
        scalar_path,
        OLD_CELL_CACHE,
        cell_cache,
        basis_root / "laplacian_scalar_cell_center_progress.json",
    )
    scalar_mode_major = basis_root / "scalar_2048_mode_major_128.npy"
    core.build_scalar_mode_major(
        cell_cache,
        scalar_mode_major,
        basis_root / "scalar_mode_major_progress.json",
        mask.size,
    )
    envelope = single_zero_envelope(
        gt_root / "double_zero_envelope_e2_128.npy",
        basis_root / "basis_single_zero_envelope_e_128.npy",
    )
    gradient_mode_major = basis_root / "gradient_e_phiD_A2048_mode_major_128.npy"
    core.build_gradient_mode_major(
        scalar_mode_major,
        envelope,
        mask,
        flat_indices,
        gradient_mode_major,
        basis_root / "gradient_mode_major_progress.json",
    )

    velocity_mode_major = basis_root / "complete_velocity_A2048_mode_major_128.npy"
    _, polarizations, rayleigh, polarization_audit = build_complete_modes(
        gradient_mode_major,
        mask,
        flat_indices,
        velocity_mode_major,
        basis_root / "internal_polarizations_A2048.npz",
        basis_root / "complete_velocity_mode_progress.json",
        args.scalar_batch,
    )
    velocity_spatial_major = basis_root / "complete_velocity_A2048_spatial_major_128.npy"
    transpose_complete_basis(
        velocity_mode_major,
        velocity_spatial_major,
        basis_root / "complete_velocity_spatial_progress.json",
        args.transpose_row_chunk,
    )

    gt_velocity = np.load(gt_root / "gt_velocity_128.npy", mmap_mode="r")
    gt_vorticity = np.load(gt_root / "gt_vorticity_128.npy", mmap_mode="r")
    target = np.asarray(gt_velocity[mask], dtype=np.float32)
    mass, correlations, mass_audit = assemble_complete_mass(
        velocity_spatial_major,
        target,
        basis_root / "complete_A2048_mass_and_gt_correlation.npz",
        basis_root / "complete_A2048_mass_progress.json",
        args.mass_spatial_blocks,
    )
    q1024, projection1024 = solve_projection(mass, correlations, K_SMALL)
    q2048, projection2048 = solve_projection(mass, correlations, K_LARGE)
    np.save(output_root / "coefficients_q_complete_A1024.npy", q1024.astype(np.float32))
    np.save(output_root / "coefficients_q_complete_A2048.npy", q2048.astype(np.float32))
    np.save(output_root / "internal_polarizations_A2048x3.npy", polarizations.astype(np.float32))
    np.save(output_root / "local_rayleigh_A2048.npy", rayleigh.astype(np.float64))

    velocity1024, velocity2048 = reconstruct_prefixes(
        velocity_spatial_major,
        q1024,
        q2048,
        flat_indices,
        output_root,
        args.reconstruction_spatial_blocks,
    )
    vorticity1024 = curl_helper.curl(velocity1024, 1.0 / GRID)
    vorticity2048 = curl_helper.curl(velocity2048, 1.0 / GRID)
    vorticity1024[~mask] = 0.0
    vorticity2048[~mask] = 0.0
    np.save(output_root / "vorticity_common_fd_curl_complete_A1024_128.npy", vorticity1024)
    np.save(output_root / "vorticity_common_fd_curl_complete_A2048_128.npy", vorticity2048)

    errors1024 = (
        relative_l2(gt_velocity, velocity1024, mask),
        relative_l2(gt_vorticity, vorticity1024, mask),
    )
    errors2048 = (
        relative_l2(gt_velocity, velocity2048, mask),
        relative_l2(gt_vorticity, vorticity2048, mask),
    )
    level = float(np.quantile(viz.magnitude(gt_vorticity, mask)[mask], 0.965))
    png_path = output_root / "output" / "png" / "fd_lap_complete_A1024_A2048_common_isovorticity.png"
    pdf_path = output_root / "output" / "pdf" / "fd_lap_complete_A1024_A2048_common_isovorticity.pdf"
    render_comparison(
        mask,
        gt_vorticity,
        vorticity1024,
        vorticity2048,
        level,
        errors1024,
        errors2048,
        png_path,
        pdf_path,
    )

    metrics = [
        {
            "method": "FD-Lap complete-A",
            "modes": K_SMALL,
            "complete_A_count": K_SMALL,
            "independent_q": K_SMALL,
            "velocity_relative_l2": errors1024[0],
            "vorticity_relative_l2": errors1024[1],
        },
        {
            "method": "FD-Lap complete-A",
            "modes": K_LARGE,
            "complete_A_count": K_LARGE,
            "independent_q": K_LARGE,
            "velocity_relative_l2": errors2048[0],
            "vorticity_relative_l2": errors2048[1],
        },
    ]
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    atomic_json(output_root / "metrics.json", {"rows": metrics})

    latex = """\\begin{tabular}{lrrrr}
\\hline
Method & Modes & Complete $A$ & Velocity error & Vorticity error \\\\
\\hline
FD--Lap & 1024 & 1024 & %.4f\\%% & %.4f\\%% \\\\
FD--Lap & 2048 & 2048 & %.4f\\%% & %.4f\\%% \\\\
\\hline
\\end{tabular}
""" % (
        100.0 * errors1024[0],
        100.0 * errors1024[1],
        100.0 * errors2048[0],
        100.0 * errors2048[1],
    )
    (output_root / "comparison_table.tex").write_text(latex, encoding="utf-8")

    completion = {
        "complete": True,
        "status": "complete",
        "valid_for_requested_mode_semantics": True,
        "mode_semantics": "one mode = one complete A_k with one external q_k",
        "complete_A_count": COMPLETE_A_COUNT,
        "modes": COMPLETE_A_COUNT,
        "independent_coefficients": COMPLETE_A_COUNT,
        "one_q_per_complete_A": True,
        "polarizations_per_A": POLARIZATIONS_PER_A,
        "polarizations_are_internal": True,
        "internal_polarization": "fixed by the lowest eigenvector of each A_k local 3x3 generalized Rayleigh problem",
        "source_scalar_shapes": COMPLETE_A_COUNT,
        "source_scalar_shapes_distinct": True,
        "scalar_high_resolution_residual_max": residual_max,
        "potential": "A_k=e phi_D,k c_k ~ d^2",
        "grid": [GRID] * 3,
        "fluid_nodes": FLUID_NODES,
        "all_fluid_nodes": True,
        "subsampling": False,
        "regularization": None,
        "tf32": False,
        "polarization_audit": polarization_audit,
        "mass_audit": mass_audit,
        "projection_k1024": projection1024,
        "projection_k2048": projection2048,
        "errors_k1024": {
            "velocity_relative_l2": errors1024[0],
            "vorticity_relative_l2": errors1024[1],
        },
        "errors_k2048": {
            "velocity_relative_l2": errors2048[0],
            "vorticity_relative_l2": errors2048[1],
        },
        "vorticity_operator": "128^3 second-order finite-difference curl",
        "isosurface_common_level": level,
        "active_wall_seconds": float(time.perf_counter() - started),
        "artifacts": [str(png_path), str(pdf_path), str(output_root / "metrics.csv")],
    }
    atomic_json(output_root / "completion.json", completion)
    atomic_json(
        output_root / "protocol.json",
        {
            **json.loads((output_root / "protocol.json").read_text(encoding="utf-8")),
            "status": "complete",
        },
    )
    (output_root / "REPORT.md").write_text(
        "\n".join(
            [
                "# Strict FD-Lap complete-A K=2048 reconstruction",
                "",
                "- One mode is one complete antisymmetric tensor potential A and one q.",
                "- 2048 distinct scalar shapes; three polarizations are fixed internally by a local 3x3 eigenproblem.",
                "- Full 128^3 all-fluid-node projection; no subsampling, ridge, or TF32.",
                f"- K=1024 speed/vorticity errors: {errors1024[0]:.8%} / {errors1024[1]:.8%}.",
                f"- K=2048 speed/vorticity errors: {errors2048[0]:.8%} / {errors2048[1]:.8%}.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
