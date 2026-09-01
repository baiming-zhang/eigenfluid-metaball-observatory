"""Build the strict boundary-matched FD-Lap vector K=1024 basis.

Trial potentials use

    A_(k,alpha) = e * phi_D,k * E_alpha ~ d^2,

for 1024 distinct scalar shapes and three tensor polarizations (3072 trial
vectors).  The full 128^3 finite-difference curl stiffness and velocity mass
operators are assembled, one generalized Rayleigh-Ritz solve retains the
lowest 1024 complete vector modes, and the matched GT is projected with one
external q_k per retained mode.

Existing scalar shapes, e*phi gradients, velocity mass, and GT correlations
are reused.  The new work is resumable vorticity generation, stiffness
assembly, the 3072->1024 eigensolve, and final reconstruction.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap
import scipy.linalg as sla
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "reconstruct_three_dns_laplacian_double_zero_A1024_DOF3072.py"
)
LOW_WAVE_SCRIPT = PROJECT_ROOT / "scripts" / "generate_envelope_matched_gt_first_frame_128.py"
SOURCE_ROOT = (
    PROJECT_ROOT / "results" / "laplacian_true_double_zero_A1024_DOF3072_full128"
)
GT_ROOT = PROJECT_ROOT / "results" / "boundary_compatible_double_zero_gt_128_v2"
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fd_lap_strict_matched_e_phiD_A1024_vector_k1024_full128"
)
GRID = 128
VOXELS = GRID**3
FLUID_NODES = 1_599_754
SCALAR_SHAPES = 1024
POLARIZATIONS = 3
CANDIDATES = SCALAR_SHAPES * POLARIZATIONS
K = 1024


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


context = load_module("strict_matched_fd_context", CONTEXT_SCRIPT)
low_wave = load_module("strict_matched_fd_low_wave", LOW_WAVE_SCRIPT)
helper = context.helper


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
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.float32:
            raise ValueError(f"unexpected cache {path}: {array.shape} {array.dtype}")
        return array
    return open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def block_ranges(size: int, count: int) -> list[tuple[int, int]]:
    return [
        (size * index // count, size * (index + 1) // count)
        for index in range(count)
    ]


def trim_windows_working_set() -> None:
    """Release clean memmap pages after a durable transpose checkpoint."""
    if os.name != "nt":
        return
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.EmptyWorkingSet(handle)


def build_candidate_vorticity_mode_major(
    gradient_path: Path,
    mask: np.ndarray,
    flat_indices: np.ndarray,
    output_path: Path,
    progress_path: Path,
    scalar_batch: int,
) -> np.memmap:
    output = open_or_create(output_path, (CANDIDATES, FLUID_NODES, 3))
    next_mode = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            log("reuse complete strict e*phi_D candidate vorticity cache")
            return np.load(output_path, mmap_mode="r")
        next_mode = int(progress.get("next_scalar_mode", 0))
    gradient = np.load(gradient_path, mmap_mode="r")
    if gradient.shape != (SCALAR_SHAPES, FLUID_NODES, 3):
        raise ValueError(f"unexpected gradient cache {gradient.shape}")
    spacing = 1.0 / GRID
    for start in range(next_mode, SCALAR_SHAPES, scalar_batch):
        stop = min(start + scalar_batch, SCALAR_SHAPES)
        width = stop - start
        volume = np.zeros((VOXELS, 3, width), dtype=np.float32)
        volume[flat_indices] = np.asarray(gradient[start:stop]).transpose(1, 2, 0)
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

        block12 = np.empty((width, FLUID_NODES, 3), dtype=np.float32)
        block12[:, :, 0] = dx_z[mask].T
        block12[:, :, 1] = dy_z[mask].T
        block12[:, :, 2] = (-(dx_x + dy_y))[mask].T
        block13 = np.empty_like(block12)
        block13[:, :, 0] = (-dx_y)[mask].T
        block13[:, :, 1] = (dz_z + dx_x)[mask].T
        block13[:, :, 2] = (-dz_y)[mask].T
        block23 = np.empty_like(block12)
        block23[:, :, 0] = (-(dy_y + dz_z))[mask].T
        block23[:, :, 1] = dy_x[mask].T
        block23[:, :, 2] = dz_x[mask].T
        output[start:stop] = block12
        output[SCALAR_SHAPES + start : SCALAR_SHAPES + stop] = block13
        output[2 * SCALAR_SHAPES + start : 2 * SCALAR_SHAPES + stop] = block23
        output.flush()
        del (
            block23,
            block13,
            block12,
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
        )
        atomic_json(
            progress_path,
            {
                "complete": stop == SCALAR_SHAPES,
                "next_scalar_mode": stop,
                "scalar_shapes": SCALAR_SHAPES,
                "polarizations": POLARIZATIONS,
                "candidate_vector_dofs": CANDIDATES,
                "potential": "e phi_D E_alpha ~ d^2",
                "curl": "128^3 second-order finite difference",
            },
        )
        log(f"candidate vorticity scalar modes {stop}/{SCALAR_SHAPES}")
    return np.load(output_path, mmap_mode="r")


def transpose_candidate_vorticity(
    mode_major_path: Path,
    output_path: Path,
    progress_path: Path,
    row_chunk: int,
) -> np.memmap:
    source = np.load(mode_major_path, mmap_mode="r")
    if source.shape != (CANDIDATES, FLUID_NODES, 3):
        raise ValueError(f"unexpected vorticity mode-major cache {source.shape}")
    output = open_or_create(output_path, (FLUID_NODES, 3, CANDIDATES))
    next_row = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            log("reuse complete strict candidate vorticity spatial-major cache")
            return np.load(output_path, mmap_mode="r")
        next_row = int(progress.get("next_row", 0))
    for start in range(next_row, FLUID_NODES, row_chunk):
        stop = min(start + row_chunk, FLUID_NODES)
        # Copy one complete tensor polarization at a time.  The old direct
        # 3072-wide strided assignment retained tens of GiB of mapped pages
        # and eventually thrashed the Windows working set.  These three
        # 1024-wide blocks preserve exactly the same spatial-major layout and
        # make both the source read and destination write page-local.
        for candidate_start in range(0, CANDIDATES, SCALAR_SHAPES):
            candidate_stop = candidate_start + SCALAR_SHAPES
            block = np.array(
                source[candidate_start:candidate_stop, start:stop, :],
                dtype=np.float32,
                order="C",
                copy=True,
            )
            output[start:stop, :, candidate_start:candidate_stop] = block.transpose(
                1, 2, 0
            )
            del block
        if stop == FLUID_NODES or stop % (row_chunk * 8) == 0:
            output.flush()
            atomic_json(
                progress_path,
                {
                    "complete": stop == FLUID_NODES,
                    "next_row": stop,
                    "candidate_vector_dofs": CANDIDATES,
                    "layout": "fluid-node x vector-component x candidate",
                },
            )
            log(f"candidate vorticity spatial rows {stop}/{FLUID_NODES}")
            trim_windows_working_set()
    return np.load(output_path, mmap_mode="r")


def assemble_stiffness(
    vorticity_path: Path,
    output_path: Path,
    progress_path: Path,
    spatial_blocks: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    vorticity = np.load(vorticity_path, mmap_mode="r")
    if vorticity.shape != (FLUID_NODES, 3, CANDIDATES):
        raise ValueError(f"unexpected vorticity spatial cache {vorticity.shape}")
    stiffness = np.zeros((CANDIDATES, CANDIDATES), dtype=np.float64)
    next_block = 0
    partial_path = output_path.with_name(output_path.stem + "_partial.npz")
    if progress_path.exists() and partial_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        next_block = int(progress.get("next_block", 0))
        with np.load(partial_path, allow_pickle=False) as cached:
            stiffness = np.asarray(cached["stiffness"], dtype=np.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    weight = (1.0 / GRID) ** 3
    ranges = block_ranges(FLUID_NODES, spatial_blocks)
    for block_index, (start, stop) in enumerate(ranges):
        if block_index < next_block:
            continue
        gib = (stop - start) * 3 * CANDIDATES * 4 / 1024**3
        log(
            f"stiffness block {block_index + 1}/{spatial_blocks}: "
            f"vorticity basis={gib:.2f} GiB"
        )
        values = torch.tensor(
            np.asarray(vorticity[start:stop]), device="cuda", dtype=torch.float32
        ).reshape(-1, CANDIDATES)
        stiffness += (weight * (values.T @ values)).double().cpu().numpy()
        del values
        torch.cuda.empty_cache()
        atomic_npz(partial_path, stiffness=stiffness)
        atomic_json(
            progress_path,
            {
                "complete": block_index + 1 == spatial_blocks,
                "next_block": block_index + 1,
                "spatial_blocks": spatial_blocks,
                "candidate_vector_dofs": CANDIDATES,
                "all_fluid_nodes": True,
                "tf32": False,
            },
        )
    stiffness = 0.5 * (stiffness + stiffness.T)
    audit = {
        "trace": float(np.trace(stiffness)),
        "frobenius_norm": float(np.linalg.norm(stiffness)),
        "symmetry_max_abs": float(np.max(np.abs(stiffness - stiffness.T))),
        "dimension": CANDIDATES,
    }
    atomic_npz(output_path, stiffness=stiffness, audit_json=np.asarray(json.dumps(audit)))
    return stiffness, audit


def generalized_modes(
    mass: np.ndarray,
    stiffness: np.ndarray,
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as cached:
            mixing = np.asarray(cached["mixing"], dtype=np.float64)
            eigenvalues = np.asarray(cached["eigenvalues"], dtype=np.float64)
            audit = json.loads(str(cached["audit_json"].item()))
        if mixing.shape == (CANDIDATES, K) and eigenvalues.shape == (K,):
            log("reuse complete strict generalized eigensolve")
            return mixing, eigenvalues, audit
    eigenvalues, mixing = sla.eigh(
        stiffness,
        mass,
        subset_by_index=(0, K - 1),
        driver="gvx",
        check_finite=False,
    )
    final_mass = mixing.T @ mass @ mixing
    residual = stiffness @ mixing - (mass @ mixing) * eigenvalues[None, :]
    denominator = max(
        np.linalg.norm(stiffness @ mixing),
        np.linalg.norm((mass @ mixing) * eigenvalues[None, :]),
        1.0e-300,
    )
    audit = {
        "eigenvalue_min": float(eigenvalues[0]),
        "eigenvalue_max": float(eigenvalues[-1]),
        "mass_orthogonality_max_abs": float(
            np.max(np.abs(final_mass - np.eye(K)))
        ),
        "generalized_eigen_relative_residual": float(
            np.linalg.norm(residual) / denominator
        ),
        "trial_candidates": CANDIDATES,
        "retained_complete_vector_modes": K,
    }
    atomic_npz(
        output_path,
        mixing=mixing,
        eigenvalues=eigenvalues,
        audit_json=np.asarray(json.dumps(audit)),
    )
    return mixing, eigenvalues, audit


def project_and_reconstruct(
    gradient_path: Path,
    mass: np.ndarray,
    correlations: np.ndarray,
    mixing: np.ndarray,
    mask: np.ndarray,
    flat_indices: np.ndarray,
    output_root: Path,
    spatial_blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    final_mass = mixing.T @ mass @ mixing
    final_correlations = correlations @ mixing
    q = sla.solve(
        final_mass,
        final_correlations.T,
        assume_a="sym",
        check_finite=False,
    ).ravel()
    candidate_coefficients = mixing @ q
    residual = float(
        np.linalg.norm(final_mass @ q - final_correlations.ravel())
        / max(np.linalg.norm(final_correlations), 1.0e-300)
    )
    np.save(output_root / "coefficients_q_k1024.npy", q.astype(np.float32))
    np.save(
        output_root / "candidate_coefficients_3072.npy",
        candidate_coefficients.astype(np.float32),
    )
    gradient = np.load(gradient_path, mmap_mode="r")
    if gradient.shape != (FLUID_NODES, 3, SCALAR_SHAPES):
        raise ValueError(f"unexpected gradient spatial cache {gradient.shape}")
    output = np.zeros((VOXELS, 3), dtype=np.float32)
    r = torch.tensor(candidate_coefficients.astype(np.float32), device="cuda")
    r12 = r[:SCALAR_SHAPES]
    r13 = r[SCALAR_SHAPES : 2 * SCALAR_SHAPES]
    r23 = r[2 * SCALAR_SHAPES :]
    for block_index, (start, stop) in enumerate(
        block_ranges(FLUID_NODES, spatial_blocks)
    ):
        values = torch.tensor(
            np.asarray(gradient[start:stop]), device="cuda", dtype=torch.float32
        )
        dx, dy, dz = values[:, 0], values[:, 1], values[:, 2]
        ux = dy @ r12 + dz @ r13
        uy = -dx @ r12 + dz @ r23
        uz = -dx @ r13 - dy @ r23
        reconstructed = torch.stack((ux, uy, uz), dim=1).cpu().numpy()
        output[flat_indices[start:stop]] = reconstructed
        del reconstructed, ux, uy, uz, dx, dy, dz, values
        torch.cuda.empty_cache()
        log(f"final reconstruction block {block_index + 1}/{spatial_blocks}")
    velocity = output.reshape((GRID, GRID, GRID, 3))
    vorticity = low_wave.curl(velocity, 1.0 / GRID)
    vorticity[~mask] = 0.0
    np.save(output_root / "velocity_k1024_128.npy", velocity)
    np.save(output_root / "vorticity_common_fd_curl_k1024_128.npy", vorticity)
    audit = {
        "final_mass_orthogonality_max_abs": float(
            np.max(np.abs(final_mass - np.eye(K)))
        ),
        "normal_equation_relative_residual": residual,
        "q_shape": list(q.shape),
        "independent_coefficients": K,
        "one_q_per_complete_vector_mode": True,
    }
    return velocity, vorticity, q, audit


def relative_l2(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    ref = np.asarray(reference[mask], dtype=np.float64)
    got = np.asarray(estimate[mask], dtype=np.float64)
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--gt-root", type=Path, default=GT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--scalar-batch", type=int, default=8)
    parser.add_argument("--transpose-row-chunk", type=int, default=4096)
    parser.add_argument("--stiffness-spatial-blocks", type=int, default=3)
    parser.add_argument("--reconstruction-spatial-blocks", type=int, default=2)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    source_root = args.source_root.resolve()
    gt_root = args.gt_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    atomic_json(
        output_root / "protocol.json",
        {
            "status": "running",
            "potential": "A_(k,alpha)=e phi_D,k E_alpha ~ d^2",
            "source_scalar_shapes": SCALAR_SHAPES,
            "trial_vector_candidates": CANDIDATES,
            "retained_complete_vector_modes": K,
            "independent_coefficients": K,
            "one_q_per_complete_vector_mode": True,
            "grid": [GRID] * 3,
            "fluid_nodes": FLUID_NODES,
            "stiffness_spatial_blocks": args.stiffness_spatial_blocks,
            "minimum_blocks_for_24GiB": True,
            "tf32": False,
            "subsampling": False,
        },
    )
    _, _, mask, _, _ = context.previous.load_context(workspace)
    _, flat_indices = helper.cell_center_points(mask)
    if len(flat_indices) != FLUID_NODES:
        raise ValueError(f"unexpected fluid-node count {len(flat_indices)}")
    gradient_mode = source_root / "gradient_true_double_zero_A1024_mode_major_128.npy"
    gradient_spatial = source_root / "gradient_true_double_zero_A1024_spatial_major_128.npy"
    mass_cache = source_root / "mass_and_correlations_boundary_compatible_gt_DOF3072.npz"
    for path in (gradient_mode, gradient_spatial, mass_cache):
        if not path.exists():
            raise FileNotFoundError(path)
    with np.load(mass_cache, allow_pickle=False) as cached:
        mass = np.asarray(cached["mass"], dtype=np.float64)
        correlations = np.asarray(cached["correlations"], dtype=np.float64)
    if mass.shape != (CANDIDATES, CANDIDATES) or correlations.shape != (1, CANDIDATES):
        raise ValueError("unexpected reused mass/correlation dimensions")

    vorticity_mode = output_root / "candidate_vorticity_3072_mode_major_128.npy"
    build_candidate_vorticity_mode_major(
        gradient_mode,
        mask,
        flat_indices,
        vorticity_mode,
        output_root / "candidate_vorticity_mode_progress.json",
        args.scalar_batch,
    )
    vorticity_spatial = output_root / "candidate_vorticity_3072_spatial_major_128.npy"
    transpose_candidate_vorticity(
        vorticity_mode,
        vorticity_spatial,
        output_root / "candidate_vorticity_spatial_progress.json",
        args.transpose_row_chunk,
    )
    stiffness, stiffness_audit = assemble_stiffness(
        vorticity_spatial,
        output_root / "stiffness_3072.npz",
        output_root / "stiffness_progress.json",
        args.stiffness_spatial_blocks,
    )
    mixing, eigenvalues, eigen_audit = generalized_modes(
        mass,
        stiffness,
        output_root / "vector_modes_k1024.npz",
    )
    velocity, vorticity, _, projection_audit = project_and_reconstruct(
        gradient_spatial,
        mass,
        correlations,
        mixing,
        mask,
        flat_indices,
        output_root,
        args.reconstruction_spatial_blocks,
    )
    gt_velocity = np.load(gt_root / "gt_velocity_128.npy", mmap_mode="r")
    gt_vorticity = np.load(gt_root / "gt_vorticity_128.npy", mmap_mode="r")
    velocity_error = relative_l2(gt_velocity, velocity, mask)
    vorticity_error = relative_l2(gt_vorticity, vorticity, mask)
    completion = {
        "complete": True,
        "status": "numerical_complete_render_pending",
        "method": "strict matched FD-Lap vector Rayleigh-Ritz",
        "potential": "A_(k,alpha)=e phi_D,k E_alpha ~ d^2",
        "double_zero_potential": True,
        "grid": [GRID] * 3,
        "fluid_nodes": FLUID_NODES,
        "source_scalar_shapes": SCALAR_SHAPES,
        "trial_vector_candidates": CANDIDATES,
        "retained_complete_vector_modes": K,
        "independent_coefficients": K,
        "one_q_per_complete_vector_mode": True,
        "internal_polarization": "fixed by each generalized eigenvector",
        "mass_source_reused": str(mass_cache),
        "scalar_eigenfunctions_recomputed": False,
        "vector_rayleigh_ritz_recomputed": True,
        "stiffness_audit": stiffness_audit,
        "eigen_audit": eigen_audit,
        "projection_audit": projection_audit,
        "projection": {
            "all_fluid_nodes": True,
            "subsampling": False,
            "tf32": False,
            "regularization": None,
            "vorticity_operator": "128^3 second-order finite-difference curl",
        },
        "errors": {
            "velocity_relative_l2": velocity_error,
            "vorticity_relative_l2": vorticity_error,
            "velocity_energy_retained": float(1.0 - velocity_error**2),
        },
        "eigenvalue_range": [float(eigenvalues[0]), float(eigenvalues[-1])],
        "active_wall_seconds": float(time.perf_counter() - started),
    }
    atomic_json(output_root / "completion.json", completion)
    atomic_json(
        output_root / "protocol.json",
        {
            **json.loads((output_root / "protocol.json").read_text(encoding="utf-8")),
            "status": "numerical_complete_render_pending",
        },
    )
    log(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
