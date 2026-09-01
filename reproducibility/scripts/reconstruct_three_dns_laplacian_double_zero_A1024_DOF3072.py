"""Reconstruct three DNS first frames with 1024 complete tensor potentials.

For every one of the first 1024 distinct scalar Dirichlet-Laplacian shapes
``phi_k`` this script retains all three antisymmetric tensor-potential
components A12, A13, and A23.  The result is exactly 1024 tensor-potential
groups and 3072 independent scalar coefficients.  No Rayleigh-Ritz selection,
peeling, or spectral truncation is performed after these 3072 DOFs are built.

The exact squared transfer envelope is applied before differentiation.  Only
the first derivatives needed for velocity are evaluated.  All mass and target
inner products use every fluid node on the 128^3 grid with TF32 disabled.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
import scipy.linalg as sla
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "reconstruct_three_dns_laplacian_double_zero_vector_k1024.py"
)
DEFAULT_SCALAR_CACHE = (
    PROJECT_ROOT
    / "results"
    / "laplacian_128_scalar_k1024"
    / "basis_full128_one_to_one"
    / "laplacian_scalar_cell_center_128.npy"
)
DEFAULT_SCALAR_METADATA = (
    PROJECT_ROOT
    / "results"
    / "laplacian_128_scalar_k1024"
    / "laplacian_scalar_metadata_128.json"
)
DEFAULT_BASIS_ROOT = (
    PROJECT_ROOT / "results" / "laplacian_double_zero_A1024_DOF3072_full128"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "reconstruction_three_dns_laplacian_double_zero_A1024_DOF3072_full128"
)

RESOLUTION = 128
A_GROUPS = 1024
TENSOR_COMPONENTS = 3
SCALAR_DOFS = A_GROUPS * TENSOR_COMPONENTS
CASES = (
    ("case_01_independent_dns", "Independent DNS", "independent_dns_gt_128"),
    ("case_02_large_eddy", "Large-eddy TG/ABC DNS", "independent_dns_large_eddy_gt_128"),
    (
        "case_03_natural_large_eddy",
        "Natural low-wave DNS",
        "independent_dns_natural_large_eddy_gt_128",
    ),
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


previous = load_module("laplacian_double_zero_previous", PREVIOUS_SCRIPT)
helper = previous.helper
render = previous.render


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
            time.sleep(0.1)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    replace_with_retry(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    replace_with_retry(temporary, path)


def open_or_create(path: Path, shape: tuple[int, ...]) -> np.memmap:
    if path.exists():
        result = np.load(path, mmap_mode="r+")
        if result.shape != shape or result.dtype != np.float32:
            raise ValueError(f"unexpected cache {path}: {result.shape} {result.dtype}")
        return result
    path.parent.mkdir(parents=True, exist_ok=True)
    return open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def block_ranges(count: int, blocks: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, count, blocks + 1, dtype=np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(blocks)]


def build_scalar_mode_major(
    scalar_path: Path,
    output_path: Path,
    progress_path: Path,
    grid_points: int,
) -> np.memmap:
    output = open_or_create(output_path, (A_GROUPS, grid_points))
    next_row = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            log("reuse complete 1024-shape mode-major scalar cache")
            return np.load(output_path, mmap_mode="r")
        next_row = int(progress.get("next_row", 0))
    source = np.load(scalar_path, mmap_mode="r")
    if source.shape != (grid_points, A_GROUPS):
        raise ValueError(f"expected exactly 1024 distinct scalar shapes, got {source.shape}")
    row_chunk = 65_536
    checkpoint_rows = row_chunk * 8
    for start in range(next_row, grid_points, row_chunk):
        stop = min(start + row_chunk, grid_points)
        output[:, start:stop] = np.asarray(source[start:stop]).T
        checkpoint = ((stop - next_row) % checkpoint_rows == 0) or stop == grid_points
        if checkpoint:
            output.flush()
            atomic_json(
                progress_path,
                {
                    "complete": stop == grid_points,
                    "next_row": stop,
                    "distinct_scalar_shapes": A_GROUPS,
                    "grid_points": grid_points,
                    "layout": "scalar-shape-major",
                    "checkpoint_rows": checkpoint_rows,
                },
            )
            log(f"scalar mode-major rows {stop}/{grid_points} (checkpoint committed)")
        elif (stop - next_row) % (row_chunk * 2) == 0:
            log(f"scalar mode-major rows {stop}/{grid_points} (staged)")
    return np.load(output_path, mmap_mode="r")


def build_gradient_mode_major(
    scalar_path: Path,
    envelope: np.ndarray,
    mask: np.ndarray,
    flat_indices: np.ndarray,
    output_path: Path,
    progress_path: Path,
) -> np.memmap:
    output = open_or_create(output_path, (A_GROUPS, len(flat_indices), 3))
    next_mode = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            log("reuse complete double-zero gradient basis for 1024 A groups")
            return np.load(output_path, mmap_mode="r")
        next_mode = int(progress.get("next_mode", 0))
    scalar = np.load(scalar_path, mmap_mode="r")
    if scalar.shape != (A_GROUPS, mask.size):
        raise ValueError(f"unexpected scalar cache {scalar.shape}")
    envelope_flat = np.asarray(envelope).ravel()
    spacing = 1.0 / RESOLUTION
    for mode in range(next_mode, A_GROUPS):
        potential = (
            np.asarray(scalar[mode], dtype=np.float32) * envelope_flat
        ).reshape(mask.shape)
        derivative_z, derivative_y, derivative_x = np.gradient(
            potential, spacing, edge_order=2
        )
        output[mode, :, 0] = derivative_x.ravel()[flat_indices]
        output[mode, :, 1] = derivative_y.ravel()[flat_indices]
        output[mode, :, 2] = derivative_z.ravel()[flat_indices]
        checkpoint = ((mode + 1 - next_mode) % 256 == 0) or mode + 1 == A_GROUPS
        if checkpoint:
            output.flush()
            atomic_json(
                progress_path,
                {
                    "complete": mode + 1 == A_GROUPS,
                    "next_mode": mode + 1,
                    "tensor_A_groups": A_GROUPS,
                    "independent_components_per_A": TENSOR_COMPONENTS,
                    "scalar_DOFs": SCALAR_DOFS,
                    "double_zero": True,
                    "potential_derivative_order": 1,
                    "jet3": False,
                    "checkpoint_modes": 256,
                },
            )
            log(f"double-zero A gradients {mode + 1}/{A_GROUPS} (checkpoint committed)")
        elif (mode + 1 - next_mode) % 32 == 0:
            log(f"double-zero A gradients {mode + 1}/{A_GROUPS} (staged)")
    return np.load(output_path, mmap_mode="r")


def transpose_gradient_basis(
    mode_major_path: Path,
    output_path: Path,
    progress_path: Path,
    fluid_nodes: int,
) -> np.memmap:
    source = np.load(mode_major_path, mmap_mode="r")
    if source.shape != (A_GROUPS, fluid_nodes, 3):
        raise ValueError(f"unexpected mode-major gradients {source.shape}")
    output = open_or_create(output_path, (fluid_nodes, 3, A_GROUPS))
    next_row = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            log("reuse complete spatial-major gradient basis")
            return np.load(output_path, mmap_mode="r")
        next_row = int(progress.get("next_row", 0))
    row_chunk = 8192
    for start in range(next_row, fluid_nodes, row_chunk):
        stop = min(start + row_chunk, fluid_nodes)
        output[start:stop] = np.asarray(source[:, start:stop, :]).transpose(1, 2, 0)
        if stop == fluid_nodes or stop % (row_chunk * 16) == 0:
            output.flush()
            atomic_json(
                progress_path,
                {
                    "complete": stop == fluid_nodes,
                    "next_row": stop,
                    "tensor_A_groups": A_GROUPS,
                    "gradient_channels": ["dx", "dy", "dz"],
                },
            )
            log(f"spatial-major gradient rows {stop}/{fluid_nodes}")
    return np.load(output_path, mmap_mode="r")


def assemble_mass_and_correlations(
    gradient_path: Path,
    targets: np.ndarray,
    output_path: Path,
    progress_path: Path,
    spatial_blocks: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if output_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            with np.load(output_path, allow_pickle=False) as cached:
                return cached["mass"], cached["correlations"], json.loads(
                    str(cached["audit_json"].item())
                )

    gradient = np.load(gradient_path, mmap_mode="r")
    if gradient.shape[1:] != (3, A_GROUPS):
        raise ValueError(f"unexpected spatial-major gradients {gradient.shape}")
    ranges = block_ranges(gradient.shape[0], spatial_blocks)
    gram = np.zeros((3, 3, A_GROUPS, A_GROUPS), dtype=np.float64)
    target_count = int(targets.shape[0])
    component_correlations = np.zeros(
        (target_count, TENSOR_COMPONENTS, A_GROUPS), dtype=np.float64
    )
    partial_path = output_path.with_name(output_path.stem + "_partial.npz")
    next_block = 0
    if partial_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        next_block = int(progress.get("next_block", 0))
        with np.load(partial_path, allow_pickle=False) as cached:
            gram = cached["gram"]
            component_correlations = cached["component_correlations"]

    if not torch.cuda.is_available():
        raise RuntimeError("the all-node 3072-DOF mass matrix requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    weight = (1.0 / RESOLUTION) ** 3
    for block_index, (start, stop) in enumerate(ranges):
        if block_index < next_block:
            continue
        gib = (stop - start) * 3 * A_GROUPS * 4 / 1024**3
        log(
            f"GPU full-gradient mass block {block_index + 1}/{spatial_blocks}: "
            f"basis={gib:.2f} GiB, all 3072 DOFs retained"
        )
        values = torch.as_tensor(
            np.asarray(gradient[start:stop]), device="cuda", dtype=torch.float32
        )
        dx, dy, dz = values[:, 0], values[:, 1], values[:, 2]
        channels = (dx, dy, dz)
        for first in range(3):
            for second in range(first, 3):
                product = weight * (channels[first].T @ channels[second])
                block = product.double().cpu().numpy()
                gram[first, second] += block
                if first != second:
                    gram[second, first] += block.T
                del product, block

        target_tensor = torch.as_tensor(
            np.asarray(targets[:, start:stop]), device="cuda", dtype=torch.float32
        )
        tx, ty, tz = (
            target_tensor[:, :, 0],
            target_tensor[:, :, 1],
            target_tensor[:, :, 2],
        )
        b12 = weight * ((dy.T @ tx.T) - (dx.T @ ty.T))
        b13 = weight * ((dz.T @ tx.T) - (dx.T @ tz.T))
        b23 = weight * ((dz.T @ ty.T) - (dy.T @ tz.T))
        component_correlations[:, 0] += b12.T.double().cpu().numpy()
        component_correlations[:, 1] += b13.T.double().cpu().numpy()
        component_correlations[:, 2] += b23.T.double().cpu().numpy()
        del b12, b13, b23, tx, ty, tz, target_tensor, channels, dx, dy, dz, values
        torch.cuda.empty_cache()
        atomic_npz(
            partial_path,
            gram=gram,
            component_correlations=component_correlations,
        )
        atomic_json(
            progress_path,
            {
                "complete": block_index + 1 == spatial_blocks,
                "next_block": block_index + 1,
                "spatial_blocks": spatial_blocks,
                "grid": [RESOLUTION] * 3,
                "all_fluid_nodes": True,
                "tensor_A_groups": A_GROUPS,
                "scalar_DOFs": SCALAR_DOFS,
                "selection": "none",
                "subsampling": False,
                "tf32": False,
            },
        )

    gxx, gxy, gxz = gram[0, 0], gram[0, 1], gram[0, 2]
    gyy, gyz, gzz = gram[1, 1], gram[1, 2], gram[2, 2]
    mass = np.block(
        [
            [gyy + gxx, gyz, -gxz],
            [gyz.T, gzz + gxx, gxy],
            [-gxz.T, gxy.T, gzz + gyy],
        ]
    )
    mass = 0.5 * (mass + mass.T)
    correlations = np.concatenate(
        [component_correlations[:, i] for i in range(3)], axis=1
    )
    eigenvalues = sla.eigvalsh(mass, check_finite=False, driver="evd")
    threshold = float(max(eigenvalues[-1], 1.0e-300) * 1.0e-10)
    audit = {
        "mass_min_eigenvalue": float(eigenvalues[0]),
        "mass_max_eigenvalue": float(eigenvalues[-1]),
        "mass_condition_number_abs": float(
            abs(eigenvalues[-1]) / max(abs(eigenvalues[0]), 1.0e-300)
        ),
        "mass_numerical_rank_1e_minus_10": int(np.count_nonzero(eigenvalues > threshold)),
        "mass_dimension": SCALAR_DOFS,
        "no_mode_selection": True,
    }
    atomic_npz(
        output_path,
        mass=mass,
        correlations=correlations,
        audit_json=np.asarray(json.dumps(audit)),
    )
    return mass, correlations, audit


def solve_all_dofs(
    mass: np.ndarray,
    correlations: np.ndarray,
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    target_count = int(correlations.shape[0])
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as cached:
            flat = np.asarray(cached["coefficients_flat"], dtype=np.float64)
            grouped = np.asarray(cached["coefficients_A1024x3"], dtype=np.float64)
            audit = json.loads(str(cached["audit_json"].item()))
        if flat.shape == (target_count, SCALAR_DOFS) and grouped.shape == (
            target_count, A_GROUPS, TENSOR_COMPONENTS
        ):
            return flat, grouped, audit
    coefficients_flat = sla.solve(
        mass,
        correlations.T,
        assume_a="sym",
        check_finite=False,
    ).T
    residual = float(
        np.linalg.norm(coefficients_flat @ mass - correlations)
        / max(np.linalg.norm(correlations), 1.0e-300)
    )
    coefficients_grouped = np.stack(
        (
            coefficients_flat[:, 0:A_GROUPS],
            coefficients_flat[:, A_GROUPS : 2 * A_GROUPS],
            coefficients_flat[:, 2 * A_GROUPS : 3 * A_GROUPS],
        ),
        axis=2,
    )
    audit = {
        "normal_equation_relative_residual": residual,
        "coefficient_shape": list(coefficients_grouped.shape),
        "all_3072_DOFs_used": True,
        "regularization": None,
        "selection": "none",
    }
    atomic_npz(
        output_path,
        coefficients_flat=coefficients_flat,
        coefficients_A1024x3=coefficients_grouped,
        audit_json=np.asarray(json.dumps(audit)),
    )
    return coefficients_flat, coefficients_grouped, audit


def reconstruct_all_dofs(
    gradient_path: Path,
    coefficients_grouped: np.ndarray,
    mask: np.ndarray,
    flat_indices: np.ndarray,
    output_path: Path,
    progress_path: Path,
    spatial_blocks: int,
) -> np.memmap:
    target_count = int(coefficients_grouped.shape[0])
    expected = (target_count, RESOLUTION, RESOLUTION, RESOLUTION, 3)
    output = open_or_create(output_path, expected)
    next_block = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("complete"):
            log("reuse complete A1024/DOF3072 reconstructed velocity")
            return np.load(output_path, mmap_mode="r")
        next_block = int(progress.get("next_block", 0))
    gradient = np.load(gradient_path, mmap_mode="r")
    ranges = block_ranges(len(flat_indices), spatial_blocks)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    coefficients = torch.as_tensor(
        coefficients_grouped.astype(np.float32), device="cuda"
    )
    c12, c13, c23 = coefficients[:, :, 0], coefficients[:, :, 1], coefficients[:, :, 2]
    output_flat = output.reshape(target_count, mask.size, 3)
    if next_block == 0:
        output[:] = 0.0
    for block_index, (start, stop) in enumerate(ranges):
        if block_index < next_block:
            continue
        gib = (stop - start) * 3 * A_GROUPS * 4 / 1024**3
        log(
            f"GPU A1024/DOF3072 reconstruction block {block_index + 1}/"
            f"{spatial_blocks}: gradient={gib:.2f} GiB"
        )
        values = torch.as_tensor(
            np.asarray(gradient[start:stop]), device="cuda", dtype=torch.float32
        )
        dx, dy, dz = values[:, 0], values[:, 1], values[:, 2]
        ux = dy @ c12.T + dz @ c13.T
        uy = -dx @ c12.T + dz @ c23.T
        uz = -dx @ c13.T - dy @ c23.T
        reconstructed = torch.stack((ux, uy, uz), dim=2)
        output_flat[:, flat_indices[start:stop], :] = (
            reconstructed.permute(1, 0, 2).cpu().numpy()
        )
        output.flush()
        del reconstructed, ux, uy, uz, dx, dy, dz, values
        torch.cuda.empty_cache()
        atomic_json(
            progress_path,
            {
                "complete": block_index + 1 == spatial_blocks,
                "next_block": block_index + 1,
                "spatial_blocks": spatial_blocks,
                "tensor_A_groups": A_GROUPS,
                "scalar_DOFs": SCALAR_DOFS,
                "all_DOFs_used": True,
                "tf32": False,
            },
        )
    return np.load(output_path, mmap_mode="r")


def relative_l2(reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    a = np.asarray(reference[mask], dtype=np.float64)
    b = np.asarray(estimate[mask], dtype=np.float64)
    return float(np.linalg.norm(b - a) / np.linalg.norm(a))


def render_results(
    workspace: Path,
    output: Path,
    mask: np.ndarray,
    velocity: np.ndarray,
    vorticity: np.ndarray,
    coefficients: np.ndarray,
    gt_vorticity: list[np.ndarray],
) -> tuple[list[dict[str, Any]], list[Path]]:
    artifacts: list[Path] = []
    metrics: list[dict[str, Any]] = []
    surfaces: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float]] = {}
    for index, (case_id, label, source_name) in enumerate(CASES):
        source = workspace / "results" / source_name
        gt_velocity = np.load(source / "velocity" / "frame_000.npy", mmap_mode="r")
        gt_omega = gt_vorticity[index]
        velocity_error = relative_l2(gt_velocity, velocity[index], mask)
        vorticity_error = relative_l2(gt_omega, vorticity[index], mask)
        case_dir = output / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        velocity_path = case_dir / "velocity_A1024_DOF3072_128.npy"
        vorticity_path = case_dir / "vorticity_A1024_DOF3072_128.npy"
        coefficient_path = case_dir / "coefficients_A1024x3.npy"
        np.save(velocity_path, velocity[index])
        np.save(vorticity_path, vorticity[index])
        np.save(coefficient_path, coefficients[index].astype(np.float32))
        artifacts.extend((velocity_path, vorticity_path, coefficient_path))

        gt_magnitude = np.linalg.norm(np.asarray(gt_omega), axis=-1).astype(np.float32)
        recon_magnitude = np.linalg.norm(np.asarray(vorticity[index]), axis=-1).astype(np.float32)
        error_magnitude = np.linalg.norm(
            np.asarray(vorticity[index]) - np.asarray(gt_omega), axis=-1
        ).astype(np.float32)
        for key, values in (
            ("gt", gt_magnitude),
            ("recon", recon_magnitude),
            ("error", error_magnitude),
        ):
            values[~mask] = 0.0
            surfaces[(case_id, key)] = render.surface(values)
        figure = plt.figure(figsize=(12.0, 4.8), facecolor="white")
        for column, (key, title, color) in enumerate(
            (
                ("gt", "DNS ground truth", render.BLUE),
                ("recon", "Double-zero Laplacian A=1024", render.RED),
                ("error", "Absolute vorticity error", render.PURPLE),
            )
        ):
            axis = figure.add_subplot(1, 3, column + 1, projection="3d")
            mesh_data = surfaces[(case_id, key)]
            render.draw(axis, mesh_data, color)
            axis.set_title(
                f"{title}\n|vorticity| iso {mesh_data[2]:.3g}",
                fontsize=10,
                color=render.DARK,
            )
        figure.suptitle(
            f"{label} - 1024 complete A tensors / 3072 DOFs - full 128^3\n"
            f"velocity rel. L2 {velocity_error:.2%}; vorticity rel. L2 {vorticity_error:.2%}",
            fontsize=12,
            color=render.DARK,
            y=0.99,
        )
        # Reserve a distinct header band for the two-line summary.  Without
        # this gap the center subplot title can overlap the metric line in PDF.
        figure.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.72, wspace=0.005)
        artifacts.extend(
            render.save_figure(figure, case_dir / "vorticity_isosurface_comparison")
        )
        metrics.append(
            {
                "id": case_id,
                "label": label,
                "frame": 0,
                "tensor_A_groups": A_GROUPS,
                "scalar_DOFs": SCALAR_DOFS,
                "velocity_relative_l2": velocity_error,
                "vorticity_relative_l2": vorticity_error,
                "coefficient_l2": float(np.linalg.norm(coefficients[index])),
            }
        )

    figure = plt.figure(figsize=(12.0, 11.2), facecolor="white")
    for row, (case_id, label, _) in enumerate(CASES):
        for column, (key, title, color) in enumerate(
            (
                ("gt", "DNS ground truth", render.BLUE),
                ("recon", "A=1024 / DOF=3072", render.RED),
                ("error", "Absolute error", render.PURPLE),
            )
        ):
            axis = figure.add_subplot(3, 3, row * 3 + column + 1, projection="3d")
            mesh_data = surfaces[(case_id, key)]
            render.draw(axis, mesh_data, color)
            axis.set_title(
                f"{label}\n{title} - iso {mesh_data[2]:.3g}",
                fontsize=9,
                color=render.DARK,
                pad=0,
            )
    figure.suptitle(
        "Three DNS first frames - double-zero Laplacian - 1024 complete A tensors / 3072 DOFs",
        fontsize=13,
        color=render.DARK,
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.005, right=0.995, bottom=0.005, top=0.94, wspace=0.002, hspace=0.08
    )
    artifacts.extend(
        render.save_figure(
            figure,
            output / "three_dns_double_zero_laplacian_A1024_DOF3072_full128",
        )
    )
    return metrics, artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--scalar-cache", type=Path, default=DEFAULT_SCALAR_CACHE)
    parser.add_argument("--scalar-metadata", type=Path, default=DEFAULT_SCALAR_METADATA)
    parser.add_argument("--basis-root", type=Path, default=DEFAULT_BASIS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spatial-blocks", type=int, default=1)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    scalar_cache = args.scalar_cache.resolve()
    scalar_metadata_path = args.scalar_metadata.resolve()
    basis_root = args.basis_root.resolve()
    output = args.output.resolve()
    basis_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    cfg, geometry, mask, targets, gt_vorticity = previous.load_context(workspace)
    _, flat_indices = helper.cell_center_points(mask)
    if len(flat_indices) != 1_599_754:
        raise ValueError(f"unexpected fluid-node count {len(flat_indices)}")
    scalar_metadata = json.loads(scalar_metadata_path.read_text(encoding="utf-8"))
    if scalar_metadata.get("scalar_modes") != A_GROUPS:
        raise ValueError("scalar cache is not the verified set of 1024 distinct shapes")

    envelope, envelope_audit = previous.double_zero_envelope(
        cfg,
        geometry,
        mask,
        flat_indices,
        basis_root / "double_zero_envelope_128.npy",
    )
    atomic_json(basis_root / "envelope_audit.json", envelope_audit)
    log(f"double-zero envelope ready: {json.dumps(envelope_audit)}")

    scalar_mode_major_path = basis_root / "scalar_1024_mode_major_128.npy"
    build_scalar_mode_major(
        scalar_cache,
        scalar_mode_major_path,
        basis_root / "scalar_mode_major_progress.json",
        mask.size,
    )
    gradient_mode_major_path = basis_root / "gradient_A1024_mode_major_128.npy"
    build_gradient_mode_major(
        scalar_mode_major_path,
        envelope,
        mask,
        flat_indices,
        gradient_mode_major_path,
        basis_root / "gradient_mode_major_progress.json",
    )
    gradient_path = basis_root / "gradient_A1024_spatial_major_128.npy"
    transpose_gradient_basis(
        gradient_mode_major_path,
        gradient_path,
        basis_root / "gradient_spatial_major_progress.json",
        len(flat_indices),
    )

    mass, correlations, mass_audit = assemble_mass_and_correlations(
        gradient_path,
        targets,
        basis_root / "mass_and_correlations_DOF3072.npz",
        basis_root / "mass_progress.json",
        args.spatial_blocks,
    )
    coefficient_path = output / "coefficients_A1024_DOF3072.npz"
    _, coefficients_grouped, solve_audit = solve_all_dofs(
        mass, correlations, coefficient_path
    )
    velocity = reconstruct_all_dofs(
        gradient_path,
        coefficients_grouped,
        mask,
        flat_indices,
        output / "velocity_three_cases_A1024_DOF3072.npy",
        output / "velocity_progress.json",
        args.spatial_blocks,
    )
    vorticity = helper.curl_trajectory(
        velocity,
        mask,
        output / "vorticity_three_cases_A1024_DOF3072.npy",
    )
    metrics, artifacts = render_results(
        workspace,
        output,
        mask,
        velocity,
        vorticity,
        coefficients_grouped,
        gt_vorticity,
    )
    artifacts.extend(
        (
            coefficient_path,
            output / "velocity_three_cases_A1024_DOF3072.npy",
            output / "vorticity_three_cases_A1024_DOF3072.npy",
        )
    )
    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    artifacts.append(metrics_path)

    basis_completion = {
        "complete": True,
        "method": "unscreened double-zero tensor-potential Laplacian basis",
        "grid": [RESOLUTION] * 3,
        "fluid_nodes": int(len(flat_indices)),
        "source_scalar_shapes": A_GROUPS,
        "source_scalar_shapes_distinct": True,
        "source_scalar_relative_residual_max": float(
            max(
                batch.get("relative_residual_max", 0.0)
                for batch in scalar_metadata.get("batches", [{}])
            )
        ),
        "tensor_A_groups": A_GROUPS,
        "tensor_components_per_A": TENSOR_COMPONENTS,
        "scalar_DOFs": SCALAR_DOFS,
        "retained_DOFs": SCALAR_DOFS,
        "selection": "none",
        "Rayleigh_Ritz_screening": False,
        "double_zero": True,
        "envelope": envelope_audit,
        "velocity_mapping": {
            "A12": "(dy(phi), -dx(phi), 0)",
            "A13": "(dz(phi), 0, -dx(phi))",
            "A23": "(0, dz(phi), -dy(phi))",
        },
        "potential_derivative_order": 1,
        "jet3": False,
        "unused_derivative_channels": False,
        "operator": "M=<u_i,u_j> only for full 3072-DOF L2 projection",
        "mass_audit": mass_audit,
        "solve_audit": solve_audit,
        "spatial_blocks": args.spatial_blocks,
        "subsampling": False,
        "tf32": False,
        "gradient_basis": str(gradient_path),
    }
    atomic_json(basis_root / "completion.json", basis_completion)

    complete = all(path.is_file() and path.stat().st_size > 0 for path in artifacts)
    completion = {
        **basis_completion,
        "complete": complete,
        "projection": "all 1,599,754 fluid nodes on the 128^3 grid",
        "cases": metrics,
        "active_wall_seconds_this_process": float(time.perf_counter() - started),
        "artifacts": [str(path) for path in artifacts],
    }
    atomic_json(output / "completion.json", completion)
    report_rows = [
        "# Double-zero Laplacian: 1024 complete A tensors / 3072 DOFs",
        "",
        "- The first 1024 distinct scalar Laplacian shapes define 1024 tensor-potential groups A.",
        "- Every A retains independent A12, A13, and A23 coefficients: 1024 x 3 = 3072 scalar DOFs.",
        "- All 3072 DOFs are used directly. There is no Rayleigh-Ritz screening or mode selection.",
        "- The exact squared transfer envelope enforces double-zero potential boundary behavior.",
        "- Only first derivatives needed by velocity are computed; Jet3 is disabled.",
        "- Projection uses every fluid node on the full 128^3 grid; no subsampling and TF32 is disabled.",
        "",
        "| Case | Velocity relative L2 | Vorticity relative L2 |",
        "|---|---:|---:|",
    ]
    report_rows.extend(
        f"| {item['label']} | {item['velocity_relative_l2']:.4%} | "
        f"{item['vorticity_relative_l2']:.4%} |"
        for item in metrics
    )
    (output / "REPORT.md").write_text("\n".join(report_rows) + "\n", encoding="utf-8")
    if not complete:
        raise RuntimeError("A1024/DOF3072 artifact validation failed")
    log(json.dumps(completion, indent=2))
    log("double-zero Laplacian A1024/DOF3072 complete")


if __name__ == "__main__":
    main()
