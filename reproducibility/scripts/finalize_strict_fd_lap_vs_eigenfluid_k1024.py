"""Finalize the strict d^2 FD-Lap versus EigenFluid K=1024 comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GT_ROOT = RESULTS / "boundary_compatible_double_zero_gt_128_v2"
STRICT_ROOT = RESULTS / "fd_lap_strict_matched_e_phiD_A1024_vector_k1024_full128"
EIGEN_ROOT = (
    RESULTS
    / "eigenfluid_fixed_geometry_laplacian_matched_k1024_p9192_e2000"
)
EIGEN_EVAL = (
    EIGEN_ROOT
    / "boundary_gt_full128_reconstruction"
    / "fair_fd_curl_evaluation"
)
HISTORICAL_FD = RESULTS / "boundary_compatible_double_zero_gt_fd_lap_vector_k1024_reused_full128"
INDEPENDENT_FD = RESULTS / "boundary_compatible_double_zero_gt_laplacian_A1024_full128"
INVALID_FD = RESULTS / "boundary_compatible_double_zero_gt_fd_lap_shared_q_A1024_full128"
TIMING_ROOT = RESULTS / "laplacian_128_scalar_k1024"
OUTPUT = RESULTS / "final_strict_fd_lap_vs_eigenfluid_k1024_comparison"
GLOBAL_REPORT = RESULTS / "RESULTS_COMPARISON_20260828.md"
GRID = 128
FLUID_NODES = 1_599_754

BLUE = "#3B6FB6"
LIGHT_BLUE = "#7CAACB"
RED = "#C83E4D"
ORANGE = "#E07B39"
PURPLE = "#7B4FA3"
DARK = "#1E293B"
GRAY = "#64748B"
PALE = "#EEF4F8"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, indent=2) + "\n")


def mask_from_indices() -> np.ndarray:
    path = EIGEN_ROOT / "boundary_gt_prepared" / "fluid_indices.u32"
    indices = np.fromfile(path, dtype=np.uint32)
    if indices.shape != (FLUID_NODES,):
        raise ValueError(f"unexpected fluid indices: {indices.shape}")
    mask = np.zeros(GRID**3, dtype=bool)
    mask[indices] = True
    return mask.reshape((GRID, GRID, GRID))


def magnitude(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.linalg.norm(np.asarray(field), axis=-1).astype(np.float32)
    result[~mask] = 0.0
    return result


def surface(values: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < level < float(values.max()):
        raise ValueError(f"invalid shared level {level} for maximum {values.max()}")
    vertices, faces, _, _ = measure.marching_cubes(
        values, level=level, step_size=2, allow_degenerate=False
    )
    if len(faces) > 150_000:
        vertices, faces, _, _ = measure.marching_cubes(
            values, level=level, step_size=3, allow_degenerate=False
        )
    return vertices[:, [2, 1, 0]] / GRID, faces


def draw_surface(axis, mesh: tuple[np.ndarray, np.ndarray], color: str) -> None:
    vertices, faces = mesh
    polygon = Poly3DCollection(
        vertices[faces], facecolor=color, edgecolor="none", alpha=0.90
    )
    polygon.set_rasterized(True)
    axis.add_collection3d(polygon)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_zlim(0.0, 1.0)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=24.0, azim=-52.0)
    axis.set_axis_off()


def render_common_isosurfaces(
    mask: np.ndarray,
    gt: np.ndarray,
    eigen: np.ndarray,
    strict: np.ndarray,
    level: float,
    eigen_velocity_error: float,
    eigen_vorticity_error: float,
    strict_velocity_error: float,
    strict_vorticity_error: float,
    png_path: Path,
    pdf_path: Path,
) -> None:
    fields = [gt, eigen, strict, eigen - gt, strict - gt]
    meshes = [surface(magnitude(field, mask), level) for field in fields]
    titles = [
        "Matched GT",
        "EigenFluid K=1024",
        "Strict FD-Lap K=1024",
        "EigenFluid error",
        "Strict FD-Lap error",
    ]
    colors = [BLUE, RED, ORANGE, PURPLE, LIGHT_BLUE]
    figure = plt.figure(figsize=(14.2, 7.2), facecolor="white")
    grid = figure.add_gridspec(2, 3, left=0.015, right=0.985, bottom=0.04, top=0.86)
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for mesh, title, color, position in zip(meshes, titles, colors, positions):
        axis = figure.add_subplot(grid[position], projection="3d")
        draw_surface(axis, mesh, color)
        axis.set_title(title, color=DARK, fontsize=10, pad=2)
    note = figure.add_subplot(grid[1, 2])
    note.axis("off")
    note.text(
        0.02,
        0.95,
        "Strictly aligned comparison",
        color=DARK,
        weight="bold",
        fontsize=12,
        va="top",
    )
    note.text(
        0.02,
        0.79,
        "Full 128 x 128 x 128\n1,599,754 fluid nodes\n1024 complete modes / 1024 q DOFs\nNo subsampling, ridge, or TF32",
        color=GRAY,
        fontsize=9.5,
        va="top",
        linespacing=1.45,
    )
    note.text(
        0.02,
        0.43,
        f"EigenFluid:  velocity {eigen_velocity_error:.2%}\n"
        f"                  vorticity {eigen_vorticity_error:.2%}\n\n"
        f"Strict FD-Lap: velocity {strict_velocity_error:.2%}\n"
        f"                  vorticity {strict_vorticity_error:.2%}",
        color=DARK,
        fontsize=10,
        va="top",
        linespacing=1.35,
    )
    figure.suptitle(
        "Common absolute isovorticity comparison - GT q96.5 level "
        f"|omega| = {level:.4f}",
        color=DARK,
        fontsize=14,
        weight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.91,
        "The same 128^3 second-order finite-difference curl and the same absolute isosurface strength are used in every panel.",
        ha="center",
        color=GRAY,
        fontsize=9.5,
    )
    figure.savefig(png_path, dpi=240, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)


def render_comparison_summary(
    rows: list[dict],
    eigenvalues: np.ndarray,
    png_path: Path,
    pdf_path: Path,
    scalar_hours: float,
    strict_seconds: float,
    eigen_seconds: float,
    orth_error: float,
    eig_residual: float,
) -> None:
    figure = plt.figure(figsize=(13.8, 8.8), facecolor="white")
    grid = figure.add_gridspec(
        2, 2, height_ratios=(1.15, 1.0), width_ratios=(1.1, 1.0),
        left=0.055, right=0.975, bottom=0.165, top=0.88, hspace=0.35, wspace=0.24
    )
    table_axis = figure.add_subplot(grid[0, :])
    table_axis.axis("off")
    columns = ["Method", "Boundary order", "DOFs", "Velocity L2", "Vorticity L2", "Energy kept", "Role"]
    cells = [
        [
            row["method"], row["boundary"], str(row["dofs"]),
            f"{row['velocity']:.2%}", f"{row['vorticity']:.2%}",
            f"{row['energy']:.2%}", row["role"],
        ]
        for row in rows
    ]
    table = table_axis.table(
        cellText=cells, colLabels=columns, cellLoc="center", colLoc="center",
        loc="center", colWidths=[0.20, 0.18, 0.08, 0.12, 0.13, 0.11, 0.18]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.65)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row == 0:
            cell.set_facecolor(DARK)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row in (1, 2):
            cell.set_facecolor("#EAF2F8" if row == 1 else "#FBEDE8")
        else:
            cell.set_facecolor("#F8FAFC")
            cell.get_text().set_color(GRAY)
    error_axis = figure.add_subplot(grid[1, 0])
    final_rows = rows[:2]
    x = np.arange(2)
    width = 0.34
    velocity = [row["velocity"] * 100.0 for row in final_rows]
    vorticity = [row["vorticity"] * 100.0 for row in final_rows]
    error_axis.bar(x - width / 2, velocity, width, color=[RED, ORANGE], label="Velocity")
    error_axis.bar(x + width / 2, vorticity, width, color=[PURPLE, BLUE], label="Vorticity")
    error_axis.set_xticks(x, ["EigenFluid", "Strict FD-Lap"])
    error_axis.set_ylabel("Relative L2 error (%)")
    error_axis.set_ylim(0.0, max(vorticity) * 1.20)
    error_axis.grid(axis="y", alpha=0.20)
    error_axis.spines[["top", "right"]].set_visible(False)
    for bars in error_axis.containers:
        error_axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    error_axis.set_title("Strict 1024-DOF reconstruction error", color=DARK, weight="bold")
    spectrum_axis = figure.add_subplot(grid[1, 1])
    spectrum_axis.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, color=BLUE, linewidth=1.8)
    spectrum_axis.fill_between(np.arange(1, len(eigenvalues) + 1), eigenvalues, color=LIGHT_BLUE, alpha=0.25)
    spectrum_axis.set_xlabel("Retained strict FD-Lap mode index")
    spectrum_axis.set_ylabel("Generalized eigenvalue")
    spectrum_axis.set_title("Strict FD-Lap Rayleigh-Ritz spectrum", color=DARK, weight="bold")
    spectrum_axis.grid(alpha=0.20)
    spectrum_axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Final K=1024 comparison: strict d^2 FD-Lap vs EigenFluid",
        fontsize=16,
        color=DARK,
        weight="bold",
        y=0.965,
    )
    figure.text(
        0.5, 0.915,
        "Same GT, geometry token 12000001, 128^3 grid, 1,599,754 fluid nodes, 1024 q coefficients, and common FD-curl operator.",
        ha="center", color=GRAY, fontsize=9.5,
    )
    figure.text(
        0.055, 0.030,
        f"Audit: max |C^T M C - I| = {orth_error:.2e}; generalized residual = {eig_residual:.2e}.\n"
        f"Compute: EigenFluid training {eigen_seconds / 3600:.3f} h; stored scalar-eigenspace reference {scalar_hours:.3f} h; "
        f"strict vector RR + reconstruction {strict_seconds / 60:.2f} min.",
        color=GRAY, fontsize=8.2, linespacing=1.45,
    )
    figure.savefig(png_path, dpi=240, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)


def main() -> None:
    png_dir = OUTPUT / "output" / "png"
    pdf_dir = OUTPUT / "output" / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    strict = load_json(STRICT_ROOT / "completion.json")
    eigen = load_json(EIGEN_EVAL / "completion.json")
    training = load_json(EIGEN_ROOT / "completion.json")
    historical = load_json(HISTORICAL_FD / "completion.json")
    independent = load_json(INDEPENDENT_FD / "completion.json")
    invalid = load_json(INVALID_FD / "completion.json")
    scalar_timing = load_json(TIMING_ROOT / "generation_timing_gt.json")
    jet_timing = load_json(EIGEN_ROOT / "data" / "jet1_vs_jet3_equivalent_timing.json")

    strict_velocity = float(strict["errors"]["velocity_relative_l2"])
    strict_vorticity = float(strict["errors"]["vorticity_relative_l2"])
    eigen_velocity = float(eigen["velocity_relative_l2"])
    eigen_vorticity = float(eigen["vorticity_relative_l2_common_fd_curl"])
    velocity_reduction = 1.0 - strict_velocity / eigen_velocity
    vorticity_reduction = 1.0 - strict_vorticity / eigen_vorticity
    rows = [
        {
            "method": "EigenFluid epoch 2000",
            "boundary": "e^2 A_raw ~ d^2",
            "dofs": 1024,
            "velocity": eigen_velocity,
            "vorticity": eigen_vorticity,
            "energy": 1.0 - eigen_velocity**2,
            "role": "Final fair result",
        },
        {
            "method": "Strict FD-Lap Rayleigh-Ritz",
            "boundary": "e phi_D E_alpha ~ d^2",
            "dofs": 1024,
            "velocity": strict_velocity,
            "vorticity": strict_vorticity,
            "energy": float(strict["errors"]["velocity_energy_retained"]),
            "role": "Final fair result",
        },
        {
            "method": "Historical FD-Lap vector",
            "boundary": "e^2 phi_D E_alpha ~ d^3",
            "dofs": 1024,
            "velocity": float(historical["errors"]["velocity_relative_l2"]),
            "vorticity": float(historical["errors"]["vorticity_relative_l2"]),
            "energy": float(historical["errors"]["velocity_energy_retained"]),
            "role": "Boundary-unmatched history",
        },
        {
            "method": "FD-Lap independent polarizations",
            "boundary": "e phi_D E_alpha ~ d^2",
            "dofs": 3072,
            "velocity": float(independent["errors"]["velocity_relative_l2"]),
            "vorticity": float(independent["errors"]["vorticity_relative_l2"]),
            "energy": float(independent["errors"]["velocity_energy_retained"]),
            "role": "Non-DOF-fair reference",
        },
        {
            "method": "Invalid fixed 1:1:1 polarization",
            "boundary": "e phi_D sum(E_alpha) ~ d^2",
            "dofs": 1024,
            "velocity": float(invalid["errors"]["velocity_relative_l2"]),
            "vorticity": float(invalid["errors"]["vorticity_relative_l2"]),
            "energy": float(invalid["errors"]["velocity_energy_retained"]),
            "role": "Invalid audit only",
        },
    ]

    mask = mask_from_indices()
    gt_vorticity = np.load(GT_ROOT / "gt_vorticity_128.npy", mmap_mode="r")
    eigen_vorticity_field = np.load(
        EIGEN_EVAL / "eigenfluid_vorticity_common_fd_curl_k1024_128.npy", mmap_mode="r"
    )
    strict_vorticity_field = np.load(
        STRICT_ROOT / "vorticity_common_fd_curl_k1024_128.npy", mmap_mode="r"
    )
    gt_magnitude = magnitude(gt_vorticity, mask)
    level = float(np.quantile(gt_magnitude[gt_magnitude > 0.0], 0.965))
    iso_png = png_dir / "common_isovorticity_q965.png"
    iso_pdf = pdf_dir / "common_isovorticity_q965.pdf"
    render_common_isosurfaces(
        mask, gt_vorticity, eigen_vorticity_field, strict_vorticity_field, level,
        eigen_velocity, eigen_vorticity, strict_velocity, strict_vorticity,
        iso_png, iso_pdf,
    )

    with np.load(STRICT_ROOT / "vector_modes_k1024.npz", allow_pickle=False) as cached:
        eigenvalues = np.asarray(cached["eigenvalues"], dtype=np.float64)
    summary_png = png_dir / "final_strict_fd_lap_vs_eigenfluid_k1024_comparison.png"
    summary_pdf = pdf_dir / "final_strict_fd_lap_vs_eigenfluid_k1024_comparison.pdf"
    render_comparison_summary(
        rows, eigenvalues, summary_png, summary_pdf,
        float(scalar_timing["hours"]["deduplicated_equivalent_full_1024_estimate"]),
        float(strict["active_wall_seconds"]),
        float(training["active_training_seconds"]),
        float(strict["eigen_audit"]["mass_orthogonality_max_abs"]),
        float(strict["eigen_audit"]["generalized_eigen_relative_residual"]),
    )

    metrics_path = OUTPUT / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["method", "boundary", "dofs", "velocity_relative_l2", "vorticity_relative_l2", "velocity_energy_retained", "role"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "boundary": row["boundary"],
                    "dofs": row["dofs"],
                    "velocity_relative_l2": f"{row['velocity']:.12g}",
                    "vorticity_relative_l2": f"{row['vorticity']:.12g}",
                    "velocity_energy_retained": f"{row['energy']:.12g}",
                    "role": row["role"],
                }
            )
    metrics = {
        "complete": True,
        "final_fair_rows": [rows[0], rows[1]],
        "strict_fd_error_reduction_vs_eigenfluid": {
            "velocity_fraction": velocity_reduction,
            "vorticity_fraction": vorticity_reduction,
        },
        "shared_configuration": {
            "grid": [GRID, GRID, GRID],
            "fluid_nodes": FLUID_NODES,
            "geometry_token": 12000001,
            "independent_coefficients": 1024,
            "vorticity_operator": "128^3 second-order finite-difference curl",
            "isosurface_level_policy": "one absolute GT q96.5 level shared by all fields and errors",
            "isosurface_common_level": level,
            "subsampling": False,
            "regularization": None,
            "tf32": False,
        },
        "strict_fd_audit": strict["eigen_audit"],
    }
    atomic_json(OUTPUT / "metrics.json", metrics)

    fair_table = [
        "| Method | Potential / boundary order | Trial construction | Independent q DOFs | Velocity rel. L2 | Common-FD-curl vorticity rel. L2 | Velocity energy retained | Status |",
        "|---|---|---|---:|---:|---:|---:|---|",
        f"| EigenFluid epoch 2000 | `e^2 A_raw ~ d^2` | 1024 independently learned complete tensor modes | 1024 | {eigen_velocity:.4%} | {eigen_vorticity:.4%} | {1-eigen_velocity**2:.4%} | Final fair result |",
        f"| Strict FD-Lap Rayleigh-Ritz | `e phi_D E_alpha ~ d^2` | 1024 scalar shapes x 3 polarizations = 3072 trials, then generalized RR -> 1024 complete vector modes | 1024 | {strict_velocity:.4%} | {strict_vorticity:.4%} | {1-strict_velocity**2:.4%} | Final fair result |",
    ]
    historical_table = [
        "| Reference | DOFs | Velocity rel. L2 | Vorticity rel. L2 | Why it is not a final fair row |",
        "|---|---:|---:|---:|---|",
        f"| Historical FD-Lap vector `~d^3` | 1024 | {rows[2]['velocity']:.4%} | {rows[2]['vorticity']:.4%} | Boundary order differs from EigenFluid/GT |",
        f"| Independent FD polarizations | 3072 | {rows[3]['velocity']:.4%} | {rows[3]['vorticity']:.4%} | Three times as many fitted coefficients |",
        f"| Fixed internal 1:1:1 polarization | 1024 | {rows[4]['velocity']:.4%} | {rows[4]['vorticity']:.4%} | Invalid overconstraint; audit only |",
    ]
    report_lines = [
        "# Final strict K=1024 FD-Lap vs EigenFluid comparison",
        "",
        "## Final fair table",
        "",
        *fair_table,
        "",
        "Both rows use the same matched GT, geometry token 12000001, complete 128^3 grid, all 1,599,754 fluid nodes, 1024 external q coefficients, unregularized projection, and the same second-order finite-difference curl.",
        "",
        "## Outcome",
        "",
        f"Strict FD-Lap reduces velocity error by **{velocity_reduction:.2%}** and vorticity error by **{vorticity_reduction:.2%}** relative to EigenFluid.",
        f"Its generalized-eigen residual is **{strict['eigen_audit']['generalized_eigen_relative_residual']:.3e}** and max mass-orthogonality error is **{strict['eigen_audit']['mass_orthogonality_max_abs']:.3e}**.",
        "",
        "## Compute time",
        "",
        f"- EigenFluid active training: **{training['active_training_seconds']:.3f} s = {training['active_training_seconds']/3600:.4f} h**.",
        f"- EigenFluid Jet1-equivalent 2000 epochs: **{jet_timing['jet1']['equivalent_2000_epoch_hours']:.4f} h**.",
        f"- Stored 1024-scalar Laplacian eigenspace timing reference: **{scalar_timing['hours']['deduplicated_equivalent_full_1024_estimate']:.4f} h**; excludes current vector RR/reconstruction.",
        f"- Current strict vector FD-curl, 3072x3072 RR, K=1024 solve, and reconstruction: **{strict['active_wall_seconds']:.3f} s = {strict['active_wall_seconds']/60:.2f} min**; scalar eigenfunctions were reused.",
        "",
        "## Historical and invalid references",
        "",
        *historical_table,
        "",
        "## Artifact audit",
        "",
        f"- Common isovorticity level: **{level:.8g}** (GT nonzero q96.5).",
        "- PNG and PDF figures use the same absolute isovorticity level for GT, both reconstructions, and both error fields.",
        "- Strict FD-Lap internal polarization is fixed by each generalized eigenvector; one external q_k multiplies each final complete mode.",
        "- No scalar eigensolve was rerun for the strict vector comparison; the 3072-candidate vector Rayleigh-Ritz was recomputed.",
    ]
    report_text = "\n".join(report_lines) + "\n"
    atomic_text(OUTPUT / "REPORT.md", report_text)
    atomic_text(GLOBAL_REPORT, report_text)

    strict_report = [
        "# Strict boundary-matched FD-Lap K=1024",
        "",
        "- Potential: `A_(k,alpha)=e phi_D,k E_alpha ~ d^2`.",
        "- 1024 scalar shapes x 3 polarizations -> 3072 vector trials -> 1024 complete generalized eigenmodes.",
        "- Exactly 1024 independent external q coefficients; internal polarization is fixed by each eigenvector.",
        "- Full 128^3 grid and all 1,599,754 fluid nodes; common second-order FD curl; no subsampling, TF32, ridge, or other regularization.",
        f"- Velocity relative L2: **{strict_velocity:.6%}**.",
        f"- Vorticity relative L2: **{strict_vorticity:.6%}**.",
        f"- Velocity energy retained: **{strict['errors']['velocity_energy_retained']:.6%}**.",
        f"- Max mass-orthogonality error: **{strict['eigen_audit']['mass_orthogonality_max_abs']:.6e}**.",
        f"- Generalized eigen relative residual: **{strict['eigen_audit']['generalized_eigen_relative_residual']:.6e}**.",
    ]
    atomic_text(STRICT_ROOT / "REPORT.md", "\n".join(strict_report) + "\n")
    atomic_text(
        STRICT_ROOT / "metrics.csv",
        "metric,value\n"
        f"velocity_relative_l2,{strict_velocity:.12g}\n"
        f"vorticity_relative_l2,{strict_vorticity:.12g}\n"
        f"velocity_energy_retained,{strict['errors']['velocity_energy_retained']:.12g}\n"
        f"mass_orthogonality_max_abs,{strict['eigen_audit']['mass_orthogonality_max_abs']:.12g}\n"
        f"generalized_eigen_relative_residual,{strict['eigen_audit']['generalized_eigen_relative_residual']:.12g}\n",
    )
    strict["status"] = "complete"
    strict["render_complete"] = True
    strict["isosurface_level_policy"] = "one absolute GT q96.5 level shared by GT, reconstructions, and errors"
    strict["isosurface_common_level"] = level
    strict["final_comparison_root"] = str(OUTPUT.resolve())
    strict["artifacts"] = [str(path.resolve()) for path in (iso_png, iso_pdf, summary_png, summary_pdf, OUTPUT / "metrics.csv", OUTPUT / "REPORT.md")]
    atomic_json(STRICT_ROOT / "completion.json", strict)
    protocol = load_json(STRICT_ROOT / "protocol.json")
    protocol["status"] = "complete"
    atomic_json(STRICT_ROOT / "protocol.json", protocol)
    completion = {
        "complete": True,
        "status": "complete",
        "final_fair_comparison": True,
        "metrics": metrics,
        "artifacts": [str(path.resolve()) for path in (iso_png, iso_pdf, summary_png, summary_pdf, OUTPUT / "metrics.csv", OUTPUT / "metrics.json", OUTPUT / "REPORT.md", GLOBAL_REPORT)],
    }
    atomic_json(OUTPUT / "completion.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
