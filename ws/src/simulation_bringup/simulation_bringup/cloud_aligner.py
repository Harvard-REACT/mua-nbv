from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from mua_nbv_py_utils.pcd_io import write_pcd_xyz_ascii


@dataclass
class CloudAlignerConfig:
    mode: str = "overlap"
    debug_save: bool = False
    submap_voxel_m: float = 0.02
    submap_max_points: int = 30000
    overlap_max_dist_m: float = 0.08
    target_crop_dist_m: float = 0.10
    min_overlap_points: int = 150
    icp_voxel_m: float = 0.02
    normal_radius_m: float = 0.06
    normal_max_nn: int = 30
    icp_max_corr_dist_m: float = 0.08
    icp_max_iter: int = 40
    icp_rel_fitness: float = 1e-6
    icp_rel_rmse: float = 1e-6
    accept_min_fitness: float = 0.30
    accept_max_rmse: float = 0.06
    accept_max_translation_m: float = 0.12
    accept_max_rotation_deg: float = 12.0


class CloudAligner:
    def __init__(self, cfg: CloudAlignerConfig, logger):
        self.cfg = cfg
        self._logger = logger
        self._submap_points = np.empty((0, 3), dtype=np.float32)
        self._rng = np.random.default_rng(seed=42)
        try:
            import open3d as o3d  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "registration_mode requires Open3D but it is not available"
            ) from exc
        self._o3d = o3d

    def align(
        self,
        *,
        provisional_points: np.ndarray,
        out_base: str | None = None,
        stamp_label: str | None = None,
    ) -> dict[str, Any]:
        pts = np.asarray(provisional_points, dtype=np.float32).reshape(-1, 3)
        metrics = self._make_base_metrics(pts.shape[0])
        debug_clouds: dict[str, np.ndarray] = {"prior": pts}

        if pts.shape[0] == 0:
            metrics["message"] = "empty provisional cloud"
            return {"points": pts, "metrics": metrics}

        if self._submap_points.shape[0] == 0:
            self._update_submap(pts)
            metrics["message"] = "initialized registration target"
            debug_clouds["target"] = self._submap_points.copy()
            self._save_debug(
                out_base=out_base,
                stamp_label=stamp_label,
                metrics=metrics,
                debug_clouds=debug_clouds,
            )
            return {"points": pts, "metrics": metrics}

        metrics["refinement_available"] = True
        metrics["icp_attempted"] = True
        debug_clouds["target"] = self._submap_points.copy()
        metrics["prealign_nn_rmse"] = self._nn_rmse(pts, self._submap_points)

        source_work = pts
        target_work = self._submap_points

        if self.cfg.mode == "overlap":
            source_work = self._filter_source_overlap(pts, self._submap_points)
            target_work = self._crop_target_near_source(self._submap_points, pts)
            debug_clouds["source_overlap"] = source_work
            debug_clouds["target_crop"] = target_work
            metrics["source_points_overlap_filtered"] = int(source_work.shape[0])
            metrics["target_points_cropped"] = int(target_work.shape[0])
        else:
            metrics["source_points_overlap_filtered"] = int(source_work.shape[0])
            metrics["target_points_cropped"] = int(target_work.shape[0])

        if source_work.shape[0] < self.cfg.min_overlap_points:
            metrics["icp_attempted"] = False
            metrics["message"] = (
                "insufficient overlap support; using prior transform"
            )
            self._update_submap(pts)
            debug_clouds["target"] = self._submap_points.copy()
            self._save_debug(
                out_base=out_base,
                stamp_label=stamp_label,
                metrics=metrics,
                debug_clouds=debug_clouds,
            )
            return {"points": pts, "metrics": metrics}

        src_ds = self._downsample(source_work, self.cfg.icp_voxel_m)
        tgt_ds = self._downsample(target_work, self.cfg.icp_voxel_m)
        metrics["source_points_downsampled"] = int(src_ds.shape[0])
        metrics["target_points_downsampled"] = int(tgt_ds.shape[0])

        if min(src_ds.shape[0], tgt_ds.shape[0]) < 20:
            metrics["icp_attempted"] = False
            metrics["message"] = (
                "downsampled clouds too small; using prior transform"
            )
            self._update_submap(pts)
            debug_clouds["target"] = self._submap_points.copy()
            self._save_debug(
                out_base=out_base,
                stamp_label=stamp_label,
                metrics=metrics,
                debug_clouds=debug_clouds,
            )
            return {"points": pts, "metrics": metrics}

        src_pcd = self._to_pcd(src_ds)
        tgt_pcd = self._to_pcd(tgt_ds)
        self._estimate_normals(src_pcd)
        self._estimate_normals(tgt_pcd)

        reg = self._o3d.pipelines.registration.registration_icp(
            src_pcd,
            tgt_pcd,
            self.cfg.icp_max_corr_dist_m,
            np.eye(4, dtype=np.float64),
            self._o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            self._o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=float(self.cfg.icp_rel_fitness),
                relative_rmse=float(self.cfg.icp_rel_rmse),
                max_iteration=int(self.cfg.icp_max_iter),
            ),
        )

        correction = np.asarray(reg.transformation, dtype=np.float64)
        metrics["fitness"] = float(reg.fitness)
        metrics["inlier_rmse"] = float(reg.inlier_rmse)
        metrics["correction_translation_norm_m"] = float(
            np.linalg.norm(correction[:3, 3])
        )
        metrics["correction_rotation_deg"] = self._rotation_angle_deg(correction)

        refined_full = self._apply_transform(pts, correction)
        metrics["postalignment_nn_rmse"] = self._nn_rmse(
            refined_full, self._submap_points
        )
        debug_clouds["refined"] = refined_full

        accepted, reasons = self._accept_icp(metrics)
        metrics["icp_accepted"] = bool(accepted)
        metrics["message"] = (
            "accepted"
            if accepted
            else "rejected: " + ", ".join(reasons)
        )
        output_points = refined_full if accepted else pts

        self._update_submap(output_points)
        debug_clouds["target"] = self._submap_points.copy()
        self._save_debug(
            out_base=out_base,
            stamp_label=stamp_label,
            metrics=metrics,
            debug_clouds=debug_clouds,
        )
        return {"points": output_points, "metrics": metrics}

    def _make_base_metrics(self, raw_count: int) -> dict[str, Any]:
        return {
            "mode": str(self.cfg.mode),
            "refinement_available": False,
            "icp_attempted": False,
            "icp_accepted": False,
            "fitness": None,
            "inlier_rmse": None,
            "correction_translation_norm_m": 0.0,
            "correction_rotation_deg": 0.0,
            "source_points_raw": int(raw_count),
            "source_points_overlap_filtered": 0,
            "target_points_cropped": 0,
            "source_points_downsampled": 0,
            "target_points_downsampled": 0,
            "prealign_nn_rmse": None,
            "postalignment_nn_rmse": None,
            "message": "not attempted",
        }

    def _to_pcd(self, points: np.ndarray):
        pcd = self._o3d.geometry.PointCloud()
        pcd.points = self._o3d.utility.Vector3dVector(
            np.asarray(points, dtype=np.float64)
        )
        return pcd

    def _downsample(self, points: np.ndarray, voxel_m: float) -> np.ndarray:
        if points.shape[0] == 0:
            return points
        if voxel_m <= 0.0:
            return points.astype(np.float32, copy=False)
        pcd = self._to_pcd(points)
        ds = pcd.voxel_down_sample(float(voxel_m))
        arr = np.asarray(ds.points, dtype=np.float32)
        if arr.size == 0:
            return points.astype(np.float32, copy=False)
        return arr.reshape(-1, 3)

    def _estimate_normals(self, pcd) -> None:
        if len(pcd.points) == 0:
            return
        pcd.estimate_normals(
            self._o3d.geometry.KDTreeSearchParamHybrid(
                radius=float(self.cfg.normal_radius_m),
                max_nn=int(self.cfg.normal_max_nn),
            )
        )

    def _filter_source_overlap(
        self, source_points: np.ndarray, target_points: np.ndarray
    ) -> np.ndarray:
        if source_points.shape[0] == 0 or target_points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)
        src = self._to_pcd(source_points)
        tgt = self._to_pcd(target_points)
        dists = np.asarray(
            src.compute_point_cloud_distance(tgt), dtype=np.float64
        )
        keep = dists <= float(self.cfg.overlap_max_dist_m)
        if not np.any(keep):
            return np.empty((0, 3), dtype=np.float32)
        return source_points[keep]

    def _crop_target_near_source(
        self, target_points: np.ndarray, source_points: np.ndarray
    ) -> np.ndarray:
        if source_points.shape[0] == 0 or target_points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)
        tgt = self._to_pcd(target_points)
        src = self._to_pcd(source_points)
        dists = np.asarray(
            tgt.compute_point_cloud_distance(src), dtype=np.float64
        )
        keep = dists <= float(self.cfg.target_crop_dist_m)
        if not np.any(keep):
            return target_points
        return target_points[keep]

    def _nn_rmse(self, source_points: np.ndarray, target_points: np.ndarray) -> float:
        if source_points.shape[0] == 0 or target_points.shape[0] == 0:
            return float("nan")
        src = self._to_pcd(source_points)
        tgt = self._to_pcd(target_points)
        dists = np.asarray(
            src.compute_point_cloud_distance(tgt), dtype=np.float64
        )
        if dists.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(dists))))

    @staticmethod
    def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return points.astype(np.float32, copy=False)
        rot = np.asarray(transform[:3, :3], dtype=np.float64)
        trans = np.asarray(transform[:3, 3], dtype=np.float64)
        out = (points.astype(np.float64, copy=False) @ rot.T) + trans
        return out.astype(np.float32, copy=False)

    @staticmethod
    def _rotation_angle_deg(transform: np.ndarray) -> float:
        rot = np.asarray(transform[:3, :3], dtype=np.float64)
        trace = float(np.trace(rot))
        c = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
        return float(math.degrees(math.acos(c)))

    def _accept_icp(self, metrics: dict[str, Any]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        fitness = float(metrics["fitness"])
        rmse = float(metrics["inlier_rmse"])
        trans = float(metrics["correction_translation_norm_m"])
        rot = float(metrics["correction_rotation_deg"])
        if fitness < float(self.cfg.accept_min_fitness):
            reasons.append(
                f"fitness<{self.cfg.accept_min_fitness:.3f}"
            )
        if rmse > float(self.cfg.accept_max_rmse):
            reasons.append(f"rmse>{self.cfg.accept_max_rmse:.3f}")
        if trans > float(self.cfg.accept_max_translation_m):
            reasons.append(
                f"translation>{self.cfg.accept_max_translation_m:.3f}"
            )
        if rot > float(self.cfg.accept_max_rotation_deg):
            reasons.append(f"rotation>{self.cfg.accept_max_rotation_deg:.1f}")
        return (len(reasons) == 0), reasons

    def _update_submap(self, new_points: np.ndarray) -> None:
        if new_points.shape[0] == 0:
            return
        merged = (
            new_points
            if self._submap_points.shape[0] == 0
            else np.concatenate([self._submap_points, new_points], axis=0)
        )
        merged = self._downsample(merged, self.cfg.submap_voxel_m)
        max_pts = int(self.cfg.submap_max_points)
        if max_pts > 0 and merged.shape[0] > max_pts:
            idx = self._rng.choice(merged.shape[0], size=max_pts, replace=False)
            merged = merged[np.sort(idx)]
        self._submap_points = merged.astype(np.float32, copy=False)

    def _save_debug(
        self,
        *,
        out_base: str | None,
        stamp_label: str | None,
        metrics: dict[str, Any],
        debug_clouds: dict[str, np.ndarray],
    ) -> None:
        if not self.cfg.debug_save or not out_base or not stamp_label:
            return
        os.makedirs(out_base, exist_ok=True)
        for name, pts in debug_clouds.items():
            if pts is None:
                continue
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
            write_pcd_xyz_ascii(
                os.path.join(out_base, f"align_{name}_{stamp_label}.pcd"),
                arr,
            )
        with open(
            os.path.join(out_base, f"alignmeta_{stamp_label}.json"),
            "w",
        ) as f:
            json.dump(metrics, f, indent=2)

