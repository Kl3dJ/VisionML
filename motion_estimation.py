"""
Етап 3: Оцінка руху камери (Motion Estimation)
Essential Matrix → Pose Recovery (R, t) з масштабом із Ground Truth
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
import os

import cv2
import numpy as np
from kitti_parser import KITTIOdometryParser
from droneVideoParser import DroneVideoCSVParser
from feature_matching import FeatureMatcher, DetectorType
import matplotlib.pyplot as plt
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])

@dataclass
class PoseEstimate:
    """Результат оцінки руху між двома кадрами."""
    R:              np.ndarray
    t:              np.ndarray
    t_scaled:       np.ndarray
    scale:          float
    inliers_mask:   np.ndarray
    n_inliers:      int


@dataclass
class TrajectoryState:
    """Накопичена глобальна позиція камери."""
    R_pos: np.ndarray = None
    t_pos: np.ndarray = None

    def __post_init__(self):
        if self.R_pos is None:
            self.R_pos = np.eye(3, dtype=np.float64)
        if self.t_pos is None:
            self.t_pos = np.zeros((3, 1), dtype=np.float64)


@dataclass
class PipelineMetrics:
    """Покадрові метрики роботи VO-пайплайна."""
    frame_times_sec: np.ndarray
    fps_per_frame: np.ndarray
    pose_success: np.ndarray
    inliers_per_frame: np.ndarray


@dataclass
class ExperimentResult:
    """Підсумок експерименту для конкретного детектора."""
    detector_name: str
    estimated_traj: np.ndarray
    gt_traj: np.ndarray
    metrics: PipelineMetrics
    avg_fps: float
    ate_rmse: float
    pose_success_rate: float
    turn_success_rate: float
    straight_success_rate: float


def compute_ate_rmse(
    estimated_traj: np.ndarray,
    gt_traj: np.ndarray,
    start_idx: int = 1,
    axes: Optional[Tuple[int, ...]] = None,
    axis_scales: Optional[Tuple[float, ...]] = None,
) -> float:
    """
    Absolute Trajectory Error (ATE) у формі RMSE.
    """
    est = estimated_traj[start_idx:]
    gt = gt_traj[start_idx:]

    if axes is not None:
        est = est[:, axes]
        gt = gt[:, axes]

    if axis_scales is not None:
        scales = np.asarray(axis_scales, dtype=np.float64)
        if est.shape[1] != len(scales):
            raise ValueError("axis_scales must match the number of selected axes")
        est = est * scales
        gt = gt * scales

    if est.size == 0 or gt.size == 0:
        return float("nan")

    errors = np.linalg.norm(est - gt, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def detect_turn_frames(
    gt_traj: np.ndarray,
    heading_threshold_deg: float = 1.5,
) -> np.ndarray:
    """
    Позначає кадри, де траєкторія має поворот (за зміною heading у площині XZ).
    """
    n = len(gt_traj)
    turn_mask = np.zeros(n, dtype=bool)
    if n < 3:
        return turn_mask

    dx = np.diff(gt_traj[:, 0])
    dz = np.diff(gt_traj[:, 2])
    heading = np.unwrap(np.arctan2(dz, dx))
    d_heading = np.diff(heading)
    threshold = np.deg2rad(heading_threshold_deg)

    # d_heading[i] відповідає кадру i+2 у вихідній послідовності
    turn_mask[2:] = np.abs(d_heading) > threshold
    return turn_mask


def _safe_rate(mask: np.ndarray, success: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(success[mask]))

def load_camera_matrix(calib_path: Union[str, Path], camera_id: int = 0) -> np.ndarray:
    """
    Зчитує внутрішню матрицю камери K з файлу calib.txt (формат KITTI).

    KITTI calib.txt містить рядки вигляду:
        P0: fx 0 cx 0  0 fy cy 0  0 0 1 0
        P1: ...

    Повертає матрицю K (3×3).
    """
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Файл калібрування не знайдено: {calib_path}")

    key = f"P{camera_id}:"
    with open(calib_path) as f:
        for line in f:
            if line.startswith(key):
                vals = list(map(float, line.strip().split()[1:]))
                P = np.array(vals).reshape(3, 4)
                K = P[:3, :3]
                return K

    raise ValueError(
        f"Рядок '{key}' не знайдено у файлі {calib_path}. "
        f"Доступні ключі: P0 … P3 (camera_id=0..3)"
    )

def compute_scale(pose_prev: np.ndarray, pose_curr: np.ndarray) -> float:
    """
    Евклідова відстань між двома позами Ground Truth.

    Parameters
    ----------
    pose_prev, pose_curr : np.ndarray (4×4) — матриці трансформації з GT

    Returns
    -------
    float — реальна метрова відстань між кадрами
    """
    t_prev = pose_prev[:3, 3]
    t_curr = pose_curr[:3, 3]
    return float(np.linalg.norm(t_curr - t_prev))


def build_parser(args, dataset_path: str = "dataset/"):
    """Створює або KITTI, або drone-парсер залежно від CLI-режиму."""
    if args.dataset_type == "drone":
        if not args.drone_csv_path:
            raise ValueError("Для drone-режиму потрібен --drone_csv_path")

        video_paths = getattr(args, "drone_video_paths", None)
        if video_paths:
            selected_paths = video_paths
        elif args.drone_video_path:
            selected_paths = [args.drone_video_path]
        else:
            selected_paths = [
                str(Path(dataset_path) / "drone_footage" / "23-02-01_FR_F01_V01.MP4"),
                str(Path(dataset_path) / "drone_footage" / "23-02-01_FR_F01_V02.MP4"),
                str(Path(dataset_path) / "drone_footage" / "23-02-01_FR_F01_V03.MP4"),
            ]

        segment_starts = getattr(args, "segment_start_times_sec", None)
        return DroneVideoCSVParser(
            video_paths=selected_paths,
            csv_path=args.drone_csv_path,
            start_frame=args.start_frame,
            time_window_sec=args.time_window_sec,
            video_time_offset_sec=args.video_time_offset_sec,
            use_gimbal_orientation=not args.no_gimbal_orientation,
            fixed_down_pitch_deg=args.fixed_down_pitch_deg,
            segment_start_times_sec=segment_starts,
            drone_fov_deg=args.drone_fov_deg,
        )

    return KITTIOdometryParser(dataset_path, sequence=args.sequence, camera=args.camera)

class MotionEstimator:
    """
    Оцінює рух камери між двома кадрами:
      1. Essential Matrix через RANSAC
      2. Pose Recovery (R, t)
      3. Масштабування t за Ground Truth

    Приклад використання:
        K = load_camera_matrix("calib.txt")
        estimator = MotionEstimator(K)
        state = TrajectoryState()

        for idx in range(1, len(parser)):
            img_prev, pose_prev = parser[idx - 1]
            img_curr, pose_curr = parser[idx]

            match = feature_matcher.match(img_prev, img_curr)
            if match is None:
                continue

            estimate = estimator.estimate(
                match.pts_prev, match.pts_curr,
                pose_prev, pose_curr
            )
            if estimate is None:
                continue

            state = estimator.update_trajectory(state, estimate)
            x, z = state.t_pos[0, 0], state.t_pos[2, 0]
    """

    MIN_SCALE: float = 0.1

    def __init__(
        self,
        K: np.ndarray,
        ransac_prob: float = 0.999,
        ransac_threshold: float = 1.0,
        use_ground_truth_scale: bool = True,
    ):
        """
        Parameters
        ----------
        K                  : внутрішня матриця камери (3×3)
        ransac_prob        : ймовірність успіху RANSAC
        ransac_threshold   : порогова відстань (пікселі) для RANSAC-інлаєрів
        """
        self.K                = K.astype(np.float64)
        self.ransac_prob      = ransac_prob
        self.ransac_threshold = ransac_threshold
        self.use_ground_truth_scale = bool(use_ground_truth_scale)

        self._focal = float(K[0, 0])
        self._pp    = (float(K[0, 2]), float(K[1, 2]))

    def estimate(
        self,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        pose_prev: np.ndarray,
        pose_curr: np.ndarray,
    ) -> Optional[PoseEstimate]:
        """
        Оцінює відносний рух між кадрами.

        Parameters
        ----------
        pts_prev, pts_curr : (N, 2) координати збігів
        pose_prev, pose_curr : (4×4) пози Ground Truth для масштабу

        Returns
        -------
        PoseEstimate або None, якщо недостатньо інлаєрів / занадто малий рух
        """
        scale = compute_scale(pose_prev, pose_curr) if self.use_ground_truth_scale else 1.0

        if scale < self.MIN_SCALE:
            return None

        E, mask = cv2.findEssentialMat(
            pts_curr,
            pts_prev,
            focal     = self._focal,
            pp        = self._pp,
            method    = cv2.RANSAC,
            prob      = self.ransac_prob,
            threshold = self.ransac_threshold,
        )
        if E is None:
            return None

        n_inliers, R, t, mask_pose = cv2.recoverPose(
            E,
            pts_curr,
            pts_prev,
            focal = self._focal,
            pp    = self._pp,
            mask  = mask,
        )

        if n_inliers < 8:
            return None

        t_scaled = t * scale

        return PoseEstimate(
            R            = R,
            t            = t,
            t_scaled     = t_scaled,
            scale        = scale,
            inliers_mask = mask_pose.ravel().astype(bool),
            n_inliers    = n_inliers,
        )

    @staticmethod
    def update_trajectory(
        state: TrajectoryState,
        estimate: PoseEstimate,
    ) -> TrajectoryState:
        """
        Застосовує оцінений рух до накопиченої позиції.

        Формула:
            t_pos = t_pos + R_pos @ t_scaled
            R_pos = R @ R_pos

        Parameters
        ----------
        state    : поточний стан (R_pos, t_pos)
        estimate : результат estimate()

        Returns
        -------
        Оновлений TrajectoryState
        """
        new_t = state.t_pos + state.R_pos @ estimate.t_scaled
        new_R = estimate.R @ state.R_pos

        return TrajectoryState(R_pos=new_R, t_pos=new_t)

def run_odometry_pipeline(
    parser,
    feature_matcher,
    estimator: MotionEstimator,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, PipelineMetrics]:
    """
    Повний прохід по датасету: видобуток ознак → оцінка руху → траєкторія.
    
    Для дрона: накопичуємо тільки переклади (без матриці R) - обчисліть перетворення потім через fit_odometry_to_world
    Для KITTI: нормально накопичуємо рухи з матриці R

    Returns
    -------
    estimated_traj : (N, 3) — обчислена траєкторія (x, y, z) у системі камери
    gt_traj        : (N, 3) — Ground Truth: GPS/odometry у метрах
    """
    n = len(parser) if max_frames is None else min(max_frames, len(parser))

    estimated_traj = np.zeros((n, 3), dtype=np.float64)
    gt_traj        = np.zeros((n, 3), dtype=np.float64)
    frame_times_sec = np.zeros(n, dtype=np.float64)
    fps_per_frame = np.zeros(n, dtype=np.float64)
    pose_success = np.zeros(n, dtype=bool)
    inliers_per_frame = np.zeros(n, dtype=np.int32)

    state = TrajectoryState()
    segment_ids = getattr(parser, "_sample_segment_ids", None)
    is_drone = hasattr(parser, "_sample_world_xy")
    drone_dir_xz = np.array([0.0, 1.0], dtype=np.float64)

    img0, pose0 = parser[0]
    _ = img0
    gt_traj[0] = pose0[:3, 3]
    estimated_traj[0] = 0.0  # VO траєкторія починається з нуля

    for idx in range(1, n):
        t_frame_start = time.perf_counter()

        if segment_ids is not None and segment_ids[idx] != segment_ids[idx - 1]:
            # Нова камера/сегмент - скинемо стан VO
            state = TrajectoryState()
            drone_dir_xz = np.array([0.0, 1.0], dtype=np.float64)
            img_curr, pose_curr = parser[idx]
            gt_traj[idx] = pose_curr[:3, 3]
            estimated_traj[idx] = 0.0
            frame_times_sec[idx] = time.perf_counter() - t_frame_start
            fps_per_frame[idx] = 1.0 / max(frame_times_sec[idx], 1e-9)
            continue

        img_prev, pose_prev = parser[idx - 1]
        img_curr, pose_curr = parser[idx]

        match_result = feature_matcher.match(img_prev, img_curr)
        if match_result is None:
            estimated_traj[idx] = estimated_traj[idx - 1]
            gt_traj[idx]        = pose_curr[:3, 3]
            frame_times_sec[idx] = time.perf_counter() - t_frame_start
            fps_per_frame[idx] = 1.0 / max(frame_times_sec[idx], 1e-9)
            continue

        estimate = estimator.estimate(
            match_result.pts_prev,
            match_result.pts_curr,
            pose_prev,
            pose_curr,
        )

        if estimate is not None:
            if is_drone:
                # ДЛЯ ДРОНА: крок беремо за нормою, а напрям - зі згладженого t(x,z).
                # Так зберігаємо повороти без накопичення глобальної R-матриці.
                t = estimate.t.ravel()
                step_norm = float(np.linalg.norm(t))

                raw_dir_xz = np.array([float(t[0]), float(t[2])], dtype=np.float64)
                raw_norm = float(np.linalg.norm(raw_dir_xz))
                if raw_norm > 1e-9:
                    raw_unit = raw_dir_xz / raw_norm

                    # Прибираємо випадкові фліпи напряму (двозначність Essential Matrix).
                    if float(np.dot(raw_unit, drone_dir_xz)) < -0.2:
                        raw_unit = -raw_unit

                    alpha_dir = 0.08
                    drone_dir_xz = (1.0 - alpha_dir) * drone_dir_xz + alpha_dir * raw_unit
                    dir_norm = float(np.linalg.norm(drone_dir_xz))
                    if dir_norm > 1e-9:
                        drone_dir_xz = drone_dir_xz / dir_norm

                step_xz = step_norm * drone_dir_xz
                estimated_traj[idx] = estimated_traj[idx - 1] + np.array([step_xz[0], 0.0, step_xz[1]])
            else:
                # Для KITTI: нормально накопичуємо рухи з матриці R
                state = MotionEstimator.update_trajectory(state, estimate)
                estimated_traj[idx] = state.t_pos.ravel()
            
            pose_success[idx] = True
            inliers_per_frame[idx] = int(estimate.n_inliers)
        else:
            if is_drone:
                estimated_traj[idx] = estimated_traj[idx - 1]
            else:
                estimated_traj[idx] = state.t_pos.ravel()

        gt_traj[idx]        = pose_curr[:3, 3]
        frame_times_sec[idx] = time.perf_counter() - t_frame_start
        fps_per_frame[idx] = 1.0 / max(frame_times_sec[idx], 1e-9)

        if idx % 100 == 0:
            err = np.linalg.norm(estimated_traj[idx] - gt_traj[idx])
            print(
                f"[{idx:04d}/{n}]  "
                f"vo_pos=({estimated_traj[idx,0]:8.2f}, {estimated_traj[idx,2]:8.2f})  "
                f"gps_pos=({gt_traj[idx,0]:8.2f}, {gt_traj[idx,2]:8.2f})  "
                f"err={err:.3f} m"
            )

    metrics = PipelineMetrics(
        frame_times_sec=frame_times_sec,
        fps_per_frame=fps_per_frame,
        pose_success=pose_success,
        inliers_per_frame=inliers_per_frame,
    )
    return estimated_traj, gt_traj, metrics


def run_detector_experiment(
    parser,
    estimator: MotionEstimator,
    detector_type: DetectorType,
    max_frames: Optional[int] = None,
    turn_heading_threshold_deg: float = 1.5,
) -> Optional[ExperimentResult]:
    """Запускає повний експеримент для одного детектора."""
    try:
        matcher = FeatureMatcher(detector_type)
    except cv2.error as e:
        logging.warning(f"Детектор {detector_type.name} недоступний: {e}")
        return None

    logging.info(f"\n=== Експеримент: {detector_type.name} ===")
    estimated_traj, gt_traj, metrics = run_odometry_pipeline(
        parser, matcher, estimator, max_frames=max_frames
    )

    if hasattr(parser, "fit_odometry_to_world"):
        estimated_traj, transforms = parser.fit_odometry_to_world(estimated_traj)
        for idx, transform in enumerate(transforms, start=1):
            logging.info(
                f"  segment {idx}: scale={transform.scale:.6f}, rotation={np.rad2deg(transform.rotation_rad):.3f} deg, "
                f"shift=({transform.translation_x:.3f}, {transform.translation_z:.3f})"
            )

    valid = np.arange(len(metrics.fps_per_frame)) > 0
    avg_fps = float(np.mean(metrics.fps_per_frame[valid]))
    ate_axes = (0, 2) if hasattr(parser, "fit_odometry_to_world") else None
    # Для дрона: GPS вже в метрах, тому не множимо на масштаби
    ate_axis_scales = None
    ate_rmse = compute_ate_rmse(
        estimated_traj,
        gt_traj,
        start_idx=1,
        axes=ate_axes,
        axis_scales=ate_axis_scales,
    )
    pose_success_rate = float(np.mean(metrics.pose_success[valid]))

    turn_mask = detect_turn_frames(gt_traj, heading_threshold_deg=turn_heading_threshold_deg)
    turn_success_rate = _safe_rate(turn_mask & valid, metrics.pose_success)
    straight_success_rate = _safe_rate((~turn_mask) & valid, metrics.pose_success)

    return ExperimentResult(
        detector_name=detector_type.name,
        estimated_traj=estimated_traj,
        gt_traj=gt_traj,
        metrics=metrics,
        avg_fps=avg_fps,
        ate_rmse=ate_rmse,
        pose_success_rate=pose_success_rate,
        turn_success_rate=turn_success_rate,
        straight_success_rate=straight_success_rate,
    )


def plot_fps_histogram(results: List[ExperimentResult]):
    """Гістограма середнього FPS для детекторів."""
    names = [r.detector_name for r in results]
    fps_values = [r.avg_fps for r in results]

    fig = plt.figure(figsize=(8, 5))
    color_map = {
        "SIFT": "#2a9d8f",
        "SURF": "#e9c46a",
        "ORB": "#e76f51",
        "OPTICAL_FLOW": "#264653",
    }
    colors = [color_map.get(name, "#6c757d") for name in names]
    bars = plt.bar(names, fps_values, color=colors)
    for bar, fps in zip(bars, fps_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{fps:.1f}",
            ha="center",
            va="bottom",
        )
    plt.title("Average FPS by Detector")
    plt.ylabel("FPS")
    plt.grid(axis="y", alpha=0.3)
    return fig


def plot_fps_timeseries(results: List[ExperimentResult]):
    """Показує FPS по кадрах, щоб бачити провали продуктивності."""
    fig = plt.figure(figsize=(10, 5))
    colors = {
        "SIFT": "#1d3557",
        "SURF": "#457b9d",
        "ORB": "#e63946",
        "OPTICAL_FLOW": "#2a9d8f",
    }

    for res in results:
        fps = res.metrics.fps_per_frame
        valid = np.isfinite(fps) & (fps > 0)
        if not np.any(valid):
            continue
        x = np.arange(len(fps))[valid]
        plt.plot(
            x,
            fps[valid],
            label=res.detector_name,
            color=colors.get(res.detector_name, None),
            linewidth=1.4,
            alpha=0.9,
        )

    plt.title("FPS by Frame")
    plt.xlabel("Frame")
    plt.ylabel("FPS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return fig


def plot_trajectories(results: List[ExperimentResult], x_label: str = "X Position", y_label: str = "Z Position"):
    """Порівняння траєкторій GT та оцінених траєкторій для всіх детекторів."""
    fig = plt.figure(figsize=(10, 7))
    gt = results[0].gt_traj
    plt.plot(gt[:, 0], gt[:, 2], label="Ground Truth", color="black", linewidth=2.0)

    colors = {
        "SIFT": "#1d3557",
        "SURF": "#457b9d",
        "ORB": "#e63946",
        "OPTICAL_FLOW": "#2a9d8f",
    }
    for res in results:
        plt.plot(
            res.estimated_traj[:, 0],
            res.estimated_traj[:, 2],
            label=f"{res.detector_name} (ATE={res.ate_rmse:.2f}m)",
            linestyle="--",
            color=colors.get(res.detector_name, None),
        )

    plt.title("Ground Truth vs Estimated Trajectories")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    return fig

if __name__ == "__main__":
    parser_args = argparse.ArgumentParser(description="Motion Estimation with KITTI або drone dataset")
    parser_args.add_argument(
        "--dataset_type",
        type=str,
        default="kitti",
        choices=["kitti", "drone"],
        help="Тип джерела даних"
    )
    parser_args.add_argument(
        "--drone_video_path",
        type=str,
        default=None,
        help="Шлях до відео дрона"
    )
    parser_args.add_argument(
        "--drone_video_paths",
        nargs="+",
        default=None,
        help="Список шляхів до відео дрона для окремих сегментів"
    )
    parser_args.add_argument(
        "--drone_csv_path",
        type=str,
        default=None,
        help="Шлях до CSV телеметрії дрона"
    )
    parser_args.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Початковий кадр для drone-парсера"
    )
    parser_args.add_argument(
        "--time_window_sec",
        type=float,
        default=5.0,
        help="Розмір часовго вікна для drone-парсера"
    )
    parser_args.add_argument(
        "--video_time_offset_sec",
        type=float,
        default=0.0,
        help="Зсув часу між відео та CSV"
    )
    parser_args.add_argument(
        "--segment_start_times_sec",
        nargs="+",
        type=float,
        default=None,
        help="Початок кожного відео в секундах CSV-логу"
    )
    parser_args.add_argument(
        "--no_gimbal_orientation",
        action="store_true",
        help="Ігнорувати gimbal-орієнтацію для drone-парсера"
    )
    parser_args.add_argument(
        "--fixed_down_pitch_deg",
        type=float,
        default=-90.0,
        help="Фіксований pitch для downward-looking камери"
    )
    parser_args.add_argument(
        "--sequence",
        type=str,
        default="00",
        help="KITTI sequence"
    )
    parser_args.add_argument(
        "--camera",
        type=str,
        default="image_0",
        help="KITTI camera folder"
    )
    parser_args.add_argument(
        "--max_frames", type=int, default=None,
        help="Максимальна кількість кадрів для обробки (None для всіх кадрів)"
    )
    parser_args.add_argument(
        "--detectors", nargs="+", default=["SIFT", "SURF", "ORB", "OPTICAL_FLOW"],
        help="Список детекторів для тесту: SIFT SURF ORB OPTICAL_FLOW"
    )
    parser_args.add_argument(
        "--turn_heading_threshold_deg", type=float, default=1.5,
        help="Поріг зміни heading (градуси) для визначення поворотів"
    )
    parser_args.add_argument(
        "--no_plot", action="store_true",
        help="Не показувати графіки"
    )
    parser_args.add_argument(
        "--save_plots_dir", type=str, default=None,
        help="Папка для збереження графіків (PNG)"
    )
    parser_args.add_argument(
        "--drone_fov_deg", type=float, default=84.0,
        help="Припущений FOV камери дрона для розрахунку фокусної відстані"
    )
    args = parser_args.parse_args()

    if args.no_plot:
        plt.switch_backend("Agg")

    DATASET_PATH = "dataset/"
    parser = build_parser(args, dataset_path=DATASET_PATH)
    logging.info(parser.summary())

    if args.dataset_type == "drone":
        K = parser.K
    else:
        calib_path = DATASET_PATH + f"sequences/{args.sequence}/calib.txt"
        K = load_camera_matrix(calib_path, camera_id=0)
    logging.info("Матриця камери K:")
    logging.info(K)

    estimator = MotionEstimator(K, use_ground_truth_scale=(args.dataset_type != "drone"))
    max_frames = args.max_frames if args.max_frames is not None else len(parser)

    selected_detectors: List[DetectorType] = []
    for name in args.detectors:
        key = name.upper()
        if key not in DetectorType.__members__:
            logging.warning(f"Невідомий детектор '{name}', пропускаю")
            continue
        selected_detectors.append(DetectorType[key])

    if not selected_detectors:
        raise ValueError("Не обрано жодного валідного детектора")

    results: List[ExperimentResult] = []
    for detector in selected_detectors:
        result = run_detector_experiment(
            parser=parser,
            estimator=estimator,
            detector_type=detector,
            max_frames=max_frames,
            turn_heading_threshold_deg=args.turn_heading_threshold_deg,
        )
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError("Не вдалося виконати жоден експеримент")

    logging.info("\n===== ПІДСУМКОВІ МЕТРИКИ =====")
    for r in results:
        logging.info(
            f"{r.detector_name:>4s} | FPS(avg)={r.avg_fps:8.2f} | "
            f"ATE(RMSE)={r.ate_rmse:8.3f} m | "
            f"success={100.0*r.pose_success_rate:6.2f}% | "
            f"turn_success={100.0*r.turn_success_rate if not np.isnan(r.turn_success_rate) else float('nan'):6.2f}% | "
            f"straight_success={100.0*r.straight_success_rate if not np.isnan(r.straight_success_rate) else float('nan'):6.2f}%"
        )

    fps_fig = plot_fps_histogram(results)
    fps_ts_fig = plot_fps_timeseries(results)
    if args.dataset_type == "drone":
        traj_fig = plot_trajectories(results, x_label="X Position (m)", y_label="Y Position (m)")
    else:
        traj_fig = plot_trajectories(results)

    if args.save_plots_dir:
        os.makedirs(args.save_plots_dir, exist_ok=True)
        fps_path = os.path.join(args.save_plots_dir, "fps_histogram.png")
        fps_ts_path = os.path.join(args.save_plots_dir, "fps_timeseries.png")
        traj_path = os.path.join(args.save_plots_dir, "trajectories_comparison.png")
        fps_fig.savefig(fps_path, dpi=150, bbox_inches="tight")
        fps_ts_fig.savefig(fps_ts_path, dpi=150, bbox_inches="tight")
        traj_fig.savefig(traj_path, dpi=150, bbox_inches="tight")
        logging.info(f"Збережено графік FPS: {fps_path}")
        logging.info(f"Збережено графік FPS по кадрах: {fps_ts_path}")
        logging.info(f"Збережено графік траєкторій: {traj_path}")

    if not args.no_plot:
        plt.show()
