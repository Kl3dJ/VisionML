import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd


@dataclass
class TrajectoryWindow:
    """Один часовий сегмент траєкторії."""

    window_index: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    start_pose_idx: int
    end_pose_idx: int
    gps_distance_m: float = 0.0
    odom_scale: float = 1.0


@dataclass(frozen=True)
class TrajectoryTransform2D:
    scale: float
    rotation_rad: float
    translation_x: float
    translation_z: float

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")

        rotation = np.array(
            [
                [math.cos(self.rotation_rad), -math.sin(self.rotation_rad)],
                [math.sin(self.rotation_rad), math.cos(self.rotation_rad)],
            ],
            dtype=np.float64,
        )
        return self.scale * (points @ rotation.T) + np.array(
            [self.translation_x, self.translation_z], dtype=np.float64
        )


@dataclass(frozen=True)
class SegmentInfo:
    video_path: Path
    start_time_sec: float
    duration_sec: float
    end_time_sec: float
    frame_count: int
    fps: float
    sample_start_idx: int
    sample_end_idx: int


class DroneVideoCSVParser:
    """Парсер відео дрона та CSV-логів DJI, синхронізований за часом."""

    DEFAULT_SEGMENT_STARTS_SEC = (36.5, 184.0, 291.0)

    def __init__(
        self,
        video_path: Union[str, Path, None] = None,
        csv_path: Union[str, Path] = None,
        camera_matrix: np.ndarray = None,
        start_frame: int = 0,
        time_window_sec: float = 5.0,
        video_time_offset_sec: float = 0.0,
        use_gimbal_orientation: bool = True,
        fixed_down_pitch_deg: float = -90.0,
        video_paths: Optional[Sequence[Union[str, Path]]] = None,
        segment_start_times_sec: Optional[Sequence[float]] = None,
        drone_fov_deg: float = 84.0,
    ):
        if csv_path is None:
            raise ValueError("csv_path is required")

        self.csv_path = Path(csv_path)
        self.start_frame = int(start_frame)
        self.time_window_sec = float(time_window_sec)
        self.video_time_offset_sec = float(video_time_offset_sec)
        self.use_gimbal_orientation = bool(use_gimbal_orientation)
        self.fixed_down_pitch_deg = float(fixed_down_pitch_deg)
        self.drone_fov_deg = float(drone_fov_deg)

        if video_paths is not None:
            self.video_paths = [Path(path) for path in video_paths]
        elif video_path is not None:
            self.video_paths = [Path(video_path)]
        else:
            raise ValueError("Either video_path or video_paths must be provided")

        if segment_start_times_sec is None:
            if len(self.video_paths) == 3:
                segment_start_times_sec = self.DEFAULT_SEGMENT_STARTS_SEC
            else:
                segment_start_times_sec = (0.0,) * len(self.video_paths)

        if len(segment_start_times_sec) != len(self.video_paths):
            raise ValueError("segment_start_times_sec must match the number of videos")

        self.segment_start_times_sec = [float(value) for value in segment_start_times_sec]
        self.cap_list: List[cv2.VideoCapture] = []
        self.segment_info: List[SegmentInfo] = []
        self._segment_last_local_idx: List[int] = []
        self._segment_last_frame: List[Optional[np.ndarray]] = []

        for path in self.video_paths:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise FileNotFoundError(f"Не вдалося відкрити відео: {path}")
            self.cap_list.append(cap)

        self.df = self._load_csv()
        self._compute_local_trajectory()
        self._compute_pose_windows()

        if self.start_frame < 0 or self.start_frame >= len(self._sample_segment_ids):
            raise ValueError("start_frame поза межами доступної траєкторії")

        self._frame_global_offset = 0
        self._length = len(self._sample_segment_ids)
        self.K = camera_matrix if camera_matrix is not None else self._default_k(fov_deg=self.drone_fov_deg)

        self._next_frame_idx = 0
        self._current_frame = None

    # ------------------------------------------------------------------ #
    #                         Завантаження CSV                             #
    # ------------------------------------------------------------------ #
    def _load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, low_memory=False)

        required_columns = ["OSD.flyTime [s]", "OSD.latitude", "OSD.longitude"]
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

        df["fly_time_sec"] = pd.to_numeric(df["OSD.flyTime [s]"], errors="coerce")
        df["latitude"] = pd.to_numeric(df["OSD.latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["OSD.longitude"], errors="coerce")
        df = df.dropna(subset=["fly_time_sec", "latitude", "longitude"])
        df = df.sort_values("fly_time_sec").reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("CSV does not contain valid lat/lon samples")

        return df

    # ------------------------------------------------------------------ #
    #                  Локальні координати та куті                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _meters_per_degree_lat(lat_deg: float) -> float:
        lat = math.radians(lat_deg)
        return (
            111132.92
            - 559.82 * math.cos(2.0 * lat)
            + 1.175 * math.cos(4.0 * lat)
            - 0.0023 * math.cos(6.0 * lat)
        )

    @staticmethod
    def _meters_per_degree_lon(lat_deg: float) -> float:
        lat = math.radians(lat_deg)
        return (
            111412.84 * math.cos(lat)
            - 93.5 * math.cos(3.0 * lat)
            + 0.118 * math.cos(5.0 * lat)
        )

    def _first_existing_column(self, candidates: List[str]) -> Optional[str]:
        for column in candidates:
            if column in self.df.columns:
                return column
        return None

    def _numeric_series(self, column_name: Optional[str], default: float = 0.0) -> np.ndarray:
        if column_name is None:
            return np.full(len(self.df), float(default), dtype=np.float64)
        return pd.to_numeric(self.df[column_name], errors="coerce").fillna(default).to_numpy(dtype=np.float64)

    def _compute_local_trajectory(self) -> None:
        self.fly_time_sec = self.df["fly_time_sec"].to_numpy(dtype=np.float64)
        self.latitudes = self.df["latitude"].to_numpy(dtype=np.float64)
        self.longitudes = self.df["longitude"].to_numpy(dtype=np.float64)

        self.origin_lat = float(self.latitudes[0])
        self.origin_lon = float(self.longitudes[0])
        self.lat_scale_m = self._meters_per_degree_lat(self.origin_lat)
        self.lon_scale_m = self._meters_per_degree_lon(self.origin_lat)

        self.gps_lonlat_deg = np.column_stack((self.longitudes, self.latitudes)).astype(np.float64)
        self.gps_local_xy_m = np.column_stack(
            (
                (self.longitudes - self.origin_lon) * self.lon_scale_m,
                (self.latitudes - self.origin_lat) * self.lat_scale_m,
            )
        ).astype(np.float64)

    def _interpolate_world_xy(self, query_times_sec: np.ndarray) -> np.ndarray:
        """
        Інтерпольовує GPS координати в локальну систему (метри від походження).
        
        Returns
        -------
        (N, 2) масив координат у метрах: (x_m, y_m) від походження
        """
        query_times_sec = np.asarray(query_times_sec, dtype=np.float64)
        lon = np.interp(query_times_sec, self.fly_time_sec, self.gps_lonlat_deg[:, 0])
        lat = np.interp(query_times_sec, self.fly_time_sec, self.gps_lonlat_deg[:, 1])
        
        # Конвертуємо у метри від походження
        x_m = (lon - self.origin_lon) * self.lon_scale_m
        y_m = (lat - self.origin_lat) * self.lat_scale_m
        
        return np.column_stack((x_m, y_m)).astype(np.float64)

    @staticmethod
    def _unwrap_yaw(euler_deg: np.ndarray) -> np.ndarray:
        yaw = np.unwrap(np.deg2rad(euler_deg[:, 0]))
        out = euler_deg.copy()
        out[:, 0] = np.rad2deg(yaw)
        return out

    @staticmethod
    def euler_to_rot_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
        yaw_rad = np.pi / 2.0 - np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)
        cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )

    def _interpolate_pose(self, frame_time: float) -> np.ndarray:
        frame_time = float(np.clip(frame_time, self.csv_times[0], self.csv_times[-1]))
        pos = np.array([np.interp(frame_time, self.csv_times, self.pos_csv[:, axis]) for axis in range(3)], dtype=np.float64)
        ang = np.array([np.interp(frame_time, self.csv_times, self.euler_csv[:, axis]) for axis in range(3)], dtype=np.float64)

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self.euler_to_rot_matrix(*ang)
        pose[:3, 3] = pos
        return pose

    def _pose_index_for_time(self, frame_time: float) -> int:
        return int(np.clip(np.searchsorted(self.csv_times, frame_time, side="right") - 1, 0, len(self.csv_times) - 1))

    def _compute_pose_windows(self) -> None:
        self.segment_info = []
        sample_segment_ids: List[int] = []
        sample_video_local_indices: List[int] = []
        sample_times_sec: List[float] = []
        sample_world_xy: List[np.ndarray] = []

        for segment_idx, (video_path, start_time_sec, cap) in enumerate(
            zip(self.video_paths, self.segment_start_times_sec, self.cap_list)
        ):
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                raise ValueError(f"Не вдалося визначити FPS для відео: {video_path}")
            if frame_count <= 0:
                raise ValueError(f"Відео не містить кадрів: {video_path}")

            duration_sec = frame_count / fps
            end_time_sec = start_time_sec + duration_sec
            frame_times_sec = start_time_sec + self.video_time_offset_sec + (np.arange(frame_count, dtype=np.float64) / fps)
            world_xy = self._interpolate_world_xy(frame_times_sec)

            sample_start_idx = len(sample_segment_ids)
            sample_end_idx = sample_start_idx + frame_count - 1

            self.segment_info.append(
                SegmentInfo(
                    video_path=video_path,
                    start_time_sec=start_time_sec,
                    duration_sec=duration_sec,
                    end_time_sec=end_time_sec,
                    frame_count=frame_count,
                    fps=fps,
                    sample_start_idx=sample_start_idx,
                    sample_end_idx=sample_end_idx,
                )
            )

            sample_segment_ids.extend([segment_idx] * frame_count)
            sample_video_local_indices.extend(list(range(frame_count)))
            sample_times_sec.extend(frame_times_sec.tolist())
            sample_world_xy.append(world_xy)

        self._sample_segment_ids = np.asarray(sample_segment_ids, dtype=np.int32)
        self._sample_video_local_indices = np.asarray(sample_video_local_indices, dtype=np.int32)
        self._sample_times_sec = np.asarray(sample_times_sec, dtype=np.float64)
        self._sample_world_xy = np.vstack(sample_world_xy).astype(np.float64)

        if self.start_frame > 0:
            self._sample_segment_ids = self._sample_segment_ids[self.start_frame :]
            self._sample_video_local_indices = self._sample_video_local_indices[self.start_frame :]
            self._sample_times_sec = self._sample_times_sec[self.start_frame :]
            self._sample_world_xy = self._sample_world_xy[self.start_frame :]

        self._segment_ranges = self._compute_segment_ranges()
        self._segment_last_local_idx = [-1 for _ in self.video_paths]
        self._segment_last_frame = [None for _ in self.video_paths]

    def _compute_segment_ranges(self) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = []
        if len(self._sample_segment_ids) == 0:
            return ranges

        start = 0
        current_segment = int(self._sample_segment_ids[0])
        for idx in range(1, len(self._sample_segment_ids)):
            segment_id = int(self._sample_segment_ids[idx])
            if segment_id != current_segment:
                ranges.append((start, idx - 1))
                start = idx
                current_segment = segment_id
        ranges.append((start, len(self._sample_segment_ids) - 1))
        return ranges

    @staticmethod
    def _fit_similarity_transform_2d(source_xy: np.ndarray, target_xy: np.ndarray) -> TrajectoryTransform2D:
        source_xy = np.asarray(source_xy, dtype=np.float64)
        target_xy = np.asarray(target_xy, dtype=np.float64)
        if source_xy.ndim != 2 or source_xy.shape[1] != 2:
            raise ValueError("source_xy must have shape (N, 2)")
        if target_xy.ndim != 2 or target_xy.shape[1] != 2:
            raise ValueError("target_xy must have shape (N, 2)")

        n = min(len(source_xy), len(target_xy))
        if n == 0:
            return TrajectoryTransform2D(scale=1.0, rotation_rad=0.0, translation_x=0.0, translation_z=0.0)

        source_xy = source_xy[:n]
        target_xy = target_xy[:n]

        if n == 1:
            offset = target_xy[0] - source_xy[0]
            return TrajectoryTransform2D(
                scale=1.0,
                rotation_rad=0.0,
                translation_x=float(offset[0]),
                translation_z=float(offset[1]),
            )

        source_mean = source_xy.mean(axis=0)
        target_mean = target_xy.mean(axis=0)
        source_centered = source_xy - source_mean
        target_centered = target_xy - target_mean

        covariance = (source_centered.T @ target_centered) / n
        U, singular_values, Vt = np.linalg.svd(covariance)
        rotation = Vt.T @ U.T
        if np.linalg.det(rotation) < 0.0:
            Vt[-1, :] *= -1.0
            rotation = Vt.T @ U.T

        source_var = np.sum(source_centered ** 2) / n
        if source_var <= 1e-12:
            scale = 1.0
        else:
            scale = float(np.sum(singular_values) / source_var)

        translation = target_mean - scale * (rotation @ source_mean)
        rotation_rad = float(math.atan2(rotation[1, 0], rotation[0, 0]))
        return TrajectoryTransform2D(
            scale=scale,
            rotation_rad=rotation_rad,
            translation_x=float(translation[0]),
            translation_z=float(translation[1]),
        )

    # ------------------------------------------------------------------ #
    #                      Читання кадрів (послідовне)                     #
    # ------------------------------------------------------------------ #
    def _read_frame_at(self, segment_idx: int, local_idx: int) -> np.ndarray:
        cap = self.cap_list[segment_idx]
        last_idx = self._segment_last_local_idx[segment_idx]

        if local_idx == last_idx + 1:
            ret, frame = cap.read()
            if not ret:
                raise IOError(f"Не вдалося прочитати кадр {local_idx} з сегмента {segment_idx}")
            self._segment_last_local_idx[segment_idx] = local_idx
            self._segment_last_frame[segment_idx] = frame
            return frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
        ret, frame = cap.read()
        if not ret:
            raise IOError(f"Не вдалося прочитати кадр {local_idx} з сегмента {segment_idx}")
        self._segment_last_local_idx[segment_idx] = local_idx
        self._segment_last_frame[segment_idx] = frame
        return frame

    # ------------------------------------------------------------------ #
    #                          Інтерфейс                                  #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Індекс {idx} поза межами [0, {self._length})")

        global_idx = idx
        segment_idx = int(self._sample_segment_ids[global_idx])
        local_idx = int(self._sample_video_local_indices[global_idx])
        frame = self._read_frame_at(segment_idx, local_idx)
        world_xy = self._sample_world_xy[global_idx]

        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = float(world_xy[0])
        pose[2, 3] = float(world_xy[1])
        return frame, pose

    def __iter__(self):
        for idx in range(self._length):
            yield self[idx]

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, mat):
        self._K = np.array(mat, dtype=np.float64)

    def _default_k(self, fov_deg: float = 84.0) -> np.ndarray:
        """
        Розраховує матрицю калібрування K для камери дрона на основі:
        - Фактичних розмірів кадру
        - Припущеного поля зору (FOV)
        
        DJI дрони типово мають FOV ~84°.
        
        Формула: fx = (width / 2) / tan(FOV / 2)
        """
        first_cap = self.cap_list[0]
        width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Розраховуємо фокусну відстань на основі FOV
        fov_rad = math.radians(fov_deg)
        fx = (width / 2.0) / math.tan(fov_rad / 2.0)
        fy = fx  # Припускаємо квадратні пікселі
        
        cx, cy = width / 2.0, height / 2.0
        
        K = np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        
        print(f"[DroneVideoParser] Розраховано матрицю K: "
              f"{width}x{height}, FOV={fov_deg}°, fx/fy={fx:.1f}")
        
        return K

    def fit_odometry_to_world(self, odom_traj: np.ndarray) -> Tuple[np.ndarray, List[TrajectoryTransform2D]]:
        """Align odometry trajectory to GPS-derived world coordinates segment by segment."""
        odom_traj = np.asarray(odom_traj, dtype=np.float64)
        if odom_traj.ndim != 2 or odom_traj.shape[1] != 3:
            raise ValueError("odom_traj має бути масиву форми (N, 3)")

        n = min(len(odom_traj), len(self._sample_world_xy))
        if n == 0:
            return odom_traj.copy(), []

        aligned = odom_traj[:n].copy()
        transforms: List[TrajectoryTransform2D] = []
        local_fit_window = 120

        for start, end in self._segment_ranges:
            if start >= n:
                break
            end = min(end, n - 1)
            if end <= start:
                continue

            previous_xy_end: Optional[np.ndarray] = None
            first_transform: Optional[TrajectoryTransform2D] = None

            for chunk_start in range(start, end + 1, local_fit_window):
                chunk_end = min(chunk_start + local_fit_window - 1, end)
                if chunk_end <= chunk_start:
                    continue

                odom_chunk = odom_traj[chunk_start : chunk_end + 1]
                gps_chunk = self._sample_world_xy[chunk_start : chunk_end + 1]
                transform = self._fit_similarity_transform_2d(odom_chunk[:, (0, 2)], gps_chunk)
                transformed_xy = transform.apply(odom_chunk[:, (0, 2)])

                if previous_xy_end is not None:
                    continuity_offset = previous_xy_end - transformed_xy[0]
                    transformed_xy = transformed_xy + continuity_offset

                aligned[chunk_start : chunk_end + 1, 0] = transformed_xy[:, 0]
                aligned[chunk_start : chunk_end + 1, 2] = transformed_xy[:, 1]
                previous_xy_end = transformed_xy[-1]

                if first_transform is None:
                    first_transform = transform

            if first_transform is None:
                first_transform = TrajectoryTransform2D(scale=1.0, rotation_rad=0.0, translation_x=0.0, translation_z=0.0)
            transforms.append(first_transform)

        if n < len(odom_traj):
            aligned = np.vstack([aligned, odom_traj[n:]])

        return aligned, transforms

    def summary(self) -> str:
        lines = [
            "DroneVideoCSVParser",
            f"  CSV rows  : {len(self.df)}",
            f"  Segments  : {len(self.video_paths)}",
            f"  Frames    : {self._length} (start_frame={self.start_frame})",
            f"  Origin    : lat={self.origin_lat:.7f}, lon={self.origin_lon:.7f}",
            f"  Scale     : lat={self.lat_scale_m:.3f} m/deg, lon={self.lon_scale_m:.3f} m/deg",
        ]

        for idx, info in enumerate(self.segment_info, start=1):
            lines.append(
                f"  Segment {idx}: {info.video_path.name} | start={info.start_time_sec:.1f}s | "
                f"duration={info.duration_sec:.2f}s | frames={info.frame_count} | fps={info.fps:.2f}"
            )

        if len(self._sample_world_xy) > 0:
            lines.append(
                f"  Start pos : lon={self._sample_world_xy[0, 0]:.7f}, lat={self._sample_world_xy[0, 1]:.7f}"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    base = Path("dataset/drone_footage")
    video_paths = [
        base / "23-02-01_FR_F01_V01.MP4",
        base / "23-02-01_FR_F01_V02.MP4",
        base / "23-02-01_FR_F01_V03.MP4",
    ]
    csv_path = base / "23-02-01_FR_F01.csv"

    if all(path.exists() for path in video_paths) and csv_path.exists():
        parser = DroneVideoCSVParser(
            video_paths=video_paths,
            csv_path=str(csv_path),
            segment_start_times_sec=DroneVideoCSVParser.DEFAULT_SEGMENT_STARTS_SEC,
        )
        print(parser.summary())
    else:
        print("Set video_paths to your drone video files and instantiate DroneVideoCSVParser(...).")