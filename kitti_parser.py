import os
import numpy as np
import cv2
from pathlib import Path
from typing import Iterator, Tuple, Optional, List


class KITTIOdometryParser:
    """Parser for KITTI Odometry Dataset."""

    def __init__(
        self,
        dataset_path: str,
        sequence: str = "00",
        camera: str = "image_0",
    ):
        """Init parser. Args: dataset_path, sequence, camera folder."""
        self.root = Path(dataset_path)
        self.sequence = sequence
        self.camera = camera

        self.images_dir = self.root / "sequences" / sequence / camera
        self.poses_file = self.root / "poses" / f"{sequence}.txt"

        self._validate_paths()

        self.image_paths: List[Path] = sorted(self.images_dir.glob("*.png"))
        self.poses: np.ndarray = self._load_poses()

        if len(self.image_paths) != len(self.poses):
            print(
                f"[WARNING] Image count ({len(self.image_paths)}) "
                f"≠ pose count ({len(self.poses)}). Using minimum."
            )
        self._length = min(len(self.image_paths), len(self.poses))

    def _validate_paths(self) -> None:
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_dir}")
        if not self.poses_file.exists():
            raise FileNotFoundError(f"Poses file not found: {self.poses_file}")

    def _load_poses(self) -> np.ndarray:
        """Load poses from file. Return (N, 4, 4) transformation matrices."""
        raw = np.loadtxt(self.poses_file)
        poses_4x4 = np.zeros((len(raw), 4, 4), dtype=np.float64)
        poses_4x4[:, :3, :] = raw.reshape(-1, 3, 4)
        poses_4x4[:, 3, 3] = 1.0
        return poses_4x4

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (image, pose) for frame idx."""
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length})")

        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise IOError(f"Failed to load image: {self.image_paths[idx]}")

        return image, self.poses[idx]

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Дозволяє: for image, pose in parser: ..."""
        for i in range(self._length):
            yield self[i]

    @property
    def translations(self) -> np.ndarray:
        """Масив (N, 3) — вектори трансляції з кожної пози."""
        return self.poses[:self._length, :3, 3]

    @property
    def rotations(self) -> np.ndarray:
        """Масив (N, 3, 3) — матриці обертання з кожної пози."""
        return self.poses[:self._length, :3, :3]

    def relative_pose(self, i: int, j: int) -> np.ndarray:
        """
        Відносна трансформація між кадрами i та j.
        T_rel = inv(T_i) @ T_j
        """
        T_i = self.poses[i]
        T_j = self.poses[j]
        return np.linalg.inv(T_i) @ T_j

    def get_image_path(self, idx: int) -> Path:
        return self.image_paths[idx]

    def summary(self) -> str:
        return (
            f"KITTIOdometryParser\n"
            f"  Sequence  : {self.sequence}\n"
            f"  Camera    : {self.camera}\n"
            f"  Frames    : {self._length}\n"
            f"  Images dir: {self.images_dir}\n"
            f"  Poses file: {self.poses_file}\n"
        )

if __name__ == "__main__":
    DATASET_PATH = "dataset/"
    SEQUENCE     = "00"
    CAMERA       = "image_0"

    parser = KITTIOdometryParser(DATASET_PATH, sequence=SEQUENCE, camera=CAMERA)
    print(parser.summary())

    for idx, (image, pose) in enumerate(parser):
        print(f"[{idx:04d}] image shape: {image.shape}  |  t = {pose[:3, 3]}")
        if idx == 4:
            print("  ...")
            break

    img, T = parser[10]
    print(f"\nКадр 10:")
    print(f"  Зображення : {img.shape}, dtype={img.dtype}")
    print(f"  Поза (4×4):\n{T}")

    T_rel = parser.relative_pose(0, 10)
    print(f"\nВідносна трансформація [0→10]:\n{T_rel}")

    xyz = parser.translations
    print(f"\nТраєкторія: {xyz.shape[0]} точок, X ∈ [{xyz[:,0].min():.1f}, {xyz[:,0].max():.1f}]")