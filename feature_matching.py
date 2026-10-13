"""
Етап 2: Виявлення та зіставлення ключових точок
Feature Extraction & Matching — SIFT / SURF / ORB
"""

import time
import argparse
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

import cv2
import numpy as np
import os
from kitti_parser import KITTIOdometryParser
from droneVideoParser import DroneVideoCSVParser

class DetectorType(Enum):
    SIFT = auto()
    SURF = auto()
    ORB  = auto()
    OPTICAL_FLOW = auto()


@dataclass
class MatchResult:
    """Результат зіставлення між двома кадрами."""
    pts_prev:       np.ndarray
    pts_curr:       np.ndarray
    kp_prev:        list
    kp_curr:        list
    good_matches:   list
    elapsed_ms:     float = 0.0
    detector_name:  str   = ""

def _build_flann_matcher(detector: DetectorType) -> cv2.FlannBasedMatcher:
    """FLANN для SIFT / SURF (float-дескриптори)."""
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def _build_bf_matcher() -> cv2.BFMatcher:
    """Brute-Force для ORB (бінарні дескриптори)."""
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def _lowe_ratio_test(
    matches: list,
    ratio_threshold: float = 0.75,
) -> list:
    """
    Тест Лоу (Lowe's ratio test).
    Залишає пари, де відстань до найкращого збігу
    значно менша, ніж до другого.
    """
    good = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance:
            good.append(pair[0])
    return good


def _matched_points(
    kp_prev: list,
    kp_curr: list,
    good_matches: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Перетворює DMatch-об'єкти на масиви координат."""
    pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good_matches])
    pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in good_matches])
    return pts_prev.reshape(-1, 2), pts_curr.reshape(-1, 2)

class FeatureMatcher:

    def __init__(
        self,
        detector_type: DetectorType = DetectorType.ORB,
        ratio_threshold: float = 0.75,
        orb_n_features: int = 3000,
        surf_hessian_threshold: float = 400.0,
        sift_n_features: int = 0,
        flow_max_corners: int = 3000,
        flow_quality_level: float = 0.01,
        flow_min_distance: float = 7.0,
        flow_block_size: int = 7,
        lk_win_size: Tuple[int, int] = (21, 21),
        lk_max_level: int = 3,
        lk_criteria: Tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
        lk_forward_backward_threshold: Optional[float] = None,
    ):
        self.detector_type   = detector_type
        self.ratio_threshold = ratio_threshold

        self.flow_max_corners = int(flow_max_corners)
        self.flow_quality_level = float(flow_quality_level)
        self.flow_min_distance = float(flow_min_distance)
        self.flow_block_size = int(flow_block_size)
        self.lk_win_size = tuple(lk_win_size)
        self.lk_max_level = int(lk_max_level)
        self.lk_criteria = lk_criteria
        self.lk_forward_backward_threshold = lk_forward_backward_threshold

        self.detector        = self._init_detector(
            detector_type, orb_n_features, surf_hessian_threshold, sift_n_features
        )
        self.matcher         = self._init_matcher(detector_type)

    @staticmethod
    def _init_detector(
        det_type: DetectorType,
        orb_n: int,
        surf_thresh: float,
        sift_n: int,
    ):
        if det_type == DetectorType.SIFT:
            factory = getattr(cv2, "SIFT_create", None)
            if factory is None:
                factory = cv2.xfeatures2d.SIFT_create
            return factory(nfeatures=sift_n)
        elif det_type == DetectorType.SURF:
            return cv2.xfeatures2d.SURF_create(hessianThreshold=surf_thresh)
        elif det_type == DetectorType.ORB:
            return cv2.ORB_create(nfeatures=orb_n)
        elif det_type == DetectorType.OPTICAL_FLOW:
            return None
        else:
            raise ValueError(f"Невідомий тип детектора: {det_type}")

    @staticmethod
    def _init_matcher(det_type: DetectorType):
        if det_type in (DetectorType.SIFT, DetectorType.SURF):
            return _build_flann_matcher(det_type)
        elif det_type == DetectorType.ORB:
            return _build_bf_matcher()
        else:
            return None

    @staticmethod
    def _build_match_visualization_payload(
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
    ) -> Tuple[list, list, list]:
        """Перетворює пари точок на KeyPoint/DMatch для drawMatches."""
        kp_prev = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts_prev]
        kp_curr = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts_curr]
        good_matches = [cv2.DMatch(i, i, 0.0) for i in range(len(pts_prev))]
        return kp_prev, kp_curr, good_matches

    def match(
        self,
        img_prev: np.ndarray,
        img_curr: np.ndarray,
        min_matches: int = 8,
    ) -> Optional[MatchResult]:
        """
        Знаходить та фільтрує збіги між двома сусідніми кадрами.

        Parameters
        ----------
        img_prev : np.ndarray   — кадр t-1 (grayscale або BGR)
        img_curr : np.ndarray   — кадр t   (grayscale або BGR)
        min_matches : int       — мінімальна кількість збігів;
                                  якщо менше — повертає None

        Returns
        -------
        MatchResult або None, якщо збігів недостатньо.
        """
        t0 = time.perf_counter()

        gray_prev = self._to_gray(img_prev)
        gray_curr = self._to_gray(img_curr)

        if self.detector_type == DetectorType.OPTICAL_FLOW:
            pts_prev = cv2.goodFeaturesToTrack(
                gray_prev,
                maxCorners=self.flow_max_corners,
                qualityLevel=self.flow_quality_level,
                minDistance=self.flow_min_distance,
                blockSize=self.flow_block_size,
            )
            if pts_prev is None or len(pts_prev) < min_matches:
                return None

            pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
                gray_prev,
                gray_curr,
                pts_prev,
                None,
                winSize=self.lk_win_size,
                maxLevel=self.lk_max_level,
                criteria=self.lk_criteria,
            )
            if pts_curr is None or status is None:
                return None

            valid = status.ravel() == 1
            valid &= np.isfinite(pts_curr.reshape(-1, 2)).all(axis=1)
            if self.lk_forward_backward_threshold is not None:
                pts_back, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray_curr, gray_prev, pts_curr, None,
                    winSize=self.lk_win_size, maxLevel=self.lk_max_level, criteria=self.lk_criteria)
                if pts_back is None or backward_status is None:
                    return None
                error = np.linalg.norm(pts_back.reshape(-1, 2) - pts_prev.reshape(-1, 2), axis=1)
                valid &= (backward_status.ravel() == 1) & (error <= self.lk_forward_backward_threshold)
            coordinates = pts_curr.reshape(-1, 2)
            valid &= ((coordinates[:, 0] >= 0) & (coordinates[:, 0] < gray_curr.shape[1]) &
                      (coordinates[:, 1] >= 0) & (coordinates[:, 1] < gray_curr.shape[0]))
            pts_prev_valid = pts_prev.reshape(-1, 2)[valid]
            pts_curr_valid = pts_curr.reshape(-1, 2)[valid]

            if len(pts_prev_valid) < min_matches:
                return None

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            kp_prev, kp_curr, good_matches = self._build_match_visualization_payload(
                pts_prev_valid,
                pts_curr_valid,
            )

            return MatchResult(
                pts_prev=pts_prev_valid.astype(np.float32),
                pts_curr=pts_curr_valid.astype(np.float32),
                kp_prev=kp_prev,
                kp_curr=kp_curr,
                good_matches=good_matches,
                elapsed_ms=elapsed_ms,
                detector_name=self.detector_type.name,
            )

        kp_prev, des_prev = self.detector.detectAndCompute(gray_prev, None)
        kp_curr, des_curr = self.detector.detectAndCompute(gray_curr, None)

        if des_prev is None or des_curr is None or len(des_curr) < 2:
            return None
        if len(kp_prev) < min_matches or len(kp_curr) < min_matches:
            return None

        des_prev, des_curr = self._ensure_float(des_prev, des_curr)
        raw_matches = self.matcher.knnMatch(des_prev, des_curr, k=2)

        good_matches = _lowe_ratio_test(raw_matches, self.ratio_threshold)
        unique = {}
        for match in sorted(good_matches, key=lambda item: (item.distance, item.queryIdx, item.trainIdx)):
            unique.setdefault(match.trainIdx, match)
        good_matches = list(unique.values())

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if len(good_matches) < min_matches:
            return None

        pts_prev, pts_curr = _matched_points(kp_prev, kp_curr, good_matches)

        return MatchResult(
            pts_prev      = pts_prev,
            pts_curr      = pts_curr,
            kp_prev       = kp_prev,
            kp_curr       = kp_curr,
            good_matches  = good_matches,
            elapsed_ms    = elapsed_ms,
            detector_name = self.detector_type.name,
        )

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        """BGR → grayscale (якщо вже grayscale — без змін)."""
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _ensure_float(
        self,
        des_prev: np.ndarray,
        des_curr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """FLANN вимагає float32."""
        if self.detector_type in (DetectorType.SIFT, DetectorType.SURF):
            return des_prev.astype(np.float32), des_curr.astype(np.float32)
        return des_prev, des_curr

    def draw_matches(
        self,
        img_prev: np.ndarray,
        img_curr: np.ndarray,
        result: MatchResult,
        max_draw: int = 50,
    ) -> np.ndarray:
        """Повертає зображення з намальованими збігами (для налагодження)."""
        vis = cv2.drawMatches(
            img_prev, result.kp_prev,
            img_curr, result.kp_curr,
            result.good_matches[:max_draw],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        label = (
            f"{result.detector_name} | "
            f"matches: {len(result.good_matches)} | "
            f"{result.elapsed_ms:.1f} ms"
        )
        cv2.putText(vis, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return vis


def build_demo_parser(args):
    if args.dataset_type == "drone":
        if not args.drone_video_path or not args.drone_csv_path:
            raise ValueError("Для drone-режиму потрібні --drone_video_path і --drone_csv_path")
        return DroneVideoCSVParser(
            video_path=args.drone_video_path,
            csv_path=args.drone_csv_path,
            start_frame=args.start_frame,
            time_window_sec=args.time_window_sec,
            video_time_offset_sec=args.video_time_offset_sec,
            use_gimbal_orientation=not args.no_gimbal_orientation,
            fixed_down_pitch_deg=args.fixed_down_pitch_deg,
        )

    return KITTIOdometryParser(args.dataset_path, sequence=args.sequence, camera=args.camera)

if __name__ == "__main__":
    np.random.seed(42)
    dummy = (np.random.rand(480, 640) * 255).astype(np.uint8)

    for det in DetectorType:
        try:
            matcher = FeatureMatcher(det)
            result  = matcher.match(dummy, dummy)
            if result:
                print(f"{det.name:4s}: {len(result.good_matches):4d} збігів | "
                      f"{result.elapsed_ms:.1f} мс")
            else:
                print(f"{det.name:4s}: недостатньо збігів")
        except cv2.error as e:
            print(f"{det.name:4s}: недоступний ({e})")

    parser_args = argparse.ArgumentParser(description="Feature matching demo for KITTI or drone footage")
    parser_args.add_argument("--dataset_type", type=str, default="kitti", choices=["kitti", "drone"])
    parser_args.add_argument("--dataset_path", type=str, default="dataset/")
    parser_args.add_argument("--sequence", type=str, default="00")
    parser_args.add_argument("--camera", type=str, default="image_0")
    parser_args.add_argument("--drone_video_path", type=str, default=None)
    parser_args.add_argument("--drone_csv_path", type=str, default=None)
    parser_args.add_argument("--start_frame", type=int, default=0)
    parser_args.add_argument("--time_window_sec", type=float, default=5.0)
    parser_args.add_argument("--video_time_offset_sec", type=float, default=0.0)
    parser_args.add_argument("--no_gimbal_orientation", action="store_true")
    parser_args.add_argument("--fixed_down_pitch_deg", type=float, default=-90.0)
    parser_args.add_argument("--max_frames", type=int, default=None)
    args = parser_args.parse_args()

    parser = build_demo_parser(args)
    print(parser.summary())

    matcher = FeatureMatcher(DetectorType.OPTICAL_FLOW)

    limit = len(parser) if args.max_frames is None else min(args.max_frames, len(parser))
    for idx in range(limit - 1):
        img_prev, pose_prev = parser[idx]
        img_curr, pose_curr = parser[idx + 1]

        result = matcher.match(img_prev, img_curr)
        if result:
            print(f"[{idx:04d}] Знайдено {len(result.good_matches)} збігів | "
                  f"{result.elapsed_ms:.1f} мс")
            vis = matcher.draw_matches(img_prev, img_curr, result, max_draw=80)
            cv2.imshow("Feature Matches", vis)
        else:
            print(f"[{idx:04d}] Недостатньо збігів")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
