"""Run reproducible planar VO experiments using the project's existing matchers."""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from feature_matching import DetectorType, FeatureMatcher
from droneVideoParser import DroneVideoCSVParser
from planar_odometry import (PlanarEstimate, align_and_evaluate, correct_points,
                             estimate_affine, homography_diagnostics, rectification_matrix)


class VideoSource:
    """Sequential video access when GPS telemetry is not available."""
    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.length = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.capture.isOpened() or self.length < 2 or not np.isfinite(self.fps) or self.fps <= 0:
            self.close()
            raise ValueError('Cannot read video, frame count or FPS: {}'.format(path))
        self._sample_times_sec = np.arange(self.length) / self.fps

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ok, frame = self.capture.read()
        if not ok:
            raise IOError('Cannot decode frame {}'.format(index))
        return frame, None

    def close(self):
        self.capture.release()


def validate_config(config):
    expected = {'seed', 'detector', 'nfeatures', 'ratio_threshold', 'lk_forward_backward_threshold_px', 'frame_selection',
                'frame_step', 'adaptive', 'min_matches', 'min_inliers', 'min_inlier_ratio',
                'ransac_threshold_px', 'homography_diagnostics', 'ground_normal_camera', 'camera'}
    if set(config) != expected:
        raise ValueError('Unexpected or missing config keys: {}'.format(sorted(set(config) ^ expected)))
    for key, lower in [('seed', 0), ('nfeatures', 4), ('frame_step', 1),
                       ('min_matches', 4), ('min_inliers', 4)]:
        if type(config[key]) is not int or not lower <= config[key] <= 2147483647:
            raise ValueError('{} must be an integer >= {}'.format(key, lower))
    if config['detector'] not in DetectorType.__members__:
        raise ValueError('Unknown detector')
    if config['frame_selection'] not in ['fixed', 'adaptive']:
        raise ValueError('frame_selection must be fixed or adaptive')
    if not 0 < config['ratio_threshold'] < 1 or not 0 < config['min_inlier_ratio'] <= 1:
        raise ValueError('Invalid ratio threshold')
    if not np.isfinite(config['ransac_threshold_px']) or config['ransac_threshold_px'] <= 0:
        raise ValueError('ransac_threshold_px must be finite and positive')
    if (not np.isfinite(config['lk_forward_backward_threshold_px']) or
            config['lk_forward_backward_threshold_px'] <= 0):
        raise ValueError('lk_forward_backward_threshold_px must be finite and positive')
    adaptive = config['adaptive']
    if set(adaptive) != {'min_gap', 'max_gap', 'target_displacement_px', 'weak_inlier_ratio'}:
        raise ValueError('Invalid adaptive config keys')
    for key in ['min_gap', 'max_gap']:
        if type(adaptive[key]) is not int or adaptive[key] < 1:
            raise ValueError('{} must be a positive integer'.format(key))
    if adaptive['max_gap'] < adaptive['min_gap']:
        raise ValueError('adaptive max_gap must be >= min_gap')
    if (not np.isfinite(adaptive['target_displacement_px']) or adaptive['target_displacement_px'] <= 0 or
            not config['min_inlier_ratio'] <= adaptive['weak_inlier_ratio'] <= 1):
        raise ValueError('Invalid adaptive displacement or weak_inlier_ratio')
    if type(config['homography_diagnostics']) is not bool:
        raise ValueError('homography_diagnostics must be boolean')
    camera = config['camera']
    if set(camera) != {'K', 'distortion'}:
        raise ValueError('camera must contain K and distortion')
    K = None if camera['K'] is None else np.asarray(camera['K'], dtype=np.float64)
    if K is not None and (K.shape != (3, 3) or not np.isfinite(K).all() or
                          K[0, 0] <= 0 or K[1, 1] <= 0 or
                          not np.allclose(K[2], [0, 0, 1]) or abs(np.linalg.det(K)) < 1e-8):
        raise ValueError('camera.K must be a finite invertible 3x3 intrinsic matrix')
    distortion = None if camera['distortion'] is None else np.asarray(camera['distortion'], dtype=np.float64)
    if distortion is not None and (K is None or distortion.ndim != 1 or
                                    distortion.size not in [4, 5, 8, 12, 14] or not np.isfinite(distortion).all()):
        raise ValueError('distortion requires K and a vector of 4/5/8/12/14 coefficients')
    rectification = None
    if config['ground_normal_camera'] is not None:
        if K is None:
            raise ValueError('Rectification requires calibrated camera.K')
        rectification = rectification_matrix(K, config['ground_normal_camera'])
    return K, distortion, rectification


def selection_decision(estimate, gap, config, final=False):
    if config['frame_selection'] == 'fixed':
        return True, 'fixed_step'
    adaptive = config['adaptive']
    if gap >= adaptive['max_gap'] or final:
        return True, 'max_gap' if gap >= adaptive['max_gap'] else 'final_frame'
    if estimate.transform is None:
        return False, 'wait_for_valid_geometry'
    if estimate.median_displacement_px >= adaptive['target_displacement_px']:
        return True, 'target_displacement'
    if estimate.inlier_ratio <= adaptive['weak_inlier_ratio']:
        return True, 'weak_inlier_support'
    return False, 'small_displacement'


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, estimated, reference=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(estimated[:, 0], estimated[:, 1], label='VO')
    ax.scatter(*estimated[0], c='green', label='Start')
    if reference is not None:
        ax.plot(reference[:, 0], reference[:, 1], label='GPS reference')
    else:
        ax.invert_yaxis()
    units = 'm (global GPS alignment for evaluation)' if reference is not None else 'px (first-image plane)'
    ax.set(xlabel='X, ' + units, ylabel='Y, ' + units)
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def run_experiment(source, matcher, config, limit, K=None, distortion=None, rectification=None):
    """Only images and camera calibration affect the estimated trajectory."""
    first, _ = source[0]
    shape = first.shape[:2]
    center = np.array([K[0, 2], K[1, 2], 1.0]) if K is not None else np.array(
        [(shape[1] - 1) / 2, (shape[0] - 1) / 2, 1.0])
    anchor, anchor_index, to_first = first, 0, np.eye(3)
    trajectories = [{'frame': 0, 'time_s': float(source._sample_times_sec[0]),
                     'x_px': 0.0, 'y_px': 0.0, 'status': 'initial'}]
    candidates, diagnostics = [], []
    start = time.perf_counter()
    for index in range(1, limit):
        image, _ = source[index]
        if image.shape[:2] != shape:
            raise ValueError('Image dimensions changed at frame {}'.format(index))
        gap = index - anchor_index
        if config['frame_selection'] == 'fixed' and index % config['frame_step']:
            continue
        if config['frame_selection'] == 'adaptive' and gap < config['adaptive']['min_gap'] and index != limit - 1:
            continue
        candidate_start = time.perf_counter()
        match = matcher.match(anchor, image, min_matches=config['min_matches'])
        if match is None:
            estimate = PlanarEstimate(None, 'insufficient_matches', 0)
        else:
            a = correct_points(match.pts_prev, K, distortion)
            b = correct_points(match.pts_curr, K, distortion)
            if config['homography_diagnostics']:
                diagnostic = homography_diagnostics(a, b, config['ransac_threshold_px'], K)
                diagnostic.update(previous_frame=anchor_index, frame=index)
                diagnostics.append(diagnostic)
            estimate = estimate_affine(correct_points(a, rectification=rectification),
                                       correct_points(b, rectification=rectification), config)
        selected, reason = selection_decision(estimate, gap, config, final=index == limit - 1)
        if selected and estimate.transform is not None:
            candidate_transform = to_first.dot(np.linalg.inv(estimate.transform))
            position = candidate_transform.dot(center)
            if not np.isfinite(position).all() or abs(position[2]) < 1e-8:
                estimate.transform, estimate.status = None, 'invalid_accumulated_transform'
            else:
                to_first = candidate_transform
        candidates.append({'previous_frame': anchor_index, 'frame': index, 'gap': gap,
                           'time_s': float(source._sample_times_sec[index]),
                           'matches': estimate.matches, 'inliers': estimate.inliers,
                           'inlier_ratio': estimate.inlier_ratio,
                           'median_displacement_px': estimate.median_displacement_px,
                           'reprojection_rmse_px': estimate.reprojection_rmse_px,
                           'status': estimate.status, 'selected': selected, 'decision': reason,
                           'processing_ms': (time.perf_counter() - candidate_start) * 1000})
        if selected:
            position = to_first.dot(center)
            position = position[:2] / position[2] - center[:2]
            trajectories.append({'frame': index, 'time_s': float(source._sample_times_sec[index]),
                                 'x_px': float(position[0]), 'y_px': float(position[1]),
                                 'status': estimate.status})
            anchor, anchor_index = image, index
    elapsed = time.perf_counter() - start
    return trajectories, candidates, diagnostics, elapsed, shape


def run(args):
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    for key, value in [('detector', args.detector), ('frame_selection', args.selection), ('frame_step', args.frame_step)]:
        if value is not None:
            config[key] = value
    K, distortion, rectification = validate_config(config)
    if args.max_frames is not None and args.max_frames < 2:
        raise ValueError('--max-frames must be >= 2')
    if not np.isfinite(args.video_start_sec):
        raise ValueError('--video-start-sec must be finite')
    if not Path(args.video).is_file():
        raise ValueError('Video does not exist: {}'.format(args.video))
    output = Path(args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be a new or empty directory')
    cv2.setRNGSeed(config['seed'])
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    matcher = FeatureMatcher(DetectorType[config['detector']], ratio_threshold=config['ratio_threshold'],
                             orb_n_features=config['nfeatures'], sift_n_features=config['nfeatures'],
                             flow_max_corners=config['nfeatures'],
                             lk_forward_backward_threshold=config['lk_forward_backward_threshold_px'])
    source = DroneVideoCSVParser(video_path=args.video, csv_path=args.telemetry,
                                segment_start_times_sec=[args.video_start_sec], camera_matrix=K) if args.telemetry else VideoSource(args.video)
    try:
        limit = len(source) if args.max_frames is None else min(args.max_frames, len(source))
        if args.telemetry and (source._sample_times_sec[0] < source.fly_time_sec[0] or
                               source._sample_times_sec[limit - 1] > source.fly_time_sec[-1]):
            raise ValueError('Telemetry does not cover the selected video interval; check --video-start-sec')
        trajectory, candidates, diagnostics, elapsed, shape = run_experiment(
            source, matcher, config, limit, K, distortion, rectification)
        if not candidates or len(trajectory) < 2:
            raise ValueError('No frame pairs selected; reduce step or increase --max-frames')
        selected = [row for row in candidates if row['selected']]
        accepted = sum(row['status'] == 'ok' for row in selected)
        estimated = np.array([[row['x_px'], row['y_px']] for row in trajectory])
        evaluation, aligned, reference = {'status': 'no_reference'}, None, None
        if args.telemetry:
            indices = [row['frame'] for row in trajectory]
            reference = source._sample_world_xy[indices]
            aligned, evaluation = align_and_evaluate(estimated, reference)
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / 'trajectory.csv', trajectory)
        write_csv(output / 'frames.csv', candidates)
        if args.telemetry:
            rows = [{'frame': row['frame'], 'time_s': row['time_s'],
                     'gps_x_m': float(reference[i, 0]), 'gps_y_m': float(reference[i, 1]),
                     'aligned_x_m': None if aligned is None else float(aligned[i, 0]),
                     'aligned_y_m': None if aligned is None else float(aligned[i, 1])}
                    for i, row in enumerate(trajectory)]
            write_csv(output / 'evaluation.csv', rows)
        saved = dict(config, input={'video': args.video, 'telemetry': args.telemetry,
                                   'video_start_sec': args.video_start_sec, 'max_frames': args.max_frames},
                     image_size={'width': shape[1], 'height': shape[0]},
                     coordinates='first-image plane, origin at principal point or image center; x right, y down',
                     rectification='supplied_constant_ground_normal' if rectification is not None else 'disabled')
        (output / 'config.json').write_text(json.dumps(saved, indent=2) + '\n', encoding='utf-8')
        summary = {'decoded_frames': limit, 'candidate_pairs': len(candidates),
                   'selected_pairs': len(selected), 'accepted_pairs': accepted,
                   'failed_pairs': len(selected) - accepted, 'success_rate': accepted / len(selected),
                   'elapsed_sec': elapsed, 'decoded_frames_per_sec': (limit - 1) / max(elapsed, 1e-9),
                   'mean_candidate_ms': float(np.mean([row['processing_ms'] for row in candidates])),
                   'trajectory_units': 'pixels; approximate planar camera-center trajectory',
                   'evaluation': evaluation, 'python': platform.python_version(),
                   'opencv': cv2.__version__, 'numpy': np.__version__,
                   'source_sha256': {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                     for path in ['research.py', 'planar_odometry.py', 'feature_matching.py', 'droneVideoParser.py']}}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        if config['homography_diagnostics']:
            with (output / 'homography.jsonl').open('w', encoding='utf-8') as stream:
                for diagnostic in diagnostics:
                    stream.write(json.dumps(diagnostic) + '\n')
        save_plot(output / 'trajectory.png', estimated)
        if aligned is not None:
            save_plot(output / 'evaluation.png', aligned, reference)
        print(json.dumps(summary, indent=2))
        return 0 if accepted else 2
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--telemetry', help='Optional DJI CSV, used only for evaluation')
    parser.add_argument('--video-start-sec', type=float, default=0.0, help='Video start in telemetry flight seconds')
    parser.add_argument('--config', default='configs/research.json')
    parser.add_argument('--detector', choices=list(DetectorType.__members__))
    parser.add_argument('--selection', choices=['fixed', 'adaptive'])
    parser.add_argument('--frame-step', type=int)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, TypeError, cv2.error, np.linalg.LinAlgError) as error:
        parser.exit(1, 'Error: {}\n'.format(error))


if __name__ == '__main__':
    raise SystemExit(main())
