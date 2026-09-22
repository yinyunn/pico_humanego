"""Geometric tests with known metric 3D, independent of real hand labels."""
import unittest

import numpy as np

from tools.pico_import.stereo_geometry import (
    camera_to_head_pair, epipolar_errors, hand_frame_valid, hand_shape_valid,
    interpolate_short_gaps, smooth_valid_points,
    match_hands, stereo_matrices, triangulate_joints,
    triangulate_joints_in_image_cameras,
)
from tools.pico_import.calibration import PicoCalibration
from tools.pico_import.humanego_writer import _midpoint_pose


class StereoGeometryTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[700., 0, 540], [0, 700, 405], [0, 0, 1]])
        self.T = np.eye(4)
        self.T[0, 3] = -.064
        self.xyz = np.array([
            [0, 0, .6], [-.02, .02, .6], [-.035, .04, .6], [-.045, .055, .6], [-.06, .07, .6],
            [-.025, .08, .6], [-.025, .11, .6], [-.025, .13, .6], [-.025, .15, .6],
            [0, .085, .6], [0, .12, .6], [0, .145, .6], [0, .17, .6],
            [.022, .08, .6], [.022, .11, .6], [.022, .13, .6], [.022, .15, .6],
            [.045, .065, .6], [.045, .09, .6], [.045, .11, .6], [.045, .13, .6],
        ])

    def project(self, xyz):
        p = xyz @ self.K.T
        return p[:, :2] / p[:, 2:]

    def pixels(self):
        return self.project(self.xyz), self.project(self.xyz @ self.T[:3, :3].T + self.T[:3, 3])

    def test_metric_recovery_and_reprojection(self):
        left, right = self.pixels()
        tri = triangulate_joints(left, right, self.K, self.K, self.T)
        np.testing.assert_allclose(tri['xyz_left_image_camera_relative'], self.xyz, atol=1e-9)
        self.assertTrue(tri['valid'].all())
        self.assertTrue(hand_shape_valid(tri['xyz_left_image_camera_relative'], tri['valid']))
        self.assertLess(tri['left_reprojection_error_px'].max(), 1e-8)

    def test_nonparallel_camera_rotation(self):
        angle = .08
        self.T[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        left, right = self.pixels()
        tri = triangulate_joints(left, right, self.K, self.K, self.T)
        np.testing.assert_allclose(tri['xyz_left_image_camera_relative'], self.xyz, atol=1e-9)
        self.assertTrue(tri['valid'].all())

    def test_explicit_K_D_inv_E_matches_image_camera_solution(self):
        D = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        D4 = np.eye(4); D4[:3, :3] = D
        # Choose head == left image camera. Since C=E inv(D), E_left=D.
        E_left = D4.copy()
        E_right = np.linalg.inv(self.T) @ D4
        left, right = self.pixels()
        tri = triangulate_joints_in_image_cameras(
            left, right, self.K, self.K, E_left, E_right, D,
        )
        np.testing.assert_allclose(tri['xyz_head'], self.xyz, atol=1e-9)
        np.testing.assert_allclose(tri['xyz_left_image_camera'], self.xyz, atol=1e-9)
        self.assertLess(tri['left_reprojection_error_px'].max(), 1e-8)
        self.assertLess(tri['right_reprojection_error_px'].max(), 1e-8)

    def test_vertical_baseline_mismatch_is_rejected(self):
        left, right = self.pixels()
        wrong = np.eye(4); wrong[1, 3] = .064
        tri = triangulate_joints(left, right, self.K, self.K, wrong)
        self.assertFalse(tri['valid'].any())
        self.assertGreater(np.median(tri['epipolar_error_px']), 50)

    def test_wrong_disparity_sign_is_negative_depth(self):
        left, right = self.pixels()
        tri = triangulate_joints(right, left, self.K, self.K, self.T)
        self.assertTrue((tri['xyz_left_image_camera_relative'][:, 2] < 0).all())
        self.assertFalse(tri['valid'].any())

    def test_zero_disparity_and_bad_joint_rejected(self):
        left, right = self.pixels()
        zero = triangulate_joints(left, left, self.K, self.K, self.T)
        self.assertFalse(zero['valid'].any())
        right[8, 1] += 30
        tri = triangulate_joints(left, right, self.K, self.K, self.T)
        self.assertFalse(tri['valid'][8])
        self.assertFalse(hand_shape_valid(tri['xyz_left_image_camera_relative'], tri['valid']))

    def test_core_frame_can_survive_noncore_joint_loss(self):
        valid = np.ones(21, dtype=bool)
        valid[[7, 11, 15, 19]] = False
        self.assertTrue(hand_frame_valid(self.xyz, valid, min_valid_ratio=.65))
        valid[8] = False
        self.assertFalse(hand_frame_valid(self.xyz, valid, min_valid_ratio=.65))

    def test_short_gap_interpolation_and_no_long_extrapolation(self):
        points = np.repeat(self.xyz[None], 8, axis=0)
        points[:, :, 0] += np.arange(8)[:, None] * .01
        valid = np.ones((8, 21), dtype=bool)
        valid[2:4, 0] = False
        valid[1:7, 1] = False
        filled, mask = interpolate_short_gaps(points, valid, max_gap=2)
        self.assertTrue(mask[2:4, 0].all())
        np.testing.assert_allclose(filled[2:4, 0, 0], [.02, .03])
        self.assertFalse(mask[1:7, 1].any())
        self.assertTrue(np.isnan(filled[1:7, 1]).all())

    def test_symmetric_smoothing_reduces_jitter_without_missing_fill(self):
        points = np.repeat(self.xyz[None], 7, axis=0)
        points[:, 0, 0] += [0, .02, -.02, .02, -.02, .02, 0]
        valid = np.ones((7, 21), dtype=bool)
        valid[3, 1] = False
        smoothed = smooth_valid_points(points, valid, alpha=.5)
        self.assertLess(np.std(smoothed[:, 0, 0]), np.std(points[:, 0, 0]))
        self.assertTrue(np.isnan(smoothed[3, 1]).all())

    def test_missing_baseline_and_nonfinite_pixels_fail(self):
        with self.assertRaises(ValueError):
            stereo_matrices(self.K, self.K, np.eye(4))
        left, right = self.pixels(); left[0, 0] = np.nan
        with self.assertRaises(ValueError):
            triangulate_joints(left, right, self.K, self.K, self.T)

    def test_matching_not_based_on_list_order_and_ambiguity_rejected(self):
        left, right = self.pixels()
        def hand(uv, side):
            return {'hand_side': side, 'score': .95, 'joints': [{'u_eye': p[0], 'v_eye': p[1]} for p in uv]}
        l, r = hand(left, 'Left'), hand(right, 'Left')
        accepted, _ = match_hands([l], [hand(right, 'Right'), r], self.K, self.T)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]['right_id'], 1)
        self.assertEqual(match_hands([l], [], self.K, self.T)[0], [])
        self.assertEqual(match_hands([l], [r, r.copy()], self.K, self.T)[0], [])

    def test_existing_midpoint_frame_preserved_by_index_mapping(self):
        native = np.zeros((26, 3))
        for n, mp in zip((1, 2, 6, 5, 10), (0, 2, 5, 4, 8)):
            native[n] = self.xyz[mp]
        expected = _midpoint_pose(native)
        actual = _midpoint_pose(self.xyz, (0, 2, 5, 4, 8))
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(actual[:3, :3].T @ actual[:3, :3], np.eye(3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(actual[:3, :3]), 1)

    def test_confirmed_basis_conversion_does_not_mutate_calibration(self):
        e1 = np.eye(4); e1[0, 3] = .03
        e0 = np.eye(4); e0[0, 3] = -.03
        c = PicoCalibration.from_header({'cameraIntrinsics': [540, 405, 700, 700], 'cameraExtrinsics': [e0.tolist(), e1.tolist()]})
        before = c.as_dict()
        left, right = camera_to_head_pair(c)
        self.assertEqual(before, c.as_dict())
        Q = np.eye(4); Q[:3, :3] = c.D.T
        np.testing.assert_allclose(left, e1 @ Q)
        np.testing.assert_allclose(right, e0 @ Q)
        raw_relative = np.linalg.inv(e0) @ e1
        image_relative = np.linalg.inv(right) @ left
        expected = np.eye(4)
        expected[:3, :3] = c.D @ raw_relative[:3, :3] @ c.D.T
        expected[:3, 3] = c.D @ raw_relative[:3, 3]
        np.testing.assert_allclose(image_relative, expected)
        self.assertAlmostEqual(np.linalg.norm(image_relative[:3, 3]),
                               np.linalg.norm(raw_relative[:3, 3]))


if __name__ == '__main__':
    unittest.main()
