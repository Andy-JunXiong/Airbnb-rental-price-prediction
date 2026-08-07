from __future__ import annotations

import unittest

import numpy as np


TRAIN_FRAC = 0.65
CALIBRATION_FRAC = 0.15


def temporal_host_disjoint_split(
    groups: np.ndarray,
    snapshot_indices: np.ndarray,
    unique_snapshots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by snapshot index, enforcing host-disjoint across splits."""
    n_snapshots = len(unique_snapshots)
    train_cut = max(1, int(n_snapshots * TRAIN_FRAC))
    cal_cut = max(train_cut + 1, int(n_snapshots * (TRAIN_FRAC + CALIBRATION_FRAC)))

    train_mask = np.isin(snapshot_indices, unique_snapshots[:train_cut])
    cal_mask = np.isin(snapshot_indices, unique_snapshots[train_cut:cal_cut])
    test_mask = np.isin(snapshot_indices, unique_snapshots[cal_cut:])

    train_hosts = set(groups[train_mask])
    cal_hosts = set(groups[cal_mask]) - train_hosts
    test_hosts = set(groups[test_mask]) - train_hosts - cal_hosts

    cal_mask = cal_mask & np.isin(groups, list(cal_hosts))
    test_mask = test_mask & np.isin(groups, list(test_hosts))

    train_idx = np.where(train_mask)[0]
    cal_idx = np.where(cal_mask)[0]
    test_idx = np.where(test_mask)[0]

    overlap_tc = len(set(groups[train_idx]) & set(groups[cal_idx]))
    overlap_tt = len(set(groups[train_idx]) & set(groups[test_idx]))
    overlap_ct = len(set(groups[cal_idx]) & set(groups[test_idx]))
    if overlap_tc or overlap_tt or overlap_ct:
        raise AssertionError(
            f"Host overlap: train-cal={overlap_tc}, train-test={overlap_tt}, cal-test={overlap_ct}"
        )
    return train_idx, cal_idx, test_idx


class TemporalHostDisjointSplitTest(unittest.TestCase):
    def test_no_host_overlap_across_splits(self) -> None:
        n = 300
        groups = np.array([f"h{i // 3}" for i in range(n)], dtype=object)
        snapshot_indices = np.array([i // 50 for i in range(n)], dtype=int)
        unique = np.unique(snapshot_indices)

        train_idx, cal_idx, test_idx = temporal_host_disjoint_split(
            groups, snapshot_indices, unique
        )

        self.assertGreater(len(train_idx), 0)
        self.assertGreater(len(cal_idx), 0)
        self.assertGreater(len(test_idx), 0)

        train_hosts = set(groups[train_idx])
        cal_hosts = set(groups[cal_idx])
        test_hosts = set(groups[test_idx])

        self.assertEqual(len(train_hosts & cal_hosts), 0)
        self.assertEqual(len(train_hosts & test_hosts), 0)
        self.assertEqual(len(cal_hosts & test_hosts), 0)

    def test_all_snapshots_represented(self) -> None:
        n = 200
        groups = np.array([f"h{i}" for i in range(n)], dtype=object)
        snapshot_indices = np.array([i // 20 for i in range(n)], dtype=int)
        unique = np.unique(snapshot_indices)
        self.assertEqual(len(unique), 10)

        train_idx, cal_idx, test_idx = temporal_host_disjoint_split(
            groups, snapshot_indices, unique
        )

        train_snapshots = set(snapshot_indices[train_idx])
        cal_snapshots = set(snapshot_indices[cal_idx])
        test_snapshots = set(snapshot_indices[test_idx])

        all_snapshots = train_snapshots | cal_snapshots | test_snapshots
        self.assertEqual(all_snapshots, set(unique))


if __name__ == "__main__":
    unittest.main()
