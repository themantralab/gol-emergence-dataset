import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from typing import Iterator
import simulator


class GoLDataset(Dataset):
    """
    PyTorch Dataset for the Stage 1 GoL dataset.

    __getitem__ returns only the initial grid and metadata for a single seed.
    Trajectory simulation happens in the collate function on the full assembled
    batch via simulator.simulate_batch(), which is substantially faster than
    per-item simulation due to numpy vectorisation.

    Large arrays (grids, signatures_norm) are opened with mmap_mode='r' so
    only the pages actually accessed are loaded into RAM. The full 23 GB
    grids.npy and 15 GB signatures_norm.npy are never fully resident.
    """

    def __init__(self, data_dir: str, split: str = 'train', val_fraction: float = 0.1):
        """
        Args:
            data_dir:     path to data/ directory
            split:        'train' or 'val'
            val_fraction: fraction of seeds held out for validation
        """
        data_dir = Path(data_dir)

        # Memory-mapped large arrays — accessed by index, never fully loaded
        self.grids      = np.load(data_dir / 'grids.npy',          mmap_mode='r')
        self.sig_norm   = np.load(data_dir / 'signatures_norm.npy', mmap_mode='r')

        # Small arrays — load fully
        self.buckets    = np.load(data_dir / 'buckets.npy')
        self.labels     = np.load(data_dir / 'labels.npy')
        self.sig_mean   = np.load(data_dir / 'sig_mean.npy')
        self.sig_std    = np.load(data_dir / 'sig_std.npy')
        n_seeds         = int(np.load(data_dir / 'n_seeds.npy'))

        # 90/10 train/val split by index
        n_val   = max(1, int(n_seeds * val_fraction))
        n_train = n_seeds - n_val
        if split == 'train':
            self.indices = np.arange(n_train)
        elif split == 'val':
            self.indices = np.arange(n_train, n_seeds)
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.n_steps = self.sig_norm.shape[1] - 1  # 256

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a lightweight dict for one seed. No simulation.

        The collate function assembles a batch of these dicts and calls
        simulate_batch() once on the stacked grids.
        """
        seed_idx = self.indices[idx]
        return {
            'grid_t0':       self.grids[seed_idx].copy(),      # (128, 128) uint8
            'sig_norm':      self.sig_norm[seed_idx].copy(),   # (257, 10) float32
            'trajectory_id': int(seed_idx),
            'bucket':        int(self.buckets[seed_idx]),
        }


class StratifiedBatchSampler(Sampler):
    """
    Yields batches that draw proportionally from each behavioral bucket.

    The Stage 1 dataset has strongly unequal bucket sizes (dying patterns are
    most common). Without stratification, random batches are dominated by dying
    patterns, which produces degenerate VICReg covariance estimates and gives
    the contrastive loss no meaningful cross-class triplets.

    Each batch is constructed by sampling floor(batch_size / n_buckets) items
    from each bucket, with the remainder filled by sampling from all buckets
    uniformly. This guarantees at least one item per bucket per batch as long
    as batch_size >= n_buckets.
    """

    def __init__(self, dataset: GoLDataset, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle    = shuffle

        # Group dataset indices by bucket value
        buckets = dataset.buckets[dataset.indices]
        bucket_ids = np.unique(buckets)
        self.bucket_indices = {
            int(b): np.where(buckets == b)[0].tolist()
            for b in bucket_ids
        }
        self.n_buckets   = len(self.bucket_indices)
        self.n_samples   = len(dataset)
        self.per_bucket  = batch_size // self.n_buckets
        self.remainder   = batch_size - self.per_bucket * self.n_buckets

    def __len__(self) -> int:
        return self.n_samples // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        # Shuffle within each bucket at the start of each epoch
        bucket_pools = {}
        for b, idxs in self.bucket_indices.items():
            pool = idxs.copy()
            if self.shuffle:
                np.random.shuffle(pool)
            bucket_pools[b] = pool

        bucket_cursors = {b: 0 for b in bucket_pools}

        def _draw(b: int, n: int) -> list[int]:
            pool   = bucket_pools[b]
            cursor = bucket_cursors[b]
            drawn  = []
            while len(drawn) < n:
                end = min(cursor + (n - len(drawn)), len(pool))
                drawn.extend(pool[cursor:end])
                cursor = end
                if cursor >= len(pool):
                    # Reshuffle and wrap
                    if self.shuffle:
                        np.random.shuffle(pool)
                        bucket_pools[b] = pool
                    cursor = 0
            bucket_cursors[b] = cursor
            return drawn

        for _ in range(len(self)):
            batch = []
            for b in bucket_pools:
                batch.extend(_draw(b, self.per_bucket))
            # Fill remainder from all buckets uniformly
            for i in range(self.remainder):
                b = list(bucket_pools.keys())[i % self.n_buckets]
                batch.extend(_draw(b, 1))
            if self.shuffle:
                np.random.shuffle(batch)
            yield batch


def collate_fn(batch: list[dict]) -> dict:
    """
    Assembles a batch of __getitem__ dicts and runs simulate_batch once.

    Simulation happens here rather than in __getitem__ to amortise numpy
    overhead: one vectorised call over B grids is substantially faster than
    B sequential calls. Memory: B * 257 * 128 * 128 * 1 byte ≈ 171 MB for
    B=32 — well within the 32 GB budget.

    Returns:
        trajectories:    (B, 257, 128, 128) uint8 tensor
        sig_norm:        (B, 257, 10)       float32 tensor
        trajectory_ids:  (B,)               int64 tensor
        buckets:         (B,)               int64 tensor
        grids_t0:        (B, 128, 128)      uint8 tensor  (convenience, = trajectories[:, 0])
    """
    grids_t0       = np.stack([item['grid_t0']       for item in batch], axis=0)  # (B, 128, 128)
    sig_norm       = np.stack([item['sig_norm']       for item in batch], axis=0)  # (B, 257, 10)
    trajectory_ids = np.array([item['trajectory_id'] for item in batch])
    buckets        = np.array([item['bucket']        for item in batch])

    # Single vectorised simulation call: (B, 257, 128, 128) uint8
    trajectories = simulator.simulate_batch(grids_t0, steps=256)

    return {
        'trajectories':   torch.from_numpy(trajectories.copy()),
        'sig_norm':       torch.from_numpy(sig_norm.astype(np.float32)),
        'trajectory_ids': torch.from_numpy(trajectory_ids),
        'buckets':        torch.from_numpy(buckets),
        'grids_t0':       torch.from_numpy(grids_t0.copy()),
    }


def make_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    val_fraction: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    num_workers=0 (default): simulation runs in the main process alongside
    the training loop. Increasing num_workers can parallelise simulation but
    requires care — each worker forks the process and holds its own mmap
    handles. Test with num_workers=2 before increasing further.

    Returns:
        train_loader, val_loader
    """
    train_ds = GoLDataset(data_dir, split='train',  val_fraction=val_fraction)
    val_ds   = GoLDataset(data_dir, split='val',    val_fraction=val_fraction)

    train_sampler = StratifiedBatchSampler(train_ds, batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader
