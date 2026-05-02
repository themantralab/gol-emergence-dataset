import numpy as np


def step(grid: np.ndarray) -> np.ndarray:
    """Apply one B3/S23 GoL step with fixed-zero (non-toroidal) boundaries."""
    count = np.zeros_like(grid, dtype=np.int8)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            count += np.roll(np.roll(grid, dr, axis=0), dc, axis=1)
    count[0, :] = 0
    count[-1, :] = 0
    count[:, 0] = 0
    count[:, -1] = 0
    birth = (count == 3) & (grid == 0)
    survive = ((count == 2) | (count == 3)) & (grid == 1)
    return (birth | survive).astype(np.uint8)


def step_batch(grids: np.ndarray) -> np.ndarray:
    """Apply one B3/S23 step to a batch of grids simultaneously.

    Input:  (N, H, W) uint8
    Output: (N, H, W) uint8

    All N grids are stepped in a single set of numpy operations — much faster
    than calling step() in a Python loop over N.
    """
    count = np.zeros_like(grids, dtype=np.int8)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            count += np.roll(np.roll(grids, dr, axis=1), dc, axis=2)
    count[:, 0, :] = 0
    count[:, -1, :] = 0
    count[:, :, 0] = 0
    count[:, :, -1] = 0
    birth = (count == 3) & (grids == 0)
    survive = ((count == 2) | (count == 3)) & (grids == 1)
    return (birth | survive).astype(np.uint8)


def simulate(grid: np.ndarray, steps: int = 250) -> np.ndarray:
    """Simulate GoL for `steps` generations.

    Returns (steps+1, H, W) uint8 trajectory including t=0.
    """
    traj = np.empty((steps + 1, grid.shape[0], grid.shape[1]), dtype=np.uint8)
    traj[0] = grid
    for t in range(steps):
        traj[t + 1] = step(traj[t])
    return traj


def simulate_batch(grids: np.ndarray, steps: int = 250) -> np.ndarray:
    """Simulate a batch of grids simultaneously using vectorised stepping.

    Input:  (N, H, W) uint8
    Output: (N, steps+1, H, W) uint8
    """
    N, H, W = grids.shape
    out = np.empty((N, steps + 1, H, W), dtype=np.uint8)
    out[:, 0] = grids
    for t in range(steps):
        out[:, t + 1] = step_batch(out[:, t])
    return out


def run_all_verifications():
    """Verify B3/S23 rules against canonical patterns. Raises AssertionError on failure."""
    results = []

    def embed(pattern):
        grid = np.zeros((64, 64), dtype=np.uint8)
        r, c = pattern.shape
        grid[24:24 + r, 24:24 + c] = pattern
        return grid

    def check(name, passed):
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        results.append((name, passed))

    print("Running GoL verifications...")

    # Block (2×2 still life)
    block = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    g = embed(block)
    g1 = step(g)
    check("Block is still life (step(grid)==grid)", np.array_equal(g, g1))
    check("Block population constant", g.sum() == g1.sum())

    # Blinker (3-cell horizontal oscillator, period 2)
    blinker_h = np.array([[1, 1, 1]], dtype=np.uint8)
    g = embed(blinker_h)
    g2 = step(step(g))
    check("Blinker period-2 (step(step(grid))==grid)", np.array_equal(g, g2))
    check("Blinker population constant", g.sum() == step(g).sum())

    # Glider: after 4 steps it's the same pattern shifted +1 row, +1 col
    glider = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ], dtype=np.uint8)
    g = embed(glider)
    g4 = g.copy()
    for _ in range(4):
        g4 = step(g4)
    orig_window = g[24:29, 24:29]
    shifted_window = g4[25:30, 25:30]
    check("Glider shifts +1,+1 after 4 steps", np.array_equal(orig_window, shifted_window))
    check("Glider population constant", g.sum() == g4.sum())

    # step_batch produces same output as step for a set of grids
    grids = np.stack([embed(block), embed(blinker_h), embed(glider)], axis=0)
    batch_out = step_batch(grids)
    single_out = np.stack([step(grids[i]) for i in range(3)], axis=0)
    check("step_batch matches step() for block/blinker/glider", np.array_equal(batch_out, single_out))

    # Birth rule: dead cell with exactly 3 alive neighbors becomes alive
    birth_grid = np.zeros((64, 64), dtype=np.uint8)
    birth_grid[30, 30] = 1
    birth_grid[30, 31] = 1
    birth_grid[31, 30] = 1
    g1 = step(birth_grid)
    check("Birth rule: dead cell with 3 neighbors becomes alive", g1[31, 31] == 1)

    # Overcrowding: alive cell with 4+ alive neighbors dies
    overcrowd = np.zeros((64, 64), dtype=np.uint8)
    overcrowd[30, 30] = 1
    overcrowd[29, 30] = 1
    overcrowd[31, 30] = 1
    overcrowd[30, 29] = 1
    overcrowd[30, 31] = 1
    g1 = step(overcrowd)
    check("Overcrowding: alive cell with 4 neighbors dies", g1[30, 30] == 0)

    failed = [name for name, ok in results if not ok]
    if failed:
        raise AssertionError(f"Verification failed: {failed}")
    print(f"All {len(results)} verifications passed.")


if __name__ == "__main__":
    run_all_verifications()
