"""Uniform-grid cell list for neighbour search (container frame).

Cells are at least the largest particle diameter wide, so every contact
partner of a particle lies in its own or one of the 26 adjacent cells.
The grid is indexed in the container frame (z relative to the moving
bottom), so it does not need to move with the shaking.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


def make_grid(size: tuple[float, float, float], max_diameter: float) -> tuple[np.ndarray, float]:
    """Return (cells per axis as int64[3], cell size [m])."""
    cs = max_diameter * 1.0001
    dims = np.array([max(1, int(math.floor(L / cs))) for L in size], dtype=np.int64)
    # Recompute so cells tile the box exactly (cells may only get larger).
    cell = max(size[k] / dims[k] for k in range(3))
    return dims, max(cell, cs)


@njit(cache=True, inline="always")
def cell_coord(val, cell, n):
    c = int(math.floor(val / cell))
    if c < 0:
        return 0
    if c >= n:
        return n - 1
    return c


@njit(cache=True)
def build_cells(x, zc, cell, dims, head, nxt):
    """Fill linked-list cell structure. head: (nx*ny*nz,) int32, nxt: (N,) int32."""
    head[:] = -1
    nx, ny, nz = dims[0], dims[1], dims[2]
    for i in range(x.shape[0]):
        cx = cell_coord(x[i, 0], cell, nx)
        cy = cell_coord(x[i, 1], cell, ny)
        cz = cell_coord(x[i, 2] - zc, cell, nz)
        c = (cx * ny + cy) * nz + cz
        nxt[i] = head[c]
        head[c] = i
