"""Moving container walls.

The container (inner box [0,Lx]×[0,Ly]×[0,Lz] in its own frame) translates
vertically in the lab frame:

    z_c(t) = A · s(t) · sin(ω t),   s(t) = ½(1 - cos(π t / τ)) for t < τ, else 1

where t is time since shaking started and τ the ramp time. The ramp avoids
an unphysical velocity jump A·ω at t = 0. All six walls share this motion,
so side walls also move vertically (their velocity enters tangential
friction). Phase is defined as (f·t) mod 1, i.e. phase 0 is the container
passing its mean position moving upward.

Wall ids (also used by the event log): 0 bottom, 1 top, 2 x-, 3 x+, 4 y-, 5 y+.
"""

from __future__ import annotations

import math

from numba import njit

WALL_NAMES = ("bottom", "top", "x-", "x+", "y-", "y+")
N_WALLS = 6


@njit(cache=True, inline="always")
def container_motion(t, amp, omega, ramp):
    """Return (z_c [m], u_c [m/s]) at time t [s] since shaking started."""
    if amp == 0.0 or t <= 0.0:
        return 0.0, 0.0
    sn = math.sin(omega * t)
    cs = math.cos(omega * t)
    if ramp > 0.0 and t < ramp:
        a = math.pi / ramp
        s = 0.5 * (1.0 - math.cos(a * t))
        ds = 0.5 * a * math.sin(a * t)
    else:
        s = 1.0
        ds = 0.0
    return amp * s * sn, amp * (ds * sn + s * omega * cs)


@njit(cache=True, inline="always")
def wall_plane(w, Lx, Ly, Lz, zc):
    """(point coordinate along axis, axis index, inward normal sign) of wall w in the lab frame."""
    if w == 0:
        return zc, 2, 1.0
    elif w == 1:
        return zc + Lz, 2, -1.0
    elif w == 2:
        return 0.0, 0, 1.0
    elif w == 3:
        return Lx, 0, -1.0
    elif w == 4:
        return 0.0, 1, 1.0
    else:
        return Ly, 1, -1.0
