"""The cancellation-free geodesic distance on the unit hyperboloid, shared by the hyperbolic metrics.

Private module: :func:`hyperboloid_haversine_distance` is the one implementation behind
``lorentz_distance`` and ``ultrahyperbolic_distance`` at signature ``(1, n)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from calibrax.metrics._utils import _MIN_NORM, floor_at, safe_asinh, safe_hypot, safe_norm


def hyperboloid_haversine_distance(x: jax.Array, y: jax.Array) -> jax.Array:
    """The geodesic distance on the unit hyperboloid, free of catastrophic cancellation.

    Ported from hyperbolix, ``hyperbolix/manifolds/hyperboloid.py`` ``_polar_frame`` and
    ``_dist_stable`` at commit 96aba0558e0b41e88a98b6670fa28928431720a7 (MIT License, Copyright (c)
    2026 Timo Klein, Thomas Lang; the notice is in :mod:`calibrax.metrics._utils`), at curvature 1.

    ``arccosh(-<x, y>_L)`` subtracts two terms of size ``e^(a + b)`` (``a``, ``b`` the points'
    distances from the origin) to leave one of size ``e^d``, so in float32 nearby points far from
    the origin lose every digit. The hyperbolic law of cosines in half-angle (haversine) form is a
    sum of two non-negative terms instead::

        sinh^2(d / 2) = sinh^2((a - b) / 2) + r_x r_y ||x_hat - y_hat||^2 / 4

    with ``r`` a point's spatial radius and ``x_hat`` its spatial direction, and
    ``d = 2 asinh(sinh(d / 2))``. Identical points give exactly 0, with an exactly-zero gradient,
    and ``asinh`` is exact near 0 where ``arccosh`` near 1 is not. hyperbolix measures each
    ordering below; they are kept as it writes them:

    * ``sinh((a - b) / 2) = (u_x - u_y) / (2 sqrt(u_x) sqrt(u_y))`` with ``u = x_0 + r``, the
      numerator formed from the spatial radii, ``(r_x - r_y)(1 + (r_x + r_y)/(x_0 + y_0))``, an
      exact Sterbenz difference rather than a difference of the stored ``x_0``.
    * ``sqrt(u_x)`` and ``sqrt(u_y)`` separately, and ``sqrt(r_x) sqrt(r_y) chord``, so no product
      of two large radii overflows.
    * every norm through the rescaled :func:`~calibrax.metrics._utils.safe_norm`, so a legitimate
      ``1e-34`` chord is not flushed.
    * the radii floored at ``_MIN_NORM`` with a floor, not an exact-zero guard: at the origin the
      two floors cancel in the gradient, which is then the analytic ``-y_hat``.

    Args:
        x: A point on the upper sheet, its timelike coordinate first.
        y: Another.

    Returns:
        The geodesic distance.
    """
    x_time, y_time = x[0], y[0]
    x_spatial, y_spatial = x[1:], y[1:]
    r_x = safe_norm(x_spatial)
    r_y = safe_norm(y_spatial)
    r_x_pos = floor_at(r_x, _MIN_NORM)
    r_y_pos = floor_at(r_y, _MIN_NORM)
    x_hat = x_spatial / r_x_pos
    y_hat = y_spatial / r_y_pos

    u_x = floor_at(x_time + r_x, _MIN_NORM)
    u_y = floor_at(y_time + r_y, _MIN_NORM)
    time_sum = floor_at(x_time + y_time, _MIN_NORM)
    u_gap = (r_x - r_y) * (1.0 + (r_x + r_y) / time_sum)
    representable = jnp.isfinite(u_x) & jnp.isfinite(u_y)
    sinh_half_gap = jnp.where(
        representable,
        0.5 * u_gap / (jnp.sqrt(u_x) * jnp.sqrt(u_y)),
        jnp.where(u_x > u_y, jnp.inf, -jnp.inf),
    )

    chord = safe_norm(x_hat - y_hat)
    q_angular = (0.5 * chord) * (jnp.sqrt(r_x_pos) * jnp.sqrt(r_y_pos))
    return 2.0 * safe_asinh(safe_hypot(sinh_half_gap, q_angular))
