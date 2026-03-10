"""Tests for graph distance metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from calibrax.metrics.functional.graph import (
    graph_edit_distance_approx,
    resistance_distance,
    shortest_path_distance,
    spectral_distance,
)


# --- Test fixtures ---


def _path_graph(n: int) -> jnp.ndarray:
    """Create adjacency matrix for a path graph with n nodes."""
    adj = jnp.zeros((n, n))
    for i in range(n - 1):
        adj = adj.at[i, i + 1].set(1.0)
        adj = adj.at[i + 1, i].set(1.0)
    return adj


def _complete_graph(n: int) -> jnp.ndarray:
    """Create adjacency matrix for a complete graph with n nodes."""
    return jnp.ones((n, n)) - jnp.eye(n)


def _cycle_graph(n: int) -> jnp.ndarray:
    """Create adjacency matrix for a cycle graph with n nodes."""
    adj = _path_graph(n)
    adj = adj.at[0, n - 1].set(1.0)
    adj = adj.at[n - 1, 0].set(1.0)
    return adj


class TestSpectralDistance:
    """Tests for spectral_distance."""

    def test_identical_graphs(self) -> None:
        """Spectral distance between identical graphs is 0."""
        adj = _path_graph(4)
        result = spectral_distance(adj, adj)
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_known_value(self) -> None:
        """Spectral distance between path and complete 3-node graphs."""
        path = _path_graph(3)
        complete = _complete_graph(3)
        result = spectral_distance(path, complete)
        assert result > 0.0

    def test_symmetric(self) -> None:
        """d(G, H) = d(H, G)."""
        g = _path_graph(4)
        h = _complete_graph(4)
        assert spectral_distance(g, h) == pytest.approx(spectral_distance(h, g), abs=1e-5)

    def test_different_sizes(self) -> None:
        """Zero-padding for different-sized graphs."""
        small = _path_graph(3)
        large = _path_graph(5)
        result = spectral_distance(small, large)
        assert isinstance(result, jax.Array)
        assert result >= 0.0

    def test_permutation_invariant(self) -> None:
        """Same graph with permuted nodes gives same distance."""
        adj = _path_graph(4)
        # Permute: swap nodes 0 and 3
        perm = jnp.array([3, 1, 2, 0])
        adj_perm = adj[perm][:, perm]

        ref = _complete_graph(4)
        d1 = spectral_distance(adj, ref)
        d2 = spectral_distance(adj_perm, ref)
        assert d1 == pytest.approx(d2, abs=1e-5)

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        adj = _path_graph(3)
        result = spectral_distance(adj, adj)
        assert isinstance(result, jax.Array)


class TestResistanceDistance:
    """Tests for resistance_distance."""

    def test_path_graph(self) -> None:
        """Path graph: resistance distance between endpoints = n-1."""
        n = 4
        adj = _path_graph(n)
        omega = resistance_distance(adj)
        # For a path graph, Ω(0, n-1) = n-1
        assert float(omega[0, n - 1]) == pytest.approx(n - 1, abs=0.1)

    def test_complete_graph(self) -> None:
        """Complete graph: Ω(i,j) = 2/n for all i≠j."""
        n = 4
        adj = _complete_graph(n)
        omega = resistance_distance(adj)
        expected = 2.0 / n
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert float(omega[i, j]) == pytest.approx(expected, abs=0.05)

    def test_symmetric_matrix(self) -> None:
        """Resistance distance matrix is symmetric."""
        adj = _cycle_graph(5)
        omega = resistance_distance(adj)
        assert jnp.allclose(omega, omega.T, atol=1e-5)

    def test_zero_diagonal(self) -> None:
        """Diagonal entries are 0."""
        adj = _path_graph(4)
        omega = resistance_distance(adj)
        assert jnp.allclose(jnp.diag(omega), 0.0, atol=1e-5)

    def test_triangle_inequality(self) -> None:
        """Ω(i,k) ≤ Ω(i,j) + Ω(j,k)."""
        adj = _cycle_graph(5)
        omega = resistance_distance(adj)
        n = omega.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert float(omega[i, k]) <= float(omega[i, j] + omega[j, k]) + 1e-5


class TestShortestPathDistance:
    """Tests for shortest_path_distance."""

    def test_path_graph(self) -> None:
        """Distance equals hop count on a path graph."""
        adj = _path_graph(4)
        dist = shortest_path_distance(adj)
        # dist[0, 3] = 3 hops
        assert float(dist[0, 3]) == pytest.approx(3.0, abs=1e-5)
        assert float(dist[0, 1]) == pytest.approx(1.0, abs=1e-5)
        assert float(dist[1, 3]) == pytest.approx(2.0, abs=1e-5)

    def test_complete_graph(self) -> None:
        """All distances = 1 on a complete graph (except diagonal)."""
        n = 4
        adj = _complete_graph(n)
        dist = shortest_path_distance(adj)
        for i in range(n):
            for j in range(n):
                if i == j:
                    assert float(dist[i, j]) == pytest.approx(0.0, abs=1e-5)
                else:
                    assert float(dist[i, j]) == pytest.approx(1.0, abs=1e-5)

    def test_disconnected(self) -> None:
        """Unreachable pairs have infinite distance."""
        # Two disconnected components: nodes {0,1} and {2,3}
        adj = jnp.zeros((4, 4))
        adj = adj.at[0, 1].set(1.0)
        adj = adj.at[1, 0].set(1.0)
        adj = adj.at[2, 3].set(1.0)
        adj = adj.at[3, 2].set(1.0)
        dist = shortest_path_distance(adj)
        assert jnp.isinf(dist[0, 2])
        assert jnp.isinf(dist[0, 3])

    def test_weighted_graph(self) -> None:
        """weighted=True uses edge weights."""
        adj = jnp.zeros((3, 3))
        adj = adj.at[0, 1].set(2.0)
        adj = adj.at[1, 0].set(2.0)
        adj = adj.at[1, 2].set(3.0)
        adj = adj.at[2, 1].set(3.0)
        adj = adj.at[0, 2].set(10.0)
        adj = adj.at[2, 0].set(10.0)

        dist = shortest_path_distance(adj, weighted=True)
        # Shortest 0→2 via 1: 2+3=5 < direct 10
        assert float(dist[0, 2]) == pytest.approx(5.0, abs=1e-5)

    def test_symmetric(self) -> None:
        """dist[i,j] = dist[j,i] for undirected graphs."""
        adj = _cycle_graph(5)
        dist = shortest_path_distance(adj)
        assert jnp.allclose(dist, dist.T, atol=1e-5)


class TestGraphEditDistanceApprox:
    """Tests for graph_edit_distance_approx."""

    def test_identical_graphs(self) -> None:
        """Distance between identical graphs is ~0."""
        adj = _path_graph(4)
        result = graph_edit_distance_approx(adj, adj)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_symmetric(self) -> None:
        """d(G, H) = d(H, G)."""
        g = _path_graph(4)
        h = _complete_graph(4)
        assert graph_edit_distance_approx(g, h) == pytest.approx(
            graph_edit_distance_approx(h, g), abs=1e-5
        )

    def test_more_different_larger(self) -> None:
        """More structural difference → larger distance."""
        base = _path_graph(5)
        # Slightly modified: add one edge
        slight = base.at[0, 2].set(1.0).at[2, 0].set(1.0)
        # Very different: complete graph
        very_diff = _complete_graph(5)

        d_slight = graph_edit_distance_approx(base, slight)
        d_very = graph_edit_distance_approx(base, very_diff)
        assert d_very > d_slight

    def test_returns_jax_scalar(self) -> None:
        """Result should be a JAX scalar array."""
        adj = _path_graph(3)
        result = graph_edit_distance_approx(adj, adj)
        assert isinstance(result, jax.Array)
