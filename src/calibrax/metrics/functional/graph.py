"""Graph distance metrics -- pure JAX linear algebra on adjacency matrices.

Covers four graph distance categories from the Erlangen Program perspective:

**Between-graph distances** (compare two graphs):
- ``spectral_distance``: Laplacian eigenvalue spectrum comparison
- ``graph_edit_distance_approx``: Spectral relaxation of NP-hard GED

**Within-graph distances** (distance matrix for a single graph):
- ``resistance_distance``: Effective electrical resistance between nodes
- ``shortest_path_distance``: Floyd-Warshall all-pairs shortest paths

All metrics operate on adjacency matrices (square arrays) or graph
Laplacians. No external graph libraries required -- pure JAX.

Registered with ``domain="graph"``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def spectral_distance(
    adj_a: Any,
    adj_b: Any,
    *,
    num_eigenvalues: int | None = None,
) -> Any:
    """Distance between two graphs based on Laplacian eigenvalue spectra.

    Computes ``||lambda(L_G) - lambda(L_H)||_2`` where ``L = D - A`` is
    the graph Laplacian. Invariant to node permutation (graph isomorphism
    invariant). Complexity: O(n^3) for eigendecomposition.

    Args:
        adj_a: Adjacency matrix of first graph, shape (n, n).
        adj_b: Adjacency matrix of second graph, shape (m, m).
        num_eigenvalues: Number of smallest eigenvalues to compare.
            If None, uses all eigenvalues.

    Returns:
        L2 norm of the eigenvalue spectrum difference. Lower is better.
        0.0 for identical graphs.

    Examples:
        >>> import jax.numpy as jnp
        >>> adj = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=jnp.float32)
        >>> spectral_distance(adj, adj)
        0.0
    """
    a = jnp.asarray(adj_a, dtype=jnp.float32)
    b = jnp.asarray(adj_b, dtype=jnp.float32)

    # Compute graph Laplacians: L = D - A
    laplacian_a = jnp.diag(jnp.sum(a, axis=1)) - a
    laplacian_b = jnp.diag(jnp.sum(b, axis=1)) - b

    # Eigenvalues (real and sorted for symmetric matrices)
    eig_a = jnp.linalg.eigvalsh(laplacian_a)
    eig_b = jnp.linalg.eigvalsh(laplacian_b)

    # Select subset of eigenvalues if requested
    if num_eigenvalues is not None:
        eig_a = eig_a[:num_eigenvalues]
        eig_b = eig_b[:num_eigenvalues]

    # Zero-pad shorter spectrum for different-sized graphs
    len_a, len_b = len(eig_a), len(eig_b)
    if len_a < len_b:
        eig_a = jnp.concatenate([eig_a, jnp.zeros(len_b - len_a)])
    elif len_b < len_a:
        eig_b = jnp.concatenate([eig_b, jnp.zeros(len_a - len_b)])

    return jnp.linalg.norm(eig_a - eig_b)


def resistance_distance(adjacency_matrix: Any) -> jnp.ndarray:
    """Resistance distance matrix for a graph.

    The resistance distance between nodes i and j is the effective resistance
    in the electrical network where each edge is a unit resistor:
    ``Omega_{ij} = L^+_{ii} + L^+_{jj} - 2*L^+_{ij}`` where ``L^+`` is
    the Moore-Penrose pseudoinverse of the graph Laplacian.

    A true metric on graph vertices satisfying all metric space axioms.
    Captures more topological information than shortest path -- nodes
    connected by many paths have lower resistance distance.

    Complexity: O(n^3) for pseudoinverse computation.

    Args:
        adjacency_matrix: Square adjacency matrix, shape (n, n).

    Returns:
        Resistance distance matrix, shape (n, n). Symmetric with zero diagonal.

    Examples:
        >>> import jax.numpy as jnp
        >>> adj = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=jnp.float32)
        >>> omega = resistance_distance(adj)
        >>> float(omega[0, 2])  # Path graph: endpoints are 2 apart
        2.0
    """
    adj = jnp.asarray(adjacency_matrix, dtype=jnp.float32)

    # Graph Laplacian: L = D - A
    laplacian = jnp.diag(jnp.sum(adj, axis=1)) - adj

    # Moore-Penrose pseudoinverse
    l_pinv = jnp.linalg.pinv(laplacian)

    # Resistance: Omega[i,j] = L_pinv[i,i] + L_pinv[j,j] - 2*L_pinv[i,j]
    diag = jnp.diag(l_pinv)
    omega = diag[:, None] + diag[None, :] - 2.0 * l_pinv

    # Ensure non-negative (numerical precision)
    return jnp.maximum(omega, 0.0)


def shortest_path_distance(
    adjacency_matrix: Any,
    *,
    weighted: bool = False,
) -> jnp.ndarray:
    """All-pairs shortest path distances via Floyd-Warshall.

    JIT-compatible via ``jax.lax.fori_loop`` with fixed trip count.
    NOT differentiable because ``jnp.minimum`` has zero gradients
    almost everywhere.

    Complexity: O(n^3). Suitable for small graphs (n < 1000).

    Args:
        adjacency_matrix: Square adjacency matrix, shape (n, n).
            Positive values indicate edges. Zero means no edge.
        weighted: If True, uses edge weights. If False, treats
            all nonzero entries as unit edges.

    Returns:
        Shortest path distance matrix, shape (n, n). Unreachable
        pairs have value ``jnp.inf``.

    Examples:
        >>> import jax.numpy as jnp
        >>> adj = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=jnp.float32)
        >>> dist = shortest_path_distance(adj)
        >>> float(dist[0, 2])
        2.0
    """
    adj = jnp.asarray(adjacency_matrix, dtype=jnp.float32)
    n = adj.shape[0]

    if weighted:
        dist = jnp.where(adj > 0, adj, jnp.inf)
    else:
        dist = jnp.where(adj > 0, 1.0, jnp.inf)

    # Zero diagonal
    dist = dist.at[jnp.arange(n), jnp.arange(n)].set(0.0)

    # Floyd-Warshall: vectorized inner loops, fori_loop for outer k
    def body_fn(k: int, d: jnp.ndarray) -> jnp.ndarray:
        # Use indexing instead of slicing for JIT compatibility
        dk_col = d[:, k].reshape(-1, 1)  # (n, 1)
        dk_row = d[k, :].reshape(1, -1)  # (1, n)
        return jnp.minimum(d, dk_col + dk_row)

    return jax.lax.fori_loop(0, n, body_fn, dist)


def graph_edit_distance_approx(adj_a: Any, adj_b: Any) -> Any:
    """Approximate graph edit distance via spectral relaxation.

    The exact GED is NP-hard. This spectral approximation computes:
    (1) eigendecompositions of both adjacency matrices,
    (2) squared eigenvalue difference, and
    (3) 2-hop structural difference via Frobenius norm.

    Provides a lower bound on true GED in polynomial time.
    NOT a true metric (approximation may violate triangle inequality).

    Complexity: O(n^3) for eigendecomposition.

    Args:
        adj_a: Adjacency matrix of first graph, shape (n, n).
        adj_b: Adjacency matrix of second graph, shape (n, n).

    Returns:
        Approximate graph edit distance. Lower is better.
        0.0 for identical graphs.

    Examples:
        >>> import jax.numpy as jnp
        >>> adj = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=jnp.float32)
        >>> graph_edit_distance_approx(adj, adj)
        0.0
    """
    a = jnp.asarray(adj_a, dtype=jnp.float32)
    b = jnp.asarray(adj_b, dtype=jnp.float32)

    # Eigenvalue decomposition of adjacency matrices
    eig_a = jnp.linalg.eigvalsh(a)
    eig_b = jnp.linalg.eigvalsh(b)

    # Eigenvalue difference
    eigenvalue_diff = jnp.sum((eig_a - eig_b) ** 2)

    # 2-hop structural difference
    structural_diff = jnp.linalg.norm(a @ a - b @ b) ** 2

    return jnp.sqrt(eigenvalue_diff + structural_diff)
