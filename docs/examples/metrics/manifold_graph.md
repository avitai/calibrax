# Manifold and Graph Metrics

> Ran in 2.8 s on 1 cpu (python 3.12.14, jax 0.11.1, substrax 0.1.27, calibrax 0.1.17), from [`examples/metrics/manifold_graph.py`](https://github.com/avitai/calibrax/blob/main/examples/metrics/manifold_graph.py) ([notebook](https://github.com/avitai/calibrax/blob/main/examples/metrics/manifold_graph.ipynb)). Run it from the repository root:

```bash
source ./activate.sh
uv run python examples/metrics/manifold_graph.py
```

| Metadata | Value |
| --- | --- |
| **Tier** | Advanced Guide |
| **Devices** | CPU-compatible |
| **Prerequisites** | The distances example; linear algebra |

## Overview

This example covers calibrax's metrics on structured mathematical objects: graphs, given as
adjacency matrices, and Riemannian manifolds. For graphs it computes the spectral distance, which
compares Laplacian eigenvalue spectra, the shortest-path distance (Floyd-Warshall) and the
resistance distance, the effective resistance of the graph as an electrical network. For manifolds
it computes distances between symmetric positive definite (SPD) matrices, between subspaces on the
Grassmann manifold, and on ultrahyperbolic spaces of mixed signature. These metrics evaluate graph
neural networks, compare covariance matrices (brain-computer interfaces, finance), track
subspaces and embed knowledge graphs in mixed-curvature spaces.

## Learning Goals

1. Compare graph topology with the spectral distance between adjacency matrices
2. Compute shortest-path and resistance distances on a graph
3. Measure the distance between SPD matrices, affine-invariant and log-Euclidean
4. Measure the separation of subspaces on the Grassmann manifold
5. Compute ultrahyperbolic distances under two metric signatures

## Setup


```python
import jax
import jax.numpy as jnp

from calibrax.metrics.functional.graph import (
    resistance_distance,
    shortest_path_distance,
    spectral_distance,
)
from calibrax.metrics.functional.manifold import (
    grassmann_distance,
    spd_affine_invariant_distance,
    spd_log_euclidean_distance,
    ultrahyperbolic_distance,
)
```

## Core Concepts

- A graph's Laplacian spectrum summarises its topology, so two graphs with different spectra are
  structurally different. Resistance distance treats each edge as a unit resistor: two nodes
  joined by several paths are closer than the hop count says.
- The affine-invariant distance between SPD matrices is the geodesic distance of their manifold
  and is unchanged by any congruence, `d(A, B) = d(M A M^T, M B M^T)`; the log-Euclidean distance
  is cheaper and unchanged only by orthogonal ones.
- The Grassmann distance depends only on the subspaces, not on the bases that span them.
- An ultrahyperbolic space of signature `(p, q)` has `p` timelike and `q` spacelike dimensions;
  `(1, n)` is the Lorentz hyperboloid, and mixed signatures model several curvature directions, as
  in UltraE's knowledge-graph embeddings (Xiong et al., KDD 2022).

## Implementation

### Step 1: Compare a path graph with a cycle

Both graphs have four nodes; the cycle adds the edge from node 3 back to node 0.


```python
PATH = jnp.array(
    [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
    dtype=jnp.float32,
)
CYCLE = jnp.array(
    [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
    dtype=jnp.float32,
)


def spectral_distances() -> dict[str, float]:
    """Compute the spectral distance of the path from itself and from the cycle.

    Returns:
        Both distances.
    """
    return {
        "path vs itself": float(spectral_distance(PATH, PATH)),
        "path vs cycle": float(spectral_distance(PATH, CYCLE)),
    }
```

### Step 2: Compute shortest-path and resistance distances


```python
def node_distances(adjacency: jax.Array) -> dict[str, list[list[float]]]:
    """Compute the shortest-path and resistance distances between every pair of nodes.

    Args:
        adjacency: The graph's adjacency matrix.

    Returns:
        Both distance matrices, as nested lists.
    """
    return {
        "shortest path": shortest_path_distance(adjacency).tolist(),
        "resistance": resistance_distance(adjacency).tolist(),
    }
```

### Step 3: Measure distances between SPD matrices

Three 3x3 SPD matrices: the identity, the identity scaled by 4 and an anisotropic matrix.


```python
IDENTITY = jnp.eye(3)
SCALED = jnp.eye(3) * 4.0
ANISOTROPIC = jnp.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 3.0]])
SPD_PAIRS = {
    "d(I, I)": (IDENTITY, IDENTITY),
    "d(I, 4*I)": (IDENTITY, SCALED),
    "d(I, aniso)": (IDENTITY, ANISOTROPIC),
    "d(4*I, aniso)": (SCALED, ANISOTROPIC),
}


def spd_distances() -> dict[str, tuple[float, float]]:
    """Compute the affine-invariant and log-Euclidean distance of each pair.

    Returns:
        Each pair's affine-invariant and log-Euclidean distance.
    """
    return {
        name: (
            float(spd_affine_invariant_distance(a, b)),
            float(spd_log_euclidean_distance(a, b)),
        )
        for name, (a, b) in SPD_PAIRS.items()
    }
```

### Step 4: Measure distances between subspaces

Each subspace of R^4 is given by an orthonormal basis of two columns.


```python
ANGLE = jnp.pi / 6
U1 = jnp.eye(4, 2)
U2, _ = jnp.linalg.qr(
    jnp.array(
        [
            [jnp.cos(ANGLE), -jnp.sin(ANGLE)],
            [jnp.sin(ANGLE), jnp.cos(ANGLE)],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
)
U3 = jnp.eye(4, 2, k=2)
SUBSPACES = {"d(U1, U1)": U1, "d(U1, U2)": U2, "d(U1, U3)": U3}


def grassmann_distances() -> dict[str, float]:
    """Compute the Grassmann distance from U1 to each subspace.

    Returns:
        Each distance.
    """
    return {name: float(grassmann_distance(U1, basis)) for name, basis in SUBSPACES.items()}
```

### Step 5: Compute ultrahyperbolic distances

Points on the `(1, 2)` hyperboloid satisfy `-t^2 + x^2 + y^2 = -1`; points on the `(2, 2)`
pseudo-hyperboloid satisfy `-t1^2 - t2^2 + x1^2 + x2^2 = -1`, here with `t1 = t2`.


```python
def hyperboloid_point(x: float, y: float) -> jax.Array:
    """Lift a point onto the (1, 2) hyperboloid.

    Args:
        x: The first spacelike coordinate.
        y: The second spacelike coordinate.

    Returns:
        The point, its timelike coordinate first.
    """
    return jnp.array([jnp.sqrt(1.0 + x**2 + y**2), x, y])


def pseudo_hyperboloid_point(x1: float, x2: float) -> jax.Array:
    """Lift a point onto the (2, 2) pseudo-hyperboloid with equal timelike coordinates.

    Args:
        x1: The first spacelike coordinate.
        x2: The second spacelike coordinate.

    Returns:
        The point, its two timelike coordinates first.
    """
    t = jnp.sqrt((1.0 + x1**2 + x2**2) / 2.0)
    return jnp.array([t, t, x1, x2])


def ultrahyperbolic_distances() -> dict[str, float]:
    """Compute distances from each space's origin under both signatures.

    Returns:
        Each distance, by signature and point.
    """
    origin, p1, p2 = (hyperboloid_point(*xy) for xy in ((0.0, 0.0), (0.5, 0.3), (1.0, 0.0)))
    origin_22, q1, q2 = (
        pseudo_hyperboloid_point(*xy) for xy in ((0.0, 0.0), (0.3, 0.4), (0.8, 0.6))
    )
    return {
        "(1, 2) d(origin, origin)": float(
            ultrahyperbolic_distance(origin, origin, signature=(1, 2))
        ),
        "(1, 2) d(origin, p1)": float(ultrahyperbolic_distance(origin, p1, signature=(1, 2))),
        "(1, 2) d(origin, p2)": float(ultrahyperbolic_distance(origin, p2, signature=(1, 2))),
        "(2, 2) d(origin, origin)": float(
            ultrahyperbolic_distance(origin_22, origin_22, signature=(2, 2))
        ),
        "(2, 2) d(origin, q1)": float(ultrahyperbolic_distance(origin_22, q1, signature=(2, 2))),
        "(2, 2) d(origin, q2)": float(ultrahyperbolic_distance(origin_22, q2, signature=(2, 2))),
    }
```

## Results

### Graph distances

The path is at spectral distance 0 from itself and not from the cycle. Node 3 is three hops from
node 0 on the path and node 2 two hops on the cycle; in resistance, the cycle's two parallel paths
bring node 2 to 1.0 from node 0, against 3.0 for node 3 on the path's single one.

### Manifold distances

The SPD, Grassmann and ultrahyperbolic tables give each pair's distance; the distance from an
ultrahyperbolic point to itself prints as 0.0001 rather than 0.


```python
def print_matrix(title: str, rows: list[list[float]]) -> None:
    """Print a distance matrix, one node per row.

    Args:
        title: The matrix's title.
        rows: The matrix.
    """
    print(f"  {title}:")
    for node, row in enumerate(rows):
        print(f"    node {node}: {[round(value, 2) for value in row]}")


def main() -> dict[str, float]:
    """Run the example and return its results.

    Returns:
        The spectral distances, the two highlighted node distances and every manifold distance.
    """
    spectral = spectral_distances()
    print("=== Spectral Distance ===")
    for name, value in spectral.items():
        print(f"  {name:16s} {value:.6f}")
    graphs = {"Path": node_distances(PATH), "Cycle": node_distances(CYCLE)}
    for graph, matrices in graphs.items():
        print(f"=== {graph} Graph Distances ===")
        for title, rows in matrices.items():
            print_matrix(title, rows)
    spd = spd_distances()
    print("=== SPD Matrix Distances (affine-invariant, log-Euclidean) ===")
    for name, (affine, log_euclidean) in spd.items():
        print(f"  {name:14s} {affine:.6f}  {log_euclidean:.6f}")
    tables = {
        "Grassmann Subspace Distances": grassmann_distances(),
        "Ultrahyperbolic Distances": ultrahyperbolic_distances(),
    }
    for title, values in tables.items():
        print(f"=== {title} ===")
        for name, value in values.items():
            print(f"  {name:26s} {value:.6f}")
    return {
        **spectral,
        "path resistance d(0, 3)": graphs["Path"]["resistance"][0][3],
        "cycle resistance d(0, 2)": graphs["Cycle"]["resistance"][0][2],
        **{f"affine-invariant {name}": affine for name, (affine, _) in spd.items()},
        **{f"log-Euclidean {name}": log_euclidean for name, (_, log_euclidean) in spd.items()},
        **{name: value for values in tables.values() for name, value in values.items()},
    }


if __name__ == "__main__":
    main()
```

    === Spectral Distance ===
      path vs itself   0.000000
      path vs cycle    1.530734
    === Path Graph Distances ===
      shortest path:
        node 0: [0.0, 1.0, 2.0, 3.0]
        node 1: [1.0, 0.0, 1.0, 2.0]
        node 2: [2.0, 1.0, 0.0, 1.0]
        node 3: [3.0, 2.0, 1.0, 0.0]
      resistance:
        node 0: [0.0, 1.0, 2.0, 3.0]
        node 1: [1.0, 0.0, 1.0, 2.0]
        node 2: [2.0, 1.0, 0.0, 1.0]
        node 3: [3.0, 2.0, 1.0, 0.0]
    === Cycle Graph Distances ===
      shortest path:
        node 0: [0.0, 1.0, 2.0, 1.0]
        node 1: [1.0, 0.0, 1.0, 2.0]
        node 2: [2.0, 1.0, 0.0, 1.0]
        node 3: [1.0, 2.0, 1.0, 0.0]
      resistance:
        node 0: [0.0, 0.75, 1.0, 0.75]
        node 1: [0.75, 0.0, 0.75, 1.0]
        node 2: [1.0, 0.75, 0.0, 0.75]
        node 3: [0.75, 1.0, 0.75, 0.0]
    === SPD Matrix Distances (affine-invariant, log-Euclidean) ===
      d(I, I)        0.000000  0.000000
      d(I, 4*I)      2.401132  2.401132
      d(I, aniso)    1.373887  1.373887
      d(4*I, aniso)  1.747975  1.747975
    === Grassmann Subspace Distances ===
      d(U1, U1)                  0.000000
      d(U1, U2)                  0.000000
      d(U1, U3)                  0.000000
    === Ultrahyperbolic Distances ===
      (1, 2) d(origin, origin)   0.000100
      (1, 2) d(origin, p1)       0.554275
      (1, 2) d(origin, p2)       0.881374
      (2, 2) d(origin, origin)   0.000100
      (2, 2) d(origin, q1)       0.481212
      (2, 2) d(origin, q2)       0.881374


## Next Steps

- [Distances and divergences](distances.md): flat and hyperbolic distances, for comparison
- [Metric learning losses](metric_learning.md): losses on the embedding spaces these metrics
  measure
- [Graph API reference](../../api-reference/metrics/graph.md): every graph metric's signature
- [Manifold API reference](../../api-reference/metrics/manifold.md): every manifold metric's
  signature

<!-- substrax-notebooks: {"page": {"body": "7852f77f94ccb17ef62dc437435c1e3f6df66d3cf9e9c58c8785097cbf858855", "figures": {}, "record": "6787f32d665b03d7ce7131c762f40d42f50eb270f1af4d090bc567748fb09cbe"}, "provenance": {"device": "1 cpu", "outputs": [[], [], [], [], [], [], ["text"]], "seconds": 2.8, "sha256": "fa41fbb9e7e5a6f3f300be1904098a75a32cf80326de44cf788974d96b73858e", "source": "examples/metrics/manifold_graph.py", "versions": {"calibrax": "0.1.17", "jax": "0.11.1", "python": "3.12.14", "substrax": "0.1.27"}}} -->
