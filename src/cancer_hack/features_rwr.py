"""Fold-local unsupervised Random Walk with Restart patient features.

The first-stage RWR experiment intentionally uses no external network and no
labels.  For every outer fold:

1. binarize the patient-by-gene mutation matrix;
2. build a cosine-weighted co-mutation graph from outer-train rows only;
3. diffuse each patient's L1-normalized mutated-gene seed over that fixed graph;
4. fit TruncatedSVD on outer-train diffusion vectors only;
5. transform validation/test and L2-normalize the final dense vector.

The fit functions accept neither a test matrix nor labels.  That signature is
part of the leakage-prevention contract, not merely a convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def _binarize(matrix: np.ndarray) -> sparse.csr_matrix:
    """enc3 0/1/2 -> mutated/not-mutated CSR matrix."""

    return sparse.csr_matrix((np.asarray(matrix) >= 1).astype(np.float32))


@dataclass(frozen=True)
class RWRGraph:
    """Fold-train co-mutation transition matrix and audit statistics."""

    transition: sparse.csr_matrix
    n_genes: int
    n_active_genes: int
    n_edges: int
    n_isolated_genes: int
    min_gene_support: int
    min_edge_support: int
    topk_neighbors: int


@dataclass(frozen=True)
class RWRBasis:
    """Fold-train RWR graph plus fold-train SVD basis."""

    graph: RWRGraph
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    restart: float
    max_iter: int
    tol: float
    n_components: int
    random_state: int


def fit_rwr_graph(
    train_matrix: np.ndarray,
    train_index: np.ndarray,
    *,
    gene_names: Sequence[str],
    min_gene_support: int = 5,
    min_edge_support: int = 2,
    topk_neighbors: int = 32,
) -> RWRGraph:
    """Fit an internal co-mutation graph from ``train_index`` rows only.

    Edge weight is cosine similarity of two genes' fold-train mutation
    occurrence vectors.  Per-node top-K pruning keeps diffusion tractable.
    Directed top-K selections are symmetrized with ``maximum``.  Genes without
    a retained edge receive an identity transition so seed mass is preserved.
    """

    if min_gene_support < 1 or min_edge_support < 1 or topk_neighbors < 1:
        raise ValueError("RWR support/top-K parameters must be positive")

    binary = _binarize(train_matrix)
    n_genes = binary.shape[1]
    if len(gene_names) != n_genes:
        raise ValueError(
            f"gene_names width {len(gene_names)} != matrix width {n_genes}"
        )
    fit_rows = binary[np.asarray(train_index)]
    support = np.asarray(fit_rows.sum(axis=0)).ravel()
    active = np.where(support >= min_gene_support)[0].astype(np.int64)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    if active.size >= 2:
        active_fit = fit_rows[:, active]
        cooc = (active_fit.T @ active_fit).tocsr()
        cooc.setdiag(0)
        cooc.eliminate_zeros()
        active_support = support[active].astype(np.float64)

        for local_row in range(active.size):
            start, end = cooc.indptr[local_row : local_row + 2]
            local_cols = cooc.indices[start:end]
            counts = cooc.data[start:end]
            keep = counts >= min_edge_support
            if not np.any(keep):
                continue
            local_cols = local_cols[keep]
            counts = counts[keep].astype(np.float64)
            weights = counts / np.sqrt(
                active_support[local_row] * active_support[local_cols]
            )
            if weights.size > topk_neighbors:
                # Weight descending, gene index ascending.  ``argpartition``
                # would choose an arbitrary subset at the K boundary when
                # several edges have equal weight.
                chosen = np.lexsort((local_cols, -weights))[:topk_neighbors]
                local_cols = local_cols[chosen]
                weights = weights[chosen]
            row_parts.append(
                np.full(local_cols.size, active[local_row], dtype=np.int64)
            )
            col_parts.append(active[local_cols].astype(np.int64, copy=False))
            data_parts.append(weights.astype(np.float32))

    if row_parts:
        adjacency = sparse.csr_matrix(
            (
                np.concatenate(data_parts),
                (np.concatenate(row_parts), np.concatenate(col_parts)),
            ),
            shape=(n_genes, n_genes),
            dtype=np.float32,
        )
        adjacency = adjacency.maximum(adjacency.T).tocsr()
    else:
        adjacency = sparse.csr_matrix((n_genes, n_genes), dtype=np.float32)

    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    n_edges = int(adjacency.nnz // 2)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    isolated = np.where(degree <= 0)[0]
    if isolated.size:
        adjacency = adjacency.tolil()
        adjacency[isolated, isolated] = 1.0
        adjacency = adjacency.tocsr()
        degree = np.asarray(adjacency.sum(axis=1)).ravel()

    transition = sparse.diags(
        np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    ) @ adjacency
    transition = transition.astype(np.float32).tocsr()

    return RWRGraph(
        transition=transition,
        n_genes=n_genes,
        n_active_genes=int(active.size),
        n_edges=n_edges,
        n_isolated_genes=int(isolated.size),
        min_gene_support=int(min_gene_support),
        min_edge_support=int(min_edge_support),
        topk_neighbors=int(topk_neighbors),
    )


def propagate_rwr(
    matrix: np.ndarray,
    graph: RWRGraph,
    *,
    restart: float = 0.5,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> np.ndarray:
    """Diffuse each row independently over a fixed fold-train graph."""

    if not 0.0 < restart <= 1.0:
        raise ValueError("restart must be in (0, 1]")
    if max_iter < 1 or tol <= 0:
        raise ValueError("max_iter and tol must be positive")

    binary = _binarize(matrix)
    if binary.shape[1] != graph.n_genes:
        raise ValueError(
            f"matrix width {binary.shape[1]} != graph width {graph.n_genes}"
        )
    seed = normalize(binary, norm="l1", axis=1).toarray().astype(np.float32)
    state = seed.copy()
    walk_weight = np.float32(1.0 - restart)
    restart_weight = np.float32(restart)
    transition_t = graph.transition.T.tocsr()

    for _ in range(max_iter):
        walked = np.asarray(transition_t.dot(state.T).T, dtype=np.float32)
        updated = restart_weight * seed + walk_weight * walked
        delta = float(np.max(np.abs(updated - state))) if updated.size else 0.0
        state = updated
        if delta <= tol:
            break
    return state.astype(np.float32, copy=False)


def fit_rwr_basis(
    train_matrix: np.ndarray,
    train_index: np.ndarray,
    *,
    gene_names: Sequence[str],
    n_components: int = 256,
    restart: float = 0.5,
    max_iter: int = 20,
    tol: float = 1e-6,
    min_gene_support: int = 5,
    min_edge_support: int = 2,
    topk_neighbors: int = 32,
    random_state: int = 0,
) -> RWRBasis:
    """Fit graph and SVD using outer-train rows only; no test or labels."""

    graph = fit_rwr_graph(
        train_matrix,
        train_index,
        gene_names=gene_names,
        min_gene_support=min_gene_support,
        min_edge_support=min_edge_support,
        topk_neighbors=topk_neighbors,
    )
    fit_matrix = np.asarray(train_matrix)[np.asarray(train_index)]
    propagated = propagate_rwr(
        fit_matrix,
        graph,
        restart=restart,
        max_iter=max_iter,
        tol=tol,
    )
    limit = min(propagated.shape)
    if limit < 2:
        components = np.zeros((0, graph.n_genes), dtype=np.float32)
        explained = np.zeros(0, dtype=np.float32)
    else:
        k = max(1, min(int(n_components), limit - 1))
        estimator = TruncatedSVD(
            n_components=k,
            algorithm="randomized",
            random_state=random_state,
        )
        estimator.fit(propagated)
        components = np.asarray(estimator.components_, dtype=np.float32)
        explained = np.asarray(
            estimator.explained_variance_ratio_, dtype=np.float32
        )

    return RWRBasis(
        graph=graph,
        components=components,
        explained_variance_ratio=explained,
        restart=float(restart),
        max_iter=int(max_iter),
        tol=float(tol),
        n_components=int(components.shape[0]),
        random_state=int(random_state),
    )


def transform_rwr_basis(matrix: np.ndarray, basis: RWRBasis) -> np.ndarray:
    """Transform rows with a fixed graph/SVD, then apply burden-safe L2 norm."""

    n_rows = np.asarray(matrix).shape[0]
    if basis.n_components == 0:
        return np.zeros((n_rows, 0), dtype=np.float32)
    propagated = propagate_rwr(
        matrix,
        basis.graph,
        restart=basis.restart,
        max_iter=basis.max_iter,
        tol=basis.tol,
    )
    projected = np.asarray(propagated @ basis.components.T, dtype=np.float32)
    return normalize(projected, norm="l2", axis=1).astype(np.float32)


def build_fold_rwr_block(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    train_index: np.ndarray,
    *,
    gene_names: Sequence[str],
    prefix: str = "rwr__",
    n_components: int = 256,
    restart: float = 0.5,
    max_iter: int = 20,
    tol: float = 1e-6,
    min_gene_support: int = 5,
    min_edge_support: int = 2,
    topk_neighbors: int = 32,
    random_state: int = 0,
) -> tuple[list[str], np.ndarray, np.ndarray, dict, RWRBasis]:
    """Build one fold's RWR block under the fold-only fit contract."""

    basis = fit_rwr_basis(
        train_matrix,
        train_index,
        gene_names=gene_names,
        n_components=n_components,
        restart=restart,
        max_iter=max_iter,
        tol=tol,
        min_gene_support=min_gene_support,
        min_edge_support=min_edge_support,
        topk_neighbors=topk_neighbors,
        random_state=random_state,
    )
    names = [f"{prefix}c{i:03d}" for i in range(basis.n_components)]
    train_out = transform_rwr_basis(train_matrix, basis)
    test_out = transform_rwr_basis(test_matrix, basis)
    diagnostics = {
        "n_genes": basis.graph.n_genes,
        "n_active_genes": basis.graph.n_active_genes,
        "n_edges": basis.graph.n_edges,
        "n_isolated_genes": basis.graph.n_isolated_genes,
        "explained_variance_ratio_sum": float(
            basis.explained_variance_ratio.sum()
        ),
        "train_output_nonzero_rows": int(
            np.count_nonzero(np.linalg.norm(train_out, axis=1))
        ),
        "test_output_nonzero_rows": int(
            np.count_nonzero(np.linalg.norm(test_out, axis=1))
        ),
    }
    return names, train_out, test_out, diagnostics, basis
