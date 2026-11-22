"""Shared approximate density/diversity coresets from an existing frozen train split.

No downstream model, external validation/test access, or fallback sampler is used.
All selection functions receive only the selection pool (never V_core).
"""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import platform
import resource
import time
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.stats import rankdata
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

METHOD = "approximate_density_diversity"
VERSION = "1.1.0"
# Explicit source-schema IDs missing from older task.json files. These are not
# inferred from uniqueness, dtype, or sample statistics; users can add metadata.
TASK_ID_COLUMNS = {
    "dsbench_playground-series-s3e17": ("Product ID",),
    "dsbench_playground-series-s4e1": ("CustomerId",),
    "mlebench_leaf-classification": ("id",),
    "datascibench_dl_10": ("Serial No.",),
    "datascibench_human_10": ("Id",),
    "datascibench_human_5": ("EmployeeNumber",),
    "imcts_colleges": ("UNITID",),
    "uci_credit-card-default": ("ID",),
    "openml_telco-customer-churn": ("customerID",),
    "paper_wild-blueberry": ("Row#",),
}


@dataclass(frozen=True)
class CoresetConfig:
    n_coresets: int = 3
    coreset_ratio: float = .10
    min_coreset_rows: int = 500
    max_coreset_rows: int = 5000
    p: int | None = None
    core_validation_ratio_to_coreset: float = .20
    core_validation_seed: int = 2026
    subset_seeds: tuple = (101, 102, 103)
    score_seed: int = 2026
    regression_bins: int = 5
    fit_max_rows: int = 5000
    max_dimensions: int = 20
    max_categories: int = 32
    batch_size: int = 2048
    n_components: int = 5
    covariance_type: str = "diag"
    max_iter: int = 100
    n_init: int = 1
    reg_covar: float = 1e-6
    similarity_lambda: float = 1.0
    candidate_multiplier: int = 4
    candidate_max_rows: int = 20000
    neighbors: int = 20
    graph_block_size: int = 2000
    threads: int = 1
    id_columns: tuple = ()
    leakage_columns: tuple = ()
    categorical_columns: tuple = ()

    def validate(self):
        positive = ("n_coresets", "min_coreset_rows", "max_coreset_rows", "regression_bins",
                    "fit_max_rows", "max_dimensions", "max_categories", "batch_size",
                    "n_components", "max_iter", "n_init", "candidate_multiplier",
                    "candidate_max_rows", "neighbors", "graph_block_size", "threads")
        for key in positive:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.p is not None and (not isinstance(self.p, int) or isinstance(self.p, bool) or self.p < 2):
            raise ValueError("p must be an integer >= 2 (V_core must be smaller)")
        if not 0 < self.coreset_ratio <= 1 or not 0 < self.core_validation_ratio_to_coreset < 1:
            raise ValueError("Invalid coreset/internal validation ratio")
        if self.min_coreset_rows > self.max_coreset_rows:
            raise ValueError("min_coreset_rows exceeds max_coreset_rows")
        if self.fit_max_rows > 5000 or self.candidate_max_rows > 20000 or self.graph_block_size > 2000:
            raise ValueError("Fit/candidate/graph block limits are 5000/20000/2000")
        if self.max_categories < 2 or self.graph_block_size < 2:
            raise ValueError("max_categories and graph_block_size must be >= 2")
        if len(self.subset_seeds) != self.n_coresets:
            raise ValueError("n_coresets must equal the number of subset_seeds")
        for seed in (*self.subset_seeds, self.score_seed, self.core_validation_seed):
            if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
                raise ValueError("Seeds must be integers in [0, 2**32)")
        if self.covariance_type != "diag" or self.n_init != 1:
            raise ValueError("This version uses a single global diag GMM with n_init=1")
        if not math.isfinite(self.reg_covar) or self.reg_covar <= 0:
            raise ValueError("reg_covar must be finite and positive")
        if not math.isfinite(self.similarity_lambda) or self.similarity_lambda < 0:
            raise ValueError("similarity_lambda must be finite and nonnegative")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def timed(timings, name):
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.) + time.perf_counter() - start


def sizes(n, cfg):
    ratio_request = math.ceil(n * cfg.coreset_ratio)
    requested = cfg.p if cfg.p is not None else min(cfg.max_coreset_rows, max(cfg.min_coreset_rows, ratio_request))
    ratio = cfg.core_validation_ratio_to_coreset
    lo, hi = 0, min(n, requested)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid + max(1, math.ceil(ratio * mid)) <= n:
            lo = mid
        else:
            hi = mid - 1
    p = lo
    v = max(1, math.ceil(ratio * p))
    if not (1 <= v < p and p + v <= n):
        raise ValueError(f"Size constraints conflict: N_train={n}, requested_p={requested}, p={p}, v={v}; require v<p and p+v<=N_train")
    return p, v, {"ratio_requested_p": ratio_request, "requested_p": requested,
                  "requested_v": max(1, math.ceil(ratio * requested)), "actual_p": p,
                  "actual_v": v, "v_over_p": v / p,
                  "adjustment_reason": "training capacity: p + ceil(ratio*p) <= N_train" if p != requested else None}


def quotas(counts, total, lower=None, upper=None):
    """Proportional integer allocation with explicit lower/upper bounds."""
    counts = np.asarray(counts, dtype=np.int64)
    lower = np.ones(len(counts), dtype=np.int64) if lower is None else np.asarray(lower, dtype=np.int64)
    upper = counts.copy() if upper is None else np.asarray(upper, dtype=np.int64)
    if (len(counts) == 0 or np.any(counts <= 0) or np.any(lower < 0)
            or np.any(lower > upper) or np.any(upper > counts)
            or not int(lower.sum()) <= total <= int(upper.sum())):
        raise ValueError(f"Infeasible stratum quotas: total={total}, counts={counts.tolist()}, lower={lower.tolist()}, upper={upper.tolist()}")
    ideal = total * counts / counts.sum()
    q = np.clip(np.floor(ideal).astype(np.int64), lower, upper)
    direction = 1 if q.sum() < total else -1
    heap = []
    for h in range(len(q)):
        eligible = q[h] < upper[h] if direction == 1 else q[h] > lower[h]
        if eligible:
            heapq.heappush(heap, (direction * (q[h] - ideal[h]), h))
    for _ in range(abs(total - int(q.sum()))):
        _, h = heapq.heappop(heap)
        q[h] += direction
        eligible = q[h] < upper[h] if direction == 1 else q[h] > lower[h]
        if eligible:
            heapq.heappush(heap, (direction * (q[h] - ideal[h]), h))
    return q


def strata(y, problem_type, max_bins=5, max_strata=None, min_count=1):
    if y.isna().any():
        raise ValueError("Training target contains missing labels")
    if problem_type == "classification":
        codes, labels = pd.factorize(y, sort=True)
        counts = np.bincount(codes)
        if counts.min() < min_count or (max_strata is not None and len(counts) > max_strata):
            raise ValueError(f"Class coverage conflicts with sizes: counts={counts.tolist()}, max_strata={max_strata}, minimum_per_class={min_count}; cannot enlarge V_core or omit classes")
        return codes, {"kind": "class", "labels": [str(x) for x in labels]}
    if problem_type != "regression":
        raise ValueError(f"Unsupported problem_type: {problem_type}")
    values = pd.to_numeric(y, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Regression targets must be finite")
    for bins in range(min(max_bins, max_strata or max_bins), 1, -1):
        codes, edges = pd.qcut(values, bins, labels=False, retbins=True, duplicates="drop")
        if not np.isfinite(codes).all():
            continue
        codes = pd.factorize(codes, sort=True)[0]
        if np.bincount(codes).min() >= min_count:
            return codes, {"kind": "target_quantile", "requested_bins": max_bins,
                           "actual_bins": len(np.unique(codes)), "edges": edges.tolist(),
                           "small_or_duplicate_bins": "reduce bin count until feasible"}
    return np.zeros(len(y), dtype=np.int64), {"kind": "target_quantile", "requested_bins": max_bins,
        "actual_bins": 1, "edges": [float(values.min()), float(values.max())],
        "small_or_duplicate_bins": "one interval; quantile boundaries collapsed or smaller bins infeasible"}


def stratified_sample(groups, allocation, seed):
    rng = np.random.default_rng(seed)
    selected = [rng.choice(np.flatnonzero(groups == h), int(n), replace=False)
                for h, n in enumerate(allocation)]
    return np.sort(np.concatenate(selected)).astype(np.int64)


def internal_split(y, problem_type, p, v, cfg):
    groups, info = strata(y, problem_type, cfg.regression_bins, min(v, len(y)-v), 2)
    counts = np.bincount(groups)
    allocation = quotas(counts, v, upper=counts-1)
    validation = stratified_sample(groups, allocation, cfg.core_validation_seed)
    pool = np.setdiff1d(np.arange(len(y)), validation, assume_unique=True)
    if p > len(pool):
        raise ValueError("Coreset exceeds selection pool capacity")
    info.update(counts=counts.tolist(), validation_quotas=allocation.tolist())
    return pool, validation, info


def excluded_fields(task, split_manifest, cfg):
    excluded = {task["target"]: "target"}
    for name in TASK_ID_COLUMNS.get(task.get("name"), ()):
        excluded[name] = "explicit source-schema ID registry"
    for source in (task, task.get("metadata", {}), task.get("coreset", {}), split_manifest):
        for key, reason in (("id_columns", "explicit ID"), ("id_column", "explicit ID"),
                            ("leakage_columns", "explicit leakage"), ("exclude_columns", "explicit exclusion")):
            values = source.get(key, [])
            for name in ([values] if isinstance(values, str) else values or []):
                excluded[name] = reason
    if split_manifest.get("alignment_key"):
        excluded[split_manifest["alignment_key"]] = "fixed split alignment ID"
    for name in cfg.id_columns:
        excluded[name] = "configured ID"
    for name in cfg.leakage_columns:
        excluded[name] = "configured leakage"
    identity = " ".join(str(task.get(k, "")) for k in ("name", "task_id", "dataset_id"))
    if "playground-series-s3e17" in identity:
        for name in ("TWF", "HDF", "PWF", "OSF", "RNF"):
            excluded[name] = "S3E17 task-defined failure subtype leakage"
    return excluded


class Representation:
    """Fit only on scorer sample; sparse OHE, bounded dimensional projection."""
    def __init__(self, cfg, excluded, categorical=()):
        self.cfg, self.excluded = cfg, dict(excluded)
        self.categorical = set(categorical) | set(cfg.categorical_columns)

    def numeric_values(self, frame):
        a = frame[self.numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        a[~np.isfinite(a)] = np.nan
        return np.where(np.isnan(a), self.medians, a)

    def category_values(self, frame):
        # Prefix distinguishes actual string values from the missing sentinel.
        return frame[self.categories].map(lambda x: "missing:" if pd.isna(x) else "value:" + str(x))

    def encoded(self, frame):
        parts = []
        if self.numeric:
            parts.append(sparse.csr_matrix(self.scaler.transform(self.numeric_values(frame))))
        if self.categories:
            parts.append(self.encoder.transform(self.category_values(frame)))
        return sparse.hstack(parts, format="csr")

    def fit(self, frame):
        self.numeric, self.categories = [], []
        for column in frame:
            if column in self.excluded:
                continue
            values = frame[column]
            converted = pd.to_numeric(values, errors="coerce")
            numeric = column not in self.categorical and converted.notna().sum() == values.notna().sum()
            cleaned = converted.replace([np.inf, -np.inf], np.nan) if numeric else values
            if cleaned.nunique(dropna=True) <= 1:
                self.excluded[column] = "constant or all missing in scorer fit sample"
            elif numeric:
                self.numeric.append(column)
            else:
                self.categories.append(column)
        if not self.numeric and not self.categories:
            raise ValueError("No effective features after target/ID/leakage/constant exclusions")
        if self.numeric:
            values = frame[self.numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            self.medians = values.median().to_numpy()
            self.scaler = StandardScaler().fit(self.numeric_values(frame))
        if self.categories:
            self.encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=True,
                                         max_categories=self.cfg.max_categories, dtype=np.float64)
            self.encoder.fit(self.category_values(frame))
        matrix = self.encoded(frame)
        d = min(self.cfg.max_dimensions, matrix.shape[0]-1, matrix.shape[1])
        if d < 1:
            raise ValueError("Insufficient fit rows/features for representation")
        self.reducer = None
        if matrix.shape[1] > d:
            self.reducer = TruncatedSVD(n_components=d, random_state=self.cfg.score_seed)
            reduced = self.reducer.fit_transform(matrix)
        else:
            reduced = matrix.toarray()  # At most max_dimensions columns, never dense high-dimensional OHE.
        # Small SVD rotates/trims the projected representation to its effective rank.
        self.center = reduced.mean(axis=0)
        _, singular, vt = np.linalg.svd(reduced - self.center, full_matrices=False)
        tolerance = max(reduced.shape) * np.finfo(float).eps * singular[0]
        rank = int(np.sum(singular > max(tolerance, 1e-12)))
        if rank == 0:
            raise ValueError("No nonconstant effective representation dimensions")
        self.rotation = vt[:rank].T
        return {"numeric_columns": self.numeric, "categorical_columns": self.categories,
                "excluded_fields": self.excluded, "encoded_dimensions": matrix.shape[1],
                "dimensions": rank, "projection": "TruncatedSVD + effective-rank rotation" if self.reducer else "effective-rank rotation",
                "numeric_processing": "infinity to missing; fit-sample median and StandardScaler",
                "category_processing": "sparse capped one-hot; unknown is infrequent or all zero",
                "rank_tolerance": max(tolerance, 1e-12)}

    def transform(self, frame):
        matrix = self.encoded(frame)
        values = self.reducer.transform(matrix) if self.reducer else matrix.toarray()
        result = (values - self.center) @ self.rotation
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite representation")
        return result


def score_pool(pool, task, excluded, cfg, timings):
    """The only scoring-model fit is one global GMM. No V_core argument."""
    groups, stratum_info = strata(pool[task["target"]], task["problem_type"], cfg.regression_bins)
    counts = np.bincount(groups)
    fit_size = min(len(pool), cfg.fit_max_rows)
    fit_indices = stratified_sample(groups, quotas(counts, fit_size), cfg.score_seed)
    with timed(timings, "representation_fit_and_transform"):
        rep = Representation(cfg, excluded, task.get("categorical_columns", []))
        rep_info = rep.fit(pool.iloc[fit_indices])
        z = np.empty((len(pool), rep_info["dimensions"]), dtype=np.float64)
        for start in range(0, len(pool), cfg.batch_size):
            z[start:start+cfg.batch_size] = rep.transform(pool.iloc[start:start+cfg.batch_size])
    with timed(timings, "gmm_fit_and_score"):
        components = min(cfg.n_components, fit_size)
        gmm = GaussianMixture(n_components=components, covariance_type=cfg.covariance_type,
                              max_iter=cfg.max_iter, n_init=cfg.n_init, reg_covar=cfg.reg_covar,
                              random_state=cfg.score_seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            gmm.fit(z[fit_indices])
        if not gmm.converged_:
            raise ValueError(f"GMM did not converge in {cfg.max_iter} iterations; no fallback")
        density = np.empty(len(pool))
        for start in range(0, len(pool), cfg.batch_size):
            density[start:start+cfg.batch_size] = gmm.score_samples(z[start:start+cfg.batch_size])
        if not np.isfinite(density).all():
            raise ValueError("GMM returned nonfinite log density")
        importance = np.empty(len(pool))
        for h, count in enumerate(counts):
            mask = groups == h
            importance[mask] = (rankdata(density[mask], method="average")-1)/(count-1) if count > 1 else .5
    info = {"representation": rep_info, "strata": stratum_info, "fit_rows": fit_size,
            "gmm": {"requested_components": cfg.n_components, "actual_components": components,
                    "adjustment_reason": "limited by scorer fit rows" if components != cfg.n_components else None,
                    "converged": True, "iterations": int(gmm.n_iter_), "covariance_type": cfg.covariance_type,
                    "warnings": [str(w.message) for w in caught]},
            "I_definition": "within-stratum (average_rank(log_density)-1)/(n_h-1); singleton=0.5; density representativeness, not validated learning contribution"}
    return {"representation": z, "log_density": density, "I": importance,
            "strata": groups, "fit_pool_positions": fit_indices}, info


def similarity_graph(z, cfg, seed):
    """Bounded block-local kNN; cross-block similarities are intentionally omitted."""
    permutation = np.random.default_rng(seed).permutation(len(z))
    rows, cols, distances = [], [], []
    for start in range(0, len(z), cfg.graph_block_size):
        positions = permutation[start:start+cfg.graph_block_size]
        block = z[positions]
        k = min(cfg.neighbors, len(block)-1)
        if k == 0:
            continue
        nn = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric="euclidean", n_jobs=1).fit(block)
        # Query chunks bound even temporary pairwise arrays to <=256 x block_size.
        for begin in range(0, len(block), 256):
            dist, indices = nn.kneighbors(block[begin:begin+256])
            for local, (ds, ns) in enumerate(zip(dist, indices), start=begin):
                valid = ns != local
                ns, ds = ns[valid][:k], ds[valid][:k]
                rows.extend([int(positions[local])] * len(ns))
                cols.extend(positions[ns].tolist())
                distances.extend(ds.tolist())
    distances = np.asarray(distances)
    nonzero = distances[distances > 0]
    bandwidth = float(np.median(nonzero)) if len(nonzero) else 1.
    weights = np.exp(-.5 * (distances / bandwidth)**2)
    graph = sparse.csr_matrix((weights, (rows, cols)), shape=(len(z), len(z)))
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph, {"kind": "random reproducible block-local symmetric kNN",
                   "approximation": "cross-block similarities omitted; not exact global kNN",
                   "block_seed": seed, "block_size": cfg.graph_block_size,
                   "blocks": math.ceil(len(z)/cfg.graph_block_size), "neighbors_before_symmetrizing": cfg.neighbors,
                   "bandwidth": bandwidth, "bandwidth_rule": "median positive directed-neighbor distance; 1 if none",
                   "all_neighbor_distances_zero": len(nonzero) == 0, "nnz": int(graph.nnz),
                   "max_distance_query_rows": 256, "sparse_bytes": graph.data.nbytes + graph.indices.nbytes + graph.indptr.nbytes}


def objective(importance, graph, chosen, p, penalty, *, normalization_degree=None):
    """Default preserves the original dense-pair normalization for GMM coresets."""
    degree = p-1 if normalization_degree is None else min(normalization_degree, p-1)
    if p < 2 or not np.isfinite(degree) or degree <= 0:
        raise ValueError("Objective requires p>=2 and a positive normalization degree")
    return float(importance[chosen].sum()/p - penalty * graph[chosen][:, chosen].sum()/(p*degree))


def greedy_select(importance, graph, groups, allocation, penalty, *, normalization_degree=None, initial=None):
    """Exact incremental gains for this approximate graph, including negative gains."""
    p = int(sum(allocation))
    if p < 2:
        raise ValueError("Greedy objective requires p>=2")
    degree = p-1 if normalization_degree is None else min(normalization_degree, p-1)
    if not np.isfinite(degree) or degree <= 0:
        raise ValueError("Greedy objective requires a positive normalization degree")
    remaining = np.asarray(allocation).copy()
    similarity_sum = np.zeros(len(importance))
    selected = np.zeros(len(importance), dtype=bool)
    versions = np.zeros(len(importance), dtype=np.int64)
    initial = np.asarray([] if initial is None else initial, dtype=np.int64)
    if (len(initial) > p or len(np.unique(initial)) != len(initial)
            or np.any(initial < 0) or np.any(initial >= len(importance))):
        raise ValueError("Invalid initial coreset")
    remaining -= np.bincount(np.asarray(groups)[initial], minlength=len(remaining))
    if np.any(remaining < 0):
        raise ValueError("Initial coreset exceeds final quotas")
    selected[initial] = True
    similarity_sum = np.asarray(graph[:, initial].sum(axis=1)).ravel()
    factor = 2*penalty/(p*degree)
    heap = [(-float(value/p-factor*similarity_sum[i]), i, 0)
            for i, value in enumerate(importance) if not selected[i]]
    heapq.heapify(heap)
    chosen, gains = initial.tolist(), []
    initial_value = (float(np.sum(np.asarray(importance)[initial]))/p -
                     float(penalty*graph[initial][:,initial].sum()/(p*degree)))
    while len(chosen) < p:
        if not heap:
            raise ValueError("Candidate quotas cannot fill requested coreset")
        negative_gain, i, version = heapq.heappop(heap)
        if selected[i] or remaining[groups[i]] == 0 or version != versions[i]:
            continue
        selected[i] = True
        remaining[groups[i]] -= 1
        chosen.append(i)
        gains.append(-negative_gain)
        start, end = graph.indptr[i:i+2]
        for j, weight in zip(graph.indices[start:end], graph.data[start:end]):
            if not selected[j] and remaining[groups[j]]:
                similarity_sum[j] += weight
                versions[j] += 1
                gain = importance[j]/p - factor*similarity_sum[j]
                heapq.heappush(heap, (-float(gain), int(j), int(versions[j])))
    chosen = np.asarray(chosen, dtype=np.int64)
    value = objective(importance, graph, chosen, p, penalty, normalization_degree=normalization_degree)
    if not np.isclose(initial_value+sum(gains), value, rtol=1e-9, atol=1e-10):
        raise ValueError("Incremental gain sum differs from objective")
    report = {"objective": value,
              "similarity_penalty": float(penalty * graph[chosen][:, chosen].sum()/(p*degree)),
              "gain_sum": float(initial_value+sum(gains)), "initial_objective": initial_value,
              "initial_count": len(initial), "supplement_gain_sum": float(sum(gains)), "negative_gain_steps": int(np.sum(np.asarray(gains) < 0))}
    if normalization_degree is None:
        report["density_term"] = float(importance[chosen].mean())
    else:
        report.update(importance_term=float(importance[chosen].mean()),
                      normalization="sparse_degree", normalization_degree=float(degree),
                      normalization_denominator=float(p*degree))
    return chosen, report


def cached_arrays(directory, key):
    meta_path, array_path = directory/f"{key}.json", directory/f"{key}.npz"
    if not meta_path.exists() or not array_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta["sha256"] != file_hash(array_path):
        raise ValueError(f"Corrupt cache artifact: {array_path}")
    with np.load(array_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return arrays, meta["info"]


def save_cache(directory, key, arrays, info):
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory/f"{key}.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    path = directory/f"{key}.npz"
    temporary.replace(path)
    write_json(directory/f"{key}.json", {"sha256": file_hash(path), "info": info})


def export_rows(train_path, outputs):
    """Stream CSV cells verbatim through csv.reader/writer, avoiding pandas coercion."""
    with ExitStack() as stack:
        source = stack.enter_context(train_path.open(newline="", encoding="utf-8"))
        reader = csv.reader(source)
        header = next(reader)
        writers, positions = [], []
        for path, indices in outputs:
            stream = stack.enter_context(path.open("w", newline="", encoding="utf-8"))
            writer = csv.writer(stream)
            writer.writerow(header)
            writers.append(writer)
            positions.append(set(map(int, indices)))
        for position, row in enumerate(reader):
            for writer, wanted in zip(writers, positions):
                if position in wanted:
                    writer.writerow(row)


def build_coresets(task_path, output, config=None, use_cache=True):
    """Build one frozen task; failed builds have manifest/report and raise an error."""
    cfg = config or CoresetConfig()
    task_path, output = Path(task_path).resolve(), Path(output).resolve()
    task = json.loads(task_path.read_text())
    train_path = (task_path.parent/task.get("files", {}).get("train", "train.csv")).resolve()
    # Refuse destinations that could overwrite any existing input artifacts.
    if output == task_path.parent or output in task_path.parents or output == train_path.parent or output in train_path.parents:
        raise ValueError("Output must not be an input directory or its ancestor")
    if output.exists() and any(output.iterdir()):
        old_manifest = output/"manifest.json"
        if not old_manifest.exists() or json.loads(old_manifest.read_text()).get("method") != METHOD:
            raise ValueError("Nonempty output is not an owned coreset directory")
        for pattern in ("coreset_*.csv", "coreset_*_indices.json", "candidate_*_indices.json",
                        "core_validation.csv", "core_validation_indices.json", "selection_pool_indices.json", "scores.npz"):
            for path in output.glob(pattern):
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timing_names = ("input", "internal_split", "representation_fit_and_transform", "gmm_fit_and_score",
                    "candidate_pool", "similarity_graph", "selection_optimization", "cache_io", "file_output")
    timings = dict.fromkeys(timing_names, 0.)
    dependencies = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    manifest = {"method": METHOD, "method_version": VERSION, "status": "running", "task_path": str(task_path),
                "task": task, "configuration": asdict(cfg), "dependencies": dependencies,
                "cache_enabled": use_cache, "scope": "coreset construction and internal validation preparation only; no predictive validation"}
    report = {"status": "running", "timings_seconds": timings, "coresets": []}
    write_json(output/"manifest.json", manifest)
    write_json(output/"report.json", report)
    stage = "input"
    try:
        cfg.validate()
        with timed(timings, "input"):
            if task.get("split", {}).get("strategy") != "frozen":
                raise ValueError("Expected frozen task; use the existing build_splits.py workflow first")
            source_manifest = json.loads((task_path.parent/"split_manifest.json").read_text())
            unsigned = {k: v for k, v in source_manifest.items() if k != "split_id"}
            # Match the existing manifest's serialization, without opening validation/test.
            signed = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
            if source_manifest.get("split_id") != signed:
                raise ValueError("Fixed split manifest identity mismatch")
            fingerprint = file_hash(train_path)
            for path, actual in ((train_path, fingerprint), (task_path, file_hash(task_path))):
                expected = source_manifest.get("sha256", {}).get(path.name)
                if expected is not None and expected != actual:
                    raise ValueError(f"Frozen input fingerprint mismatch: {path.name}")
            frame = pd.read_csv(train_path, low_memory=False)
            if len(frame) != source_manifest["rows"]["train"]:
                raise ValueError("Frozen training row count mismatch")
            if task["seed"] != source_manifest["seed"]:
                raise ValueError("Frozen task and split seed mismatch")
            source_indices = source_manifest.get("indices", {}).get("train")
            if source_indices is not None:
                source_indices = np.asarray(source_indices, dtype=np.int64)
                if len(source_indices) != len(frame) or len(np.unique(source_indices)) != len(frame) or np.any(source_indices < 0):
                    raise ValueError("Invalid original training-row mapping")
            manifest.update(split_id=source_manifest["split_id"], train_sha256=fingerprint,
                            train_path=str(train_path), train_rows=len(frame))
        stage = "internal_split"
        with timed(timings, stage):
            from .prepared_protocol import partition_coreset_prepared
            p, v, size_info = sizes(len(frame), cfg)
            manifest["sizes"] = size_info
            pool_positions, validation_positions, split_info = partition_coreset_prepared(
                frame[task["target"]], task, task_path, source_manifest, p, v, cfg)
            if len(validation_positions) != v:
                reason = f"whole protocol units: validation rows rounded up from {v} to {len(validation_positions)}"
                previous = size_info["adjustment_reason"]
                size_info["adjustment_reason"] = previous + "; " + reason if previous else reason
            v = len(validation_positions)
            size_info.update(actual_v=v, v_over_p=v/p)
            internal_id = digest({"train": fingerprint, "parent_split": source_manifest["split_id"],
                                  "pool": pool_positions.tolist(), "validation": validation_positions.tolist()})
            manifest.update(internal_split_id=internal_id, internal_split=split_info)
            pool = frame.iloc[pool_positions].reset_index(drop=True)
            del frame  # The scorer below has no access to V_core content.
        def index_record(positions):
            positions = np.sort(positions)
            return {"train_positions": positions.tolist(),
                    "source_positions": source_indices[positions].tolist() if source_indices is not None else None,
                    "source_reference": source_manifest.get("index_reference", {}).get("train"),
                    "split_id": source_manifest["split_id"], "internal_split_id": internal_id}
        with timed(timings, "file_output"):
            write_json(output/"core_validation_indices.json", index_record(validation_positions))
            write_json(output/"selection_pool_indices.json", index_record(pool_positions))
            export_rows(train_path, [(output/"core_validation.csv", validation_positions)])
        excluded = excluded_fields(task, source_manifest, cfg)
        score_fields = ("score_seed", "regression_bins", "fit_max_rows", "max_dimensions", "max_categories", "batch_size",
                        "n_components", "covariance_type", "max_iter", "n_init", "reg_covar", "categorical_columns", "threads")
        score_identity = {"version": VERSION, "dependencies": dependencies, "train": fingerprint,
                          "internal_split": internal_id, "task_target": task["target"], "problem_type": task["problem_type"],
                          "excluded": excluded, "task_categories": task.get("categorical_columns", []),
                          "config": {k: getattr(cfg, k) for k in score_fields}}
        score_key = digest(score_identity)
        manifest["score_cache_key"] = score_key
        stage = "scoring"
        with threadpool_limits(limits=cfg.threads):
            with timed(timings, "cache_io"):
                cache = cached_arrays(output/"cache", score_key) if use_cache else None
            if cache is None:
                arrays, score_info = score_pool(pool, task, excluded, cfg, timings)
                if use_cache:
                    with timed(timings, "cache_io"):
                        save_cache(output/"cache", score_key, arrays, score_info)
            else:
                arrays, score_info = cache
            report["score_cache_hit"] = cache is not None
            manifest["scoring"] = score_info
            z, importance, groups = arrays["representation"], arrays["I"], arrays["strata"]
            if len(z) != len(pool_positions) or not np.isfinite(z).all() or not np.isfinite(arrays["log_density"]).all() or not np.isfinite(importance).all() or np.any((importance < 0) | (importance > 1)):
                raise ValueError("Invalid scoring arrays")
            counts = np.bincount(groups)
            allocation = quotas(counts, p)
            manifest["strata"] = {"counts": counts.tolist(), "coreset_quotas": allocation.tolist()}
            with timed(timings, "file_output"):
                np.savez_compressed(output/"scores.npz", train_positions=pool_positions,
                    source_positions=source_indices[pool_positions] if source_indices is not None else np.array([], dtype=int),
                    log_density=arrays["log_density"], I=importance, strata=groups,
                    scorer_fit_train_positions=pool_positions[arrays["fit_pool_positions"]])
            selected_positions = []
            selection_fields = ("candidate_multiplier", "candidate_max_rows", "neighbors", "graph_block_size", "similarity_lambda")
            for subset, seed in enumerate(cfg.subset_seeds):
                stage = f"coreset_{subset}"
                selection_key = digest({"score_key": score_key, "p": p, "quotas": allocation.tolist(), "seed": seed,
                                        "config": {k: getattr(cfg, k) for k in selection_fields}})
                with timed(timings, "cache_io"):
                    cache = cached_arrays(output/"cache", selection_key) if use_cache else None
                if cache is None:
                    with timed(timings, "candidate_pool"):
                        m = min(len(pool), cfg.candidate_multiplier*p, cfg.candidate_max_rows)
                        candidate_quotas = quotas(counts, m, lower=allocation)
                        candidate = stratified_sample(groups, candidate_quotas, seed)
                    with timed(timings, "similarity_graph"):
                        graph, graph_info = similarity_graph(z[candidate], cfg, seed)
                    with timed(timings, "selection_optimization"):
                        chosen, opt_info = greedy_select(importance[candidate], graph, groups[candidate], allocation, cfg.similarity_lambda)
                        chosen_pool = np.sort(candidate[chosen])
                    selection_info = {"seed": seed, "candidate_rows": m, "candidate_quotas": candidate_quotas.tolist(),
                                      "graph": graph_info, "optimization": opt_info}
                    if use_cache:
                        with timed(timings, "cache_io"):
                            save_cache(output/"cache", selection_key, {"candidate": candidate, "chosen_pool": chosen_pool}, selection_info)
                else:
                    saved, selection_info = cache
                    candidate, chosen_pool = saved["candidate"], saved["chosen_pool"]
                if (len(chosen_pool) != p or len(np.unique(chosen_pool)) != p or np.any(chosen_pool < 0)
                        or np.any(chosen_pool >= len(pool)) or not np.isin(chosen_pool, candidate).all()
                        or not np.array_equal(np.bincount(groups[chosen_pool], minlength=len(allocation)), allocation)):
                    raise ValueError("Coreset membership/uniqueness/size/stratum check failed")
                positions = np.sort(pool_positions[chosen_pool])
                selected_positions.append(positions)
                with timed(timings, "file_output"):
                    write_json(output/f"candidate_{subset}_indices.json", index_record(pool_positions[candidate]))
                    write_json(output/f"coreset_{subset}_indices.json", index_record(positions))
                report["coresets"].append({**selection_info, "cache_key": selection_key, "cache_hit": cache is not None})
        stage = "output"
        if len(np.intersect1d(pool_positions, validation_positions)) or not np.array_equal(np.union1d(pool_positions, validation_positions), np.arange(manifest["train_rows"])):
            raise ValueError("Internal split partition invariant failed")
        report["structural_checks"] = {"partition": True, "v_less_than_p": v < p,
            "v_matches_ceiling_ratio": v == max(1, math.ceil(cfg.core_validation_ratio_to_coreset*p)),
            "coreset_membership_unique_size_and_quotas": True, "original_mapping_available": source_indices is not None,
            "original_csv_cells_preserved": True}
        report["overlap"] = [{"pair": [i, j], "shared_rows": int(len(np.intersect1d(a, b))),
                              "fraction_of_p": len(np.intersect1d(a, b))/p,
                              "jaccard": len(np.intersect1d(a, b))/len(np.union1d(a, b))}
                             for i, a in enumerate(selected_positions) for j, b in enumerate(selected_positions) if i < j]
        with timed(timings, "file_output"):
            export_rows(train_path, [(output/f"coreset_{i}.csv", rows) for i, rows in enumerate(selected_positions)])
        manifest["status"] = report["status"] = "success"
    except Exception as error:
        manifest.update(status="failed", failure_stage=stage, error=f"{type(error).__name__}: {error}")
        report.update(status="failed", failure_stage=stage, error=manifest["error"])
        raise
    finally:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["memory"] = {"peak_rss_bytes": int(rss if platform.system() == "Darwin" else rss*1024),
                            "measurement": "resource.getrusage(RUSAGE_SELF).ru_maxrss; process lifetime high-water RSS, includes earlier batch tasks; not per-task allocation"}
        report["timing_scope"] = "total wall time through data/cache output and checks, excluding final manifest/report serialization; cache hits have zero fit/optimization time"
        timings["total_wall"] = time.perf_counter()-started
        write_json(output/"manifest.json", manifest)
        write_json(output/"report.json", report)
    return report
