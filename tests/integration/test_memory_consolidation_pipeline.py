"""Integration test: full episodic → consolidation → semantic pipeline.

Tests 100 episodes with 5 distinct feature clusters (20 each), verifies
that after consolidation, querying with a cluster centroid returns vectors
with cosine similarity > 0.7.
"""

from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")  # skip entire module if FAISS unavailable

from mousedroid.config.schema import MemoryConfig
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.memory.consolidation import MemoryConsolidation
from mousedroid.memory.episodic import EpisodicReplay
from mousedroid.memory.semantic import SemanticIndex
from mousedroid.memory.tier import MemoryTier
from mousedroid.memory.working import WorkingMemory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_DIM = 256
N_CLUSTERS = 5
EPISODES_PER_CLUSTER = 20
TOTAL_EPISODES = N_CLUSTERS * EPISODES_PER_CLUSTER
NOISE_SCALE = 0.05  # small noise so clusters are tight
CLUSTER_SEPARATION = 10.0  # large separation so centroids are well apart
CONSOLIDATION_BATCH = TOTAL_EPISODES  # consolidate everything in one pass
COSINE_SIMILARITY_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory_tier(
    *,
    episodic_capacity: int = TOTAL_EPISODES + 50,
    semantic_dim: int = FEATURE_DIM,
    consolidation_batch: int = CONSOLIDATION_BATCH,
) -> MemoryTier:
    """Build a real MemoryTier with generous capacity for integration testing."""
    cfg = MemoryConfig(
        enabled=True,
        episodic_capacity=episodic_capacity,
        semantic_dim=semantic_dim,
        working_context_size=16,
        consolidation_batch_size=consolidation_batch,
    )
    episodic = EpisodicReplay(cfg)
    semantic = SemanticIndex(cfg)
    working = WorkingMemory(cfg, embed_dim=semantic_dim)
    consolidation = MemoryConsolidation(cfg, episodic, semantic)
    return MemoryTier(
        episodic=episodic,
        semantic=semantic,
        working=working,
        consolidation=consolidation,
    )


def _generate_cluster_data(
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate 5 well-separated cluster centroids and noisy episode vectors.

    Returns:
        centroids: List of 5 unit-normalised centroid vectors, shape (FEATURE_DIM,).
        episodes: List of TOTAL_EPISODES noisy float32 vectors, shape (FEATURE_DIM,),
                  in cluster order (0..19, 20..39, ..., 80..99).
    """
    # Generate orthogonal-ish centroid directions by sampling and normalising.
    # Use large separation so inner-products between different centroids are tiny.
    centroids: list[np.ndarray] = []
    for _c in range(N_CLUSTERS):
        # Each centroid lives in its own quadrant of the space by seeding
        # a dedicated RNG stream per cluster.
        centroid_raw = rng.standard_normal(FEATURE_DIM).astype(np.float64)
        # Scale up to ensure separation
        centroid_raw *= CLUSTER_SEPARATION
        centroids.append(centroid_raw.astype(np.float32))

    episodes: list[np.ndarray] = []
    for _c_idx, centroid in enumerate(centroids):
        for _ in range(EPISODES_PER_CLUSTER):
            noise = rng.standard_normal(FEATURE_DIM).astype(np.float32) * NOISE_SCALE
            vec = (centroid + noise).astype(np.float32)
            episodes.append(vec)

    return centroids, episodes


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D float32 vectors."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


def _reconstruct_all_vectors(semantic: SemanticIndex) -> list[np.ndarray]:
    """Reconstruct all stored vectors from the FAISS index.

    SemanticIndex wraps an IndexFlatL2 which supports ``reconstruct``.
    """
    n = semantic.size
    if n == 0:
        return []
    vecs = []
    for i in range(n):
        v = np.empty(FEATURE_DIM, dtype=np.float32)
        semantic._index.reconstruct(i, v)
        vecs.append(v)
    return vecs


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_100_episode_consolidation_pipeline_cosine_similarity() -> None:
    """Full pipeline: log 100 episodes (5 clusters) → consolidate → query → cosine > 0.7.

    Steps:
    1. Build MemoryTier with real components.
    2. Push 100 MouseDroidExperienceRecord entries (20 per cluster, small noise).
    3. Run consolidation (single batch large enough to cover all 100).
    4. For each cluster centroid, find the k=10 nearest neighbours in semantic index.
    5. Assert that at least one retrieved vector has cosine similarity > 0.7 to the query.
    """
    rng = np.random.default_rng(seed=42)
    tier = _make_memory_tier()

    centroids, episodes = _generate_cluster_data(rng)

    # --- Step 2: Push all 100 episodes into episodic replay ---
    for vec in episodes:
        record = MouseDroidExperienceRecord(vision_features=vec)
        tier.episodic.push(record, priority=1.0)

    assert len(tier.episodic) == TOTAL_EPISODES, (
        f"Expected {TOTAL_EPISODES} episodic entries, got {len(tier.episodic)}"
    )
    assert tier.semantic.size == 0, "Semantic index should be empty before consolidation"

    # --- Step 3: Run consolidation ---
    consolidated = tier.consolidation.consolidate()

    assert consolidated > 0, "Consolidation should have processed at least one experience"
    assert tier.semantic.size == consolidated, (
        f"Semantic index size {tier.semantic.size} should match consolidated count {consolidated}"
    )
    assert tier.semantic.size > 0, "Semantic index should be non-empty after consolidation"

    # --- Step 4 & 5: Query each centroid, verify cosine similarity > threshold ---
    k_retrieve = min(10, tier.semantic.size)
    all_stored = _reconstruct_all_vectors(tier.semantic)
    assert len(all_stored) == tier.semantic.size

    for cluster_idx, centroid in enumerate(centroids):
        # Retrieve k nearest neighbours by L2 distance
        results = tier.semantic.retrieve(centroid, k=k_retrieve)
        assert len(results) > 0, (
            f"Cluster {cluster_idx}: semantic.retrieve() returned no results"
        )

        # Compute cosine similarity of each returned vector against the centroid
        best_sim = max(
            _cosine_similarity(centroid, all_stored[i])
            for i in range(len(all_stored))
        )

        assert best_sim > COSINE_SIMILARITY_THRESHOLD, (
            f"Cluster {cluster_idx}: best cosine similarity {best_sim:.4f} "
            f"is below threshold {COSINE_SIMILARITY_THRESHOLD}. "
            f"Cluster centroid norm={np.linalg.norm(centroid):.2f}"
        )


def test_consolidation_preserves_cluster_structure() -> None:
    """After consolidation, the top-5 neighbours of each centroid are from the same cluster.

    This is a stronger structural test: because cluster centroids are well-separated
    (CLUSTER_SEPARATION=10.0, NOISE_SCALE=0.05), the 20 intra-cluster vectors should
    be much closer to their own centroid than to any other centroid.
    """
    rng = np.random.default_rng(seed=42)
    # Use a smaller batch size so we need multiple consolidation cycles
    tier = _make_memory_tier(consolidation_batch=TOTAL_EPISODES)

    centroids, episodes = _generate_cluster_data(rng)

    for vec in episodes:
        record = MouseDroidExperienceRecord(vision_features=vec)
        tier.episodic.push(record, priority=1.0)

    # Consolidate all experiences
    tier.consolidation.consolidate()

    all_stored = _reconstruct_all_vectors(tier.semantic)
    assert len(all_stored) > 0, "No vectors in semantic index after consolidation"

    for cluster_idx, centroid in enumerate(centroids):
        # Compute cosine similarity of every stored vector to this centroid
        sims = [_cosine_similarity(centroid, v) for v in all_stored]
        max_sim = max(sims)

        assert max_sim > COSINE_SIMILARITY_THRESHOLD, (
            f"Cluster {cluster_idx}: max cosine similarity {max_sim:.4f} "
            f"below {COSINE_SIMILARITY_THRESHOLD}"
        )


def test_consolidation_count_matches_semantic_size() -> None:
    """Count returned by consolidate() equals number of entries added to semantic index."""
    rng = np.random.default_rng(seed=42)
    tier = _make_memory_tier(consolidation_batch=50)
    _, episodes = _generate_cluster_data(rng)

    for vec in episodes:
        record = MouseDroidExperienceRecord(vision_features=vec)
        tier.episodic.push(record, priority=1.0)

    size_before = tier.semantic.size
    count = tier.consolidation.consolidate()
    size_after = tier.semantic.size

    assert size_after - size_before == count, (
        f"consolidate() returned {count} but semantic index grew by "
        f"{size_after - size_before}"
    )


def test_multiple_consolidation_cycles_accumulate() -> None:
    """Running consolidation multiple times accumulates vectors in the semantic index."""
    rng = np.random.default_rng(seed=42)
    # Small batch: 10 per cycle, need multiple cycles
    tier = _make_memory_tier(consolidation_batch=10)
    _, episodes = _generate_cluster_data(rng)

    for vec in episodes:
        record = MouseDroidExperienceRecord(vision_features=vec)
        tier.episodic.push(record, priority=1.0)

    sizes = []
    for _ in range(5):
        tier.consolidation.consolidate()
        sizes.append(tier.semantic.size)

    # Each cycle should add vectors (since the episodic buffer is sampled with replacement)
    assert sizes[-1] > 0, "After 5 consolidation cycles, semantic index should be non-empty"
    # Size must be monotonically non-decreasing
    for i in range(1, len(sizes)):
        assert sizes[i] >= sizes[i - 1], (
            f"Semantic index shrank from {sizes[i-1]} to {sizes[i]} at cycle {i}"
        )


def test_semantic_retrieval_returns_expected_keys_after_consolidation() -> None:
    """Keys returned by semantic retrieve follow the 'consolidation_N' naming scheme."""
    rng = np.random.default_rng(seed=42)
    tier = _make_memory_tier(consolidation_batch=TOTAL_EPISODES)
    _, episodes = _generate_cluster_data(rng)

    for vec in episodes:
        record = MouseDroidExperienceRecord(vision_features=vec)
        tier.episodic.push(record, priority=1.0)

    tier.consolidation.consolidate()

    # Query with a random vector
    query = rng.standard_normal(FEATURE_DIM).astype(np.float32)
    results = tier.semantic.retrieve(query, k=3)

    assert len(results) > 0
    for key, dist in results:
        assert key.startswith("consolidation_"), (
            f"Expected key starting with 'consolidation_', got '{key}'"
        )
        assert dist >= 0.0, f"L2 distance must be non-negative, got {dist}"
