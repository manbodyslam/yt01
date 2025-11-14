#!/usr/bin/env python3
"""
Analyze face embeddings to find optimal DBSCAN parameters
"""
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from modules.face_service import FaceService
from modules.ingestor import Ingestor

def analyze_embeddings():
    """Analyze face embeddings from workspace/raw"""

    print("=" * 80)
    print("FACE CLUSTERING ANALYSIS")
    print("=" * 80)

    # Initialize services
    ingestor = Ingestor()
    face_service = FaceService()

    # Ingest images
    raw_dir = Path("workspace/raw")
    if not raw_dir.exists() or not list(raw_dir.glob("*.jpg")):
        print(f"\n❌ No images found in {raw_dir}")
        print("Please extract frames from video first!")
        return

    image_metadata_list = ingestor.ingest()
    print(f"\n✓ Loaded {len(image_metadata_list)} images")

    # Analyze faces
    face_service.analyze_all_images(image_metadata_list)
    print(f"✓ Found {len(face_service.face_db)} faces")

    if len(face_service.face_db) == 0:
        print("\n❌ No faces detected!")
        return

    # Extract embeddings
    embeddings = np.array([face['embedding'] for face in face_service.face_db])
    print(f"✓ Extracted {len(embeddings)} embeddings (shape: {embeddings.shape})")

    print("\n" + "=" * 80)
    print("STEP 1: PAIRWISE DISTANCE ANALYSIS")
    print("=" * 80)

    # Calculate pairwise distances
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(embeddings)

    # Get upper triangle (excluding diagonal)
    upper_tri_idx = np.triu_indices_from(distances, k=1)
    pairwise_dists = distances[upper_tri_idx]

    print(f"\nPairwise distances statistics:")
    print(f"  Min:  {pairwise_dists.min():.4f}")
    print(f"  Max:  {pairwise_dists.max():.4f}")
    print(f"  Mean: {pairwise_dists.mean():.4f}")
    print(f"  Std:  {pairwise_dists.std():.4f}")
    print(f"  Median: {np.median(pairwise_dists):.4f}")

    # Show percentiles
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(pairwise_dists, p):.4f}")

    print("\n" + "=" * 80)
    print("STEP 2: TEST DIFFERENT DBSCAN PARAMETERS")
    print("=" * 80)

    # Test different eps values
    eps_values = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    min_samples_values = [1, 2, 3]

    results = []

    for min_samples in min_samples_values:
        print(f"\n--- MIN_SAMPLES = {min_samples} ---")
        for eps in eps_values:
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            labels = clusterer.fit_predict(embeddings)

            # Count clusters (excluding noise = -1)
            unique_labels = set(labels)
            n_clusters = len(unique_labels - {-1})
            n_noise = list(labels).count(-1)

            # Get cluster sizes
            cluster_sizes = []
            for label in unique_labels:
                if label != -1:
                    cluster_sizes.append(sum(labels == label))

            result = {
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'cluster_sizes': cluster_sizes,
                'labels': labels
            }
            results.append(result)

            print(f"  eps={eps:.2f}: {n_clusters} clusters, {n_noise} noise, sizes={cluster_sizes}")

    print("\n" + "=" * 80)
    print("STEP 3: RECOMMENDATIONS")
    print("=" * 80)

    # Find best parameters
    # Goal: 2-4 clusters with minimal noise
    best_result = None
    best_score = -1

    for r in results:
        if r['n_clusters'] >= 2 and r['n_clusters'] <= 5:
            # Score = n_clusters - (noise_ratio * 5)
            noise_ratio = r['n_noise'] / len(embeddings)
            score = r['n_clusters'] - (noise_ratio * 5)

            if score > best_score:
                best_score = score
                best_result = r

    if best_result:
        print(f"\n✓ RECOMMENDED PARAMETERS:")
        print(f"  DBSCAN_EPS: {best_result['eps']}")
        print(f"  DBSCAN_MIN_SAMPLES: {best_result['min_samples']}")
        print(f"  Result: {best_result['n_clusters']} clusters, {best_result['n_noise']} noise")
        print(f"  Cluster sizes: {best_result['cluster_sizes']}")
    else:
        print(f"\n⚠️  No optimal parameters found!")
        print(f"  Suggestion: Use min_samples=1 with eps in range 0.4-0.6")

    # Show face details for recommended clustering
    if best_result:
        print("\n" + "=" * 80)
        print("FACE CLUSTERING DETAILS (RECOMMENDED PARAMS)")
        print("=" * 80)

        labels = best_result['labels']
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            print(f"\n--- Cluster {cluster_id} ---")
            cluster_faces = [face_service.face_db[i] for i, label in enumerate(labels) if label == cluster_id]

            for face in cluster_faces:
                num_faces = face.get('num_faces_in_image', 'unknown')
                solo = "SOLO" if num_faces == 1 else f"GROUP({num_faces})"
                eyes = face.get('eyes_open_score', 0)
                score = face.get('composite_score', 0)
                print(f"  - {face['face_id']}: {solo}, eyes={eyes:.2f}, score={score:.3f}")

    # HDBSCAN analysis
    if HDBSCAN_AVAILABLE:
        print("\n" + "=" * 80)
        print("STEP 4: HDBSCAN ANALYSIS (Auto-tuning)")
        print("=" * 80)

        for min_cluster_size in [2, 3, 4]:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(embeddings)

            unique_labels = set(labels)
            n_clusters = len(unique_labels - {-1})
            n_noise = list(labels).count(-1)

            cluster_sizes = []
            for label in unique_labels:
                if label != -1:
                    cluster_sizes.append(sum(labels == label))

            print(f"\nHDBSCAN (min_cluster_size={min_cluster_size}):")
            print(f"  {n_clusters} clusters, {n_noise} noise")
            print(f"  Cluster sizes: {cluster_sizes}")

if __name__ == "__main__":
    analyze_embeddings()
