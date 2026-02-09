"""
Clustering Service using DBSCAN algorithm.

This service:
1. Takes face embeddings as input
2. Groups similar embeddings into clusters (each cluster = one person)
3. Uses DBSCAN (Density-Based Spatial Clustering)
4. Stores clusters in database

Why DBSCAN?
- Doesn't require knowing number of clusters beforehand
- Can handle varying cluster sizes
- Identifies outliers (noise points)
- Works well with cosine distance in embedding space
"""
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)


class ClusteringService:
    """
    Clusters face embeddings to identify unique people.
    
    How DBSCAN works:
    - eps: Maximum distance between two samples to be in same neighborhood
    - min_samples: Minimum samples in neighborhood to form dense region
    - Distance metric: cosine distance (1 - cosine_similarity)
    
    For face recognition:
    - If similarity_threshold = 0.85, then eps = 1 - 0.85 = 0.15
    - min_samples = 1 (a single face can represent a person)
    """
    
    def __init__(self, similarity_threshold: float = 0.85, min_samples: int = 1):
        """
        Initialize clustering service.
        
        Args:
            similarity_threshold: Minimum cosine similarity for same cluster (0-1)
            min_samples: Minimum samples to form core point in DBSCAN
        """
        self.similarity_threshold = similarity_threshold
        self.min_samples = min_samples
        
        # Convert similarity to distance (eps parameter)
        # cosine_distance = 1 - cosine_similarity
        self.eps = 1.0 - similarity_threshold
        
        logger.info(
            f"ClusteringService initialized: "
            f"similarity_threshold={similarity_threshold}, "
            f"eps={self.eps:.3f}, "
            f"min_samples={min_samples}"
        )
    
    def cluster_embeddings(
        self, 
        embeddings: List[np.ndarray]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Cluster embeddings into groups representing unique people.
        
        Args:
            embeddings: List of face embedding vectors
            
        Returns:
            Tuple of (labels, metrics):
            - labels: Array of cluster labels (-1 for noise/outliers)
            - metrics: Dictionary with clustering statistics
        """
        if not embeddings:
            logger.warning("No embeddings to cluster")
            return np.array([]), {}
        
        # Convert to numpy array
        X = np.array(embeddings)
        
        logger.info(f"Clustering {len(X)} embeddings")
        
        # Compute cosine distance matrix
        distance_matrix = cosine_distances(X)
        
        # Apply DBSCAN
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='precomputed'  # We provide distance matrix
        )
        
        labels = dbscan.fit_predict(distance_matrix)
        
        # Calculate metrics
        metrics = self._calculate_metrics(labels, distance_matrix)
        
        logger.info(
            f"Clustering complete: "
            f"{metrics['n_clusters']} clusters, "
            f"{metrics['n_noise']} noise points"
        )
        
        return labels, metrics
    
    def _calculate_metrics(
        self, 
        labels: np.ndarray, 
        distance_matrix: np.ndarray
    ) -> Dict:
        """
        Calculate clustering quality metrics.
        
        Args:
            labels: Cluster labels
            distance_matrix: Pairwise distance matrix
            
        Returns:
            Dictionary of metrics
        """
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        n_samples = len(labels)
        
        # Calculate cluster sizes
        cluster_sizes = {}
        for label in set(labels):
            if label != -1:
                cluster_sizes[int(label)] = int(list(labels).count(label))
        
        # Calculate average intra-cluster distance
        avg_intra_distances = {}
        for cluster_id in cluster_sizes.keys():
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) > 1:
                # Get distances within cluster
                cluster_distances = distance_matrix[cluster_indices][:, cluster_indices]
                # Average distance (excluding diagonal)
                mask = ~np.eye(len(cluster_indices), dtype=bool)
                avg_dist = cluster_distances[mask].mean()
                avg_intra_distances[cluster_id] = float(avg_dist)
            else:
                avg_intra_distances[cluster_id] = 0.0
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_samples': n_samples,
            'cluster_sizes': cluster_sizes,
            'avg_intra_distances': avg_intra_distances
        }
    
    def assign_to_cluster(
        self,
        new_embedding: np.ndarray,
        cluster_centers: List[np.ndarray],
        cluster_ids: List[int]
    ) -> Optional[int]:
        """
        Assign a new embedding to existing cluster (Stage 2 - incremental).
        
        This is for future incremental clustering when new images arrive.
        Instead of reclustering everything, we compare against cluster centers.
        
        Args:
            new_embedding: New face embedding
            cluster_centers: List of cluster center embeddings
            cluster_ids: Corresponding cluster IDs
            
        Returns:
            Cluster ID if match found, None if no match
        """
        if not cluster_centers:
            return None
        
        # Calculate similarities to all cluster centers
        similarities = []
        for center in cluster_centers:
            # Cosine similarity = 1 - cosine distance
            distance = cosine_distances([new_embedding], [center])[0][0]
            similarity = 1.0 - distance
            similarities.append(similarity)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= self.similarity_threshold:
            assigned_cluster = cluster_ids[best_idx]
            logger.debug(
                f"Assigned to cluster {assigned_cluster} "
                f"with similarity {best_similarity:.3f}"
            )
            return assigned_cluster
        else:
            logger.debug(
                f"No cluster match (best similarity: {best_similarity:.3f})"
            )
            return None
    
    def calculate_cluster_center(
        self, 
        embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """
        Calculate center (mean) of a cluster.
        
        Args:
            embeddings: List of embeddings in cluster
            
        Returns:
            Center embedding (normalized)
        """
        if not embeddings:
            raise ValueError("Cannot calculate center of empty cluster")
        
        # Average embeddings
        center = np.mean(embeddings, axis=0)
        
        # Normalize to unit length
        norm = np.linalg.norm(center)
        if norm > 0:
            center = center / norm
        
        return center
    
    def merge_clusters(
        self,
        cluster1_embeddings: List[np.ndarray],
        cluster2_embeddings: List[np.ndarray],
        merge_threshold: float = 0.92
    ) -> bool:
        """
        Determine if two clusters should be merged (Stage 3 - cluster merge).
        
        This is for background maintenance to merge clusters of the same person.
        Uses a higher threshold than initial clustering.
        
        Args:
            cluster1_embeddings: Embeddings from cluster 1
            cluster2_embeddings: Embeddings from cluster 2
            merge_threshold: Minimum similarity to merge (higher than initial)
            
        Returns:
            True if clusters should be merged, False otherwise
        """
        if not cluster1_embeddings or not cluster2_embeddings:
            return False
        
        # Compare multiple embeddings between clusters
        # Require majority of comparisons to exceed threshold
        matches = 0
        total_comparisons = 0
        
        # Sample up to 5 embeddings from each cluster
        sample_size = min(5, min(len(cluster1_embeddings), len(cluster2_embeddings)))
        
        for i in range(sample_size):
            for j in range(sample_size):
                emb1 = cluster1_embeddings[i]
                emb2 = cluster2_embeddings[j]
                
                distance = cosine_distances([emb1], [emb2])[0][0]
                similarity = 1.0 - distance
                
                if similarity >= merge_threshold:
                    matches += 1
                total_comparisons += 1
        
        # Require >50% matches
        should_merge = (matches / total_comparisons) > 0.5
        
        if should_merge:
            logger.info(
                f"Clusters should merge: {matches}/{total_comparisons} "
                f"comparisons exceeded threshold {merge_threshold}"
            )
        
        return should_merge
