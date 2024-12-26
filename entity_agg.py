from typing import List, Dict, Set
from sqlalchemy.orm import Session
import numpy as np
from Levenshtein import distance as levenshtein_distance
from rank_bm25 import BM25Okapi
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed  # For parallel computation

from models.entity import Entity

# ----------------------------------------------------
# Helper Functions or Static Methods
# ----------------------------------------------------


def calculate_embedding_similarity_packed(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate embedding similarity (cosine) for pre-extracted vectors.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def calculate_name_similarity_packed(name1: str, name2: str) -> float:
    """
    Calculate name similarity using Levenshtein distance.
    """
    max_length = max(len(name1), len(name2))
    if max_length == 0:
        return 0.0
    return 1 - (levenshtein_distance(name1, name2) / max_length)


def calculate_description_similarity_packed(desc1: str, desc2: str) -> float:
    """
    Calculate BM25 score for two descriptions, using direct text strings.
    """
    tokens1 = desc1.lower().split()
    tokens2 = desc2.lower().split()
    bm25 = BM25Okapi([tokens1, tokens2])
    scores = bm25.get_scores(tokens1)
    return float(scores[1])  # Score of desc2 vs desc1


def combined_similarity_packed(
    entity_i_data,
    entity_j_data,
    embedding_weight=0.5,
    name_weight=0.3,
    desc_weight=0.2,
) -> float:
    """
    Combine embedding, name, and description similarities with weighting.
    entity_i_data / entity_j_data is a tuple or dict containing:
      - embedding
      - name
      - description
    """
    embedding_sim = calculate_embedding_similarity_packed(
        entity_i_data["embedding"], entity_j_data["embedding"]
    )
    name_sim = calculate_name_similarity_packed(
        entity_i_data["name"], entity_j_data["name"]
    )
    desc_sim = calculate_description_similarity_packed(
        entity_i_data["description"], entity_j_data["description"]
    )
    return (
        (embedding_weight * embedding_sim)
        + (name_weight * name_sim)
        + (desc_weight * desc_sim)
    )


def compute_similarity_row(
    packed_entities: List[dict],
    i: int,
    embedding_weight: float,
    name_weight: float,
    desc_weight: float,
) -> np.ndarray:
    """
    Function for parallel calls. Computes the similarity of row i to all entities after i.
    Returns a 1D array of size len(packed_entities).
    """
    num_entities = len(packed_entities)
    row_sim = np.zeros(num_entities)
    for j in range(i + 1, num_entities):
        sim = combined_similarity_packed(
            packed_entities[i],
            packed_entities[j],
            embedding_weight,
            name_weight,
            desc_weight,
        )
        row_sim[j] = sim
    return row_sim

def angle_distance_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a cosine similarity matrix to an angle distance matrix (scaled into [0,1]):
    d = arccos(sim) / π

    - Cosine similarity ranges in [-1,1].
    - We use arccos(sim) to transform the similarity into an angle in [0, π].
    - Dividing by π produces a normalized distance in [0,1].
    - This preserves the relative order of similarity, and can handle the case where the
      cosine similarity is negative (i.e., vectors with an angle > 90°).
    """
    # Ensure values lie within [-1,1] to avoid potential domain errors in arccos
    sim_clipped = np.clip(similarity_matrix, -1, 1)
    # arccos(sim) yields angles in [0, π]
    angles = np.arccos(sim_clipped)
    # Normalize by π to obtain distances in [0,1]
    dist_mat = angles / np.pi
    return dist_mat

class EntityAggregator:
    def __init__(self, db_session: Session):
        self.db_session = db_session

    def get_entities(self, offset: int = 0, limit: int = 10000) -> List[Entity]:
        """
        Retrieve a list of Entity objects from the database using the specified
        offset and limit.

        :param offset: The starting index from which to begin retrieving records.
        :param limit: The maximum number of records to retrieve.
        :return: A list of Entity objects.
        """
        return self.db_session.query(Entity).offset(offset).limit(limit).all()

    def cluster_entities(
        self,
        entities,
        similarity_threshold: float = 0.8,
        embedding_weight: float = 0.5,
        name_weight: float = 0.3,
        desc_weight: float = 0.2,
        min_samples: int = 2,
        chunk_size: int = 50,
    ) -> List[Set]:
        """
        1. Convert each Entity into a picklable dict (packed_entities).
        2. Use joblib to parallelize similarity computations.
        3. Convert similarity into distance; cluster with DBSCAN.
        4. Rebuild clusters returning sets of the original entity objects.
        """
        num_entities = len(entities)

        # ------------------------------------------------
        # Build picklable entity data
        # ------------------------------------------------
        # We'll only store the needed fields.
        # description_vec is assumed to be a list or np array. Make sure it's a real np array.
        packed_entities = []
        for e in entities:
            # Convert to np.array if not already
            embed_vec = (
                np.array(e.description_vec)
                if e.description_vec is not None
                else np.zeros(1536)
            )
            packed_entities.append(
                {
                    "id": e.id,
                    "name": e.name,
                    "description": e.description,
                    "embedding": embed_vec,
                }
            )

        # ------------------------------------------------
        # Prepare a similarity matrix
        # ------------------------------------------------
        similarity_matrix = np.zeros((num_entities, num_entities))

        # We'll gather row indices for iteration
        row_indices = list(range(num_entities))

        # ------------------------------------------------
        # Parallel computation of similarity matrix rows
        # ------------------------------------------------
        with tqdm(total=num_entities, desc="Computing similarity rows") as pbar:
            # Chunk-based processing
            for start in range(0, num_entities, chunk_size):
                end = min(start + chunk_size, num_entities)
                chunk_indices = row_indices[start:end]

                # Parallel compute for this chunk of rows
                row_arrays = Parallel(n_jobs=-1)(
                    delayed(compute_similarity_row)(
                        packed_entities, i, embedding_weight, name_weight, desc_weight
                    )
                    for i in chunk_indices
                )

                # Merge partial results back into the similarity matrix
                for idx, i in enumerate(chunk_indices):
                    row_sim = row_arrays[idx]
                    similarity_matrix[i, i + 1 :] = row_sim[i + 1 :]
                    # Fill symmetric entries
                    for j in range(i + 1, num_entities):
                        similarity_matrix[j, i] = row_sim[j]

                # Update the progress bar by the chunk size
                pbar.update(len(chunk_indices))

        # ------------------------------------------------
        # Convert similarity -> angle distance
        # ------------------------------------------------
        distance_matrix = angle_distance_matrix(similarity_matrix)

        # ------------------------------------------------
        # Use DBSCAN with an eps value that corresponds
        # to the angle distance for the given threshold:
        #
        #   sim >= similarity_threshold
        #   => angle_distance <= arccos(similarity_threshold) / π
        #
        # So we set eps to arccos(similarity_threshold)/π
        # ------------------------------------------------
        eps_val = float(np.arccos(similarity_threshold) / np.pi)

        with tqdm(total=1, desc="DBSCAN clustering") as pbar:
            db = DBSCAN(eps=eps_val, min_samples=min_samples, metric="precomputed")
            labels = db.fit_predict(distance_matrix)
            pbar.update(1)

        # ------------------------------------------------
        # Reconstruct clusters from labels
        # ------------------------------------------------
        clusters_dict = {}
        for label, entity_obj in zip(labels, entities):
            if label == -1:
                # Noise (optional: you may print them or handle differently)
                print(f"-1 (Noise): {entity_obj}")
                continue
            clusters_dict.setdefault(label, []).append(entity_obj)

        clusters = list(clusters_dict.values())
        return clusters

    def find_concentrated_entities(self, threshold: float = 0.8):
        """
        Fetch entities, then cluster.
        """
        entities = self.get_entities()
        return self.cluster_entities(entities, similarity_threshold=threshold)
