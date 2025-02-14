import json
from typing import List, Set, Optional
from sqlalchemy.orm import Session
import numpy as np
from Levenshtein import distance as levenshtein_distance
from rank_bm25 import BM25Okapi
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from joblib import Parallel, delayed  # For parallel computation

from models.entity import get_entity_model
from llm_inference.base import LLMInterface
from json_utils import extract_json
from utils.token import count_tokens

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
    def __init__(
        self, db_session: Session, entity_table_name: str = "entities", dim: int = 1536
    ):
        self.db_session = db_session
        self._entity_model = get_entity_model(entity_table_name, dim)

    def get_entities(self, offset: int = 0, limit: int = 10000) -> List:
        """
        Retrieve a list of Entity objects from the database using the specified
        offset and limit.

        :param offset: The starting index from which to begin retrieving records.
        :param limit: The maximum number of records to retrieve.
        :return: A list of Entity objects.
        """
        return (
            self.db_session.query(self._entity_model)
            .order_by(self._entity_model.id)
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_entities_by_name(
        self, names: List[str], offset: int = 0, limit: int = 10000
    ) -> List:
        """
        Retrieve a list of Entity objects from the database using the specified
        offset and limit.

        :param offset: The starting index from which to begin retrieving records.
        :param limit: The maximum number of records to retrieve.
        :return: A list of Entity objects.
        """
        return (
            self.db_session.query(self._entity_model)
            .filter(self._entity_model.name.in_(names))
            .order_by(self._entity_model.id)
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_entities_by_name_groups(
        self, min_group_size: int = 10, offset: int = 0, limit: int = 10000
    ) -> List:
        """
        Retrieve entities that have duplicate names (name appears more than min_group_size times).

        :param min_group_size: Minimum number of entities that should share the same name
        :param offset: The starting index from which to begin retrieving records
        :param limit: The maximum number of records to retrieve
        :return: A list of Entity objects that belong to groups with size >= min_group_size
        """
        from sqlalchemy import func

        # First, find names that appear multiple times
        name_counts = (
            self.db_session.query(
                self._entity_model.name,
                func.count(self._entity_model.id).label("name_count"),
            )
            .group_by(self._entity_model.name)
            .having(func.count(self._entity_model.id) >= min_group_size)
            .subquery()
        )

        # Then get all entities with these names
        return (
            self.db_session.query(self._entity_model)
            .join(name_counts, self._entity_model.name == name_counts.c.name)
            .order_by(self._entity_model.name, self._entity_model.id)
            .offset(offset)
            .limit(limit)
            .all()
        )

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
                print(f"-1 (Noise): {entity_obj.name} - {entity_obj.description}")
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


def should_merge_entities(
    llm_client: LLMInterface, cluster_entities: List, **model_kwargs
) -> bool:
    """
    Determine if the given entities should be merged based on their names, descriptions, and metadata.

    :param llm_client: LLM interface for making the determination
    :param cluster_entities: List of Entity objects to evaluate
    :return: Boolean indicating whether the entities should be merged
    """
    if not cluster_entities or len(cluster_entities) < 2:
        return False

    # Prepare the cluster data in JSON
    cluster_data = []
    for e in cluster_entities:
        cluster_data.append(
            {
                "name": e.name,
                "description": e.description,
                "meta": e.meta if e.meta else {},
            }
        )

        prompt = f"""You are a knowledge expert assistant. Your task is to determine if the following entities represent the same underlying concept and should be merged.

Please analyze:
1. Names: Are they the same concept with slight variations (e.g., "PostgreSQL" vs "Postgres")?
2. Descriptions: Do they describe the same thing from different angles or with different levels of detail?
3. Metadata: Is the metadata complementary or contradictory?

Rules for determining merger:
- Entities should clearly refer to the same concept, tool, or technology
- Minor variations in names are acceptable if they clearly refer to the same thing
- Descriptions should be complementary or overlapping, not contradictory
- Metadata should not contain conflicting critical information

Here are the entities to analyze:
{json.dumps(cluster_data, indent=2)}

Please respond with a JSON object containing:
{{"should_merge": true/false, "reason": "brief explanation of your decision"}}
"""

    try:
        response = llm_client.generate(prompt=prompt, **model_kwargs)
        result = json.loads(extract_json(response))
        return result
    except Exception as e:
        print("[ERROR in should_merge_entities]:", str(e))
        return False


def merge_entities(
    llm_client: LLMInterface,
    cluster_entities: List,
    only_count_token: bool = False,
    **model_kwargs,
):
    """
    Call an LLM (ChatCompletion) to produce a merged entity for a given cluster of entities.

    :param cluster_entities: A list of Entity objects belonging to the same cluster.
    :param only_count_token: If True, only return the token count of the prompt without calling LLM
    :param model_kwargs: Additional arguments to pass to the LLM client
    :return: A single merged Entity or token count (if only_count_token=True)
    """
    if not cluster_entities:
        return None

    # Prepare the cluster data in JSON
    cluster_data = []
    for e in cluster_entities:
        cluster_data.append(
            {
                "name": e.name,
                "description": e.description,
                "meta": e.meta if e.meta else {},
            }
        )

    # Build system instruction (Prompt design)
    prompt = f"""You are a knowledge expert assistant specialized in database technologies. You are given a cluster of entities that need to be merged. Each entity contains:
- name: A short string identifier (e.g., "TiKV")
- description: A potentially extensive text describing the entity
- meta: A JSON-like object containing additional data

CORE PRINCIPLES:
1. Information Preservation
   - Minimize information loss during merging
   - Preserve unique details and perspectives from each entity
   - When encountering similar content, carefully analyze for subtle differences

2. Content Coherence
   - Maintain logical flow and readability in the merged content
   - Group related information together
   - Use clear structure to organize different aspects of information

3. Consistency Check
   - Ensure merged content remains technically accurate
   - Resolve any apparent contradictions by preserving both viewpoints
   - Maintain the original context of each piece of information

Merging Guidelines:

1. Description Merging:
   - Start with the most comprehensive description as a base
   - Incorporate unique details from other descriptions
   - Use clear transitions between different aspects
   - If information seems contradictory, preserve both versions with appropriate context

2. Metadata Handling:
   - Combine all metadata fields while preserving their original context
   - For similar fields, analyze for subtle differences before merging
   - Use arrays or structured formats to preserve multiple valid values

3. Name Selection:
   - Choose the most widely recognized or official name
   - Document significant name variations in metadata

Here is the cluster data as a JSON list:
{json.dumps(cluster_data, indent=2)}

Return a JSON object with the merged entity. Prioritize information preservation while maintaining readability. Important: Ensure the output is a valid JSON - avoid using any control characters or special characters that might break JSON parsing:

```json
{{
"name": "...",
"description": "...",
"meta": {{}}
}}
```
"""

    if only_count_token:
        model = model_kwargs.get("model", "gpt-4o")
        return count_tokens(prompt, model)

    json_str = None
    try:
        # Call OpenAI ChatCompletion
        response = llm_client.generate(prompt=prompt, **model_kwargs)
        json_str = extract_json(response)
        json_str = "".join(
            char for char in json_str if ord(char) >= 32 or char in "\r\t"
        )
        return json.loads(json_str)
    except Exception as e:
        print("[ERROR in call_llm_to_merge_entities]:", str(e), json_str or response)
        raise e


def group_mergeable_entities(
    llm_client: LLMInterface, cluster_entities: List, **model_kwargs
) -> List[List]:
    """
    Analyze a group of entities and identify subgroups that can be merged together.

    :param llm_client: LLM interface for making the determination
    :param cluster_entities: List of Entity objects to evaluate
    :return: List of lists, where each inner list contains entities that can be merged
    """
    if not cluster_entities or len(cluster_entities) < 2:
        return []

    # Prepare the cluster data in JSON
    cluster_data = []
    for e in cluster_entities:
        cluster_data.append(
            {
                "id": e.id,  # Using array index as identifier
                "name": e.name,
                "description": e.description,
                "meta": e.meta if e.meta else {},
            }
        )

    prompt = f"""You are a knowledge expert assistant. Your task is to analyze the following entities and identify which ones represent the same underlying concepts and can be merged together.

Please analyze:
1. Names: Look for same concepts with variations (e.g., "PostgreSQL" vs "Postgres")
2. Descriptions: Identify descriptions that describe the same thing from different angles
3. Metadata: Check if metadata is complementary or contradictory

Rules for grouping:
- Entities should clearly refer to the same concept, tool, or technology
- Minor variations in names are acceptable if they clearly refer to the same thing
- Descriptions should be complementary or overlapping, not contradictory
- Metadata should not contain conflicting critical information

Here are the entities to analyze:
{json.dumps(cluster_data, indent=2)}

Please respond with a JSON object containing groups of mergeable entities:
{{
    "groups": [
        {{
            "ids": [list of entity ids that can be merged],
            "reason": "brief explanation why these entities can be merged"
        }},
        // ... more groups if applicable
    ]
}}

Note: Each entity should appear in at most one group. If an entity cannot be merged with any others, exclude it from the results.
"""

    try:
        response = llm_client.generate(prompt=prompt, **model_kwargs)
        result = json.loads(extract_json(response))

        # Convert the LLM response into groups of actual entity objects
        mergeable_groups = []

        for group in result.get("groups", []):
            ids = group.get("ids", [])
            entity_group = [entity for entity in cluster_entities if entity.id in ids]

            if len(entity_group) >= 2:  # Only include groups with at least 2 entities
                mergeable_groups.append(entity_group)

        return mergeable_groups
    except Exception as e:
        print("[ERROR in group_mergeable_entities]:", str(e), response)
        return []
