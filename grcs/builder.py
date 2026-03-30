import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from grcs.utils import chunk_text
from grcs.checker import check_answer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("grcs.builder")

GRCS_VERSION = "1.0"
DEFAULT_EMBEDDING_MODEL = "google/embeddinggemma-300m"


class GRCSBuilder:
    """
    Analyzes labeled data and builds a GRCS Map (.grcs).
    """

    def __init__(self, model_name: str = "google/embedding-gemma-300m"):
        logger.info(f"[*] Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
        except Exception as e:
            logger.error(f"[!] Failed to load model {model_name}: {e}")
            raise

    def load_samples(self, input_path: str) -> List[Dict[str, Any]]:
        samples = []
        if not Path(input_path).exists():
            logger.error(f"[!] Input file {input_path} does not exist.")
            raise FileNotFoundError(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def build(
        self,
        input_path: str,
        output_path: str = "maps/v1.grcs",
        n_clusters: int = 50,
        alpha: float = 0.1,
        expected_type: str = "html",
    ):
        """
        Builds the GRCS map (the .grcs file).
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        samples = self.load_samples(str(input_path))
        
        # 1. Filter Pos/Neg samples
        pos_samples = [s for s in samples if s.get("label") == "P"]
        neg_samples = [s for s in samples if s.get("label") == "N"]

        if not pos_samples:
            logger.error("[!] No Positive samples ('P') found. Aborting.")
            sys.exit(1)
        if not neg_samples:
            logger.error("[!] No Negative samples ('N') found. Aborting.")
            sys.exit(1)

        logger.info(f"[*] Processing {len(pos_samples)} positive and {len(neg_samples)} negative samples.")

        # 2. Anchor Selection
        anchor_data = self._select_anchor(pos_samples, expected_type=expected_type)
        anchor_prompt = anchor_data.get("prompt", "")
        anchor_completion = anchor_data.get("completion", "")

        # 3. Clustering (Vector Essay)
        pos_centroids = self._get_centroids([check_answer(s["completion"], expected_type=expected_type) for s in pos_samples], n_clusters)
        neg_centroids = self._get_centroids([check_answer(s["completion"], expected_type=expected_type) for s in neg_samples], n_clusters)

        # 4. Metadata & Storage
        self.map_data = {
            "metadata": {
                "grcs_version": GRCS_VERSION,
                "model_name": self.model_name,
                "alpha": alpha,
                "expected_type": expected_type,
                "created_at": str(Path(input_path).stat().st_mtime),
            },
            "config": {
                "alpha": alpha,
                "default_prompt": anchor_prompt,
            },
        }
        
        # 7. Final Map Structure
        self.map_data["anchor"] = {
            "prompt": anchor_prompt,
            "completion": anchor_completion,
        }
        self.map_data["essay"] = {
            "pos_centroids": pos_centroids,
            "neg_centroids": neg_centroids,
        }

        # 8. Save to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.map_data, f, indent=4)
            
        logger.info(f"[*] GRCS map saved to {output_path}")

    def _get_centroids(self, texts: List[str], n_clusters: int) -> List[List[float]]:
        """
        Embeds the texts (with chunking) and returns centroids for the 'essay'.
        """
        # Chunk the texts into segments of 512 tokens
        all_chunks = []
        for text in texts:
            all_chunks.extend(chunk_text(text, max_tokens=512))
            
        n_samples = len(all_chunks)
        if n_samples == 0:
            return []
            
        logger.info(f"[*] Embedding {n_samples} chunks using 'Clustering' instruction...")
        if hasattr(self.model, "encode_document"):
            embeddings = self.model.encode_document(all_chunks, prompt_name="Clustering")
        else:
            embeddings = self.model.encode(all_chunks, prompt_name="Clustering")
        
        if n_samples <= n_clusters:
            logger.info(f"  [!] Using all {n_samples} chunks as centroids.")
            return embeddings.tolist()
        
        logger.info(f"[*] Running K-Means (k={n_clusters}) on {n_samples} chunks.")
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
        kmeans.fit(embeddings)
        return kmeans.cluster_centers_.tolist()

    def _select_anchor(self, samples: List[Dict[str, Any]], expected_type: str = "html") -> Dict[str, Any]:
        """
        Calculates medoid among positive samples using pooled chunk embeddings.
        """
        logger.info(f"[*] Selecting Static Anchor from {len(samples)} positive samples...")
        
        # 1. Embed pooled chunks for each sample to calculate medoid
        sample_vectors = []
        for sample in samples:
            text = check_answer(sample["completion"], expected_type=expected_type)
            chunks = chunk_text(text, max_tokens=512)
            if hasattr(self.model, "encode_document"):
                chunk_embs = self.model.encode_document(chunks, prompt_name="Clustering")
            else:
                chunk_embs = self.model.encode(chunks, prompt_name="Clustering")
            
            # Pool chunks (average) to get one vector per sample
            pooled_emb = np.mean(chunk_embs, axis=0)
            sample_vectors.append(pooled_emb)
        
        vectors = np.array(sample_vectors)
        
        # 2. Similarity Matrix
        sim_matrix = cosine_similarity(vectors)
        
        # 3. Find sample with highest mean similarity to all other samples
        scores = sim_matrix.mean(axis=1)
        best_idx = np.argmax(scores)
        
        anchor = samples[best_idx]
        logger.info(f"  [+] Selected anchor sample (ID: {anchor.get('id', 'unknown')})")
        return {
            "id": anchor.get("id"),
            "completion": check_answer(anchor["completion"], expected_type=expected_type)
        }


def run_builder(input_path: str, output_path: str, model_name: str, n_clusters: int, alpha: float, expected_type: str = "html"):
    builder = GRCSBuilder(model_name=model_name)
    builder.build(
        input_path=input_path,
        output_path=output_path,
        n_clusters=n_clusters,
        alpha=alpha,
        expected_type=expected_type
    )
