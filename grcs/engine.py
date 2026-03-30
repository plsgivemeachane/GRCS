import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from grcs.generator import BaseGenerator, LMStudioGenerator, OpenAIGenerator, SYSTEM_PROMPT
from grcs.utils import chunk_text
from grcs.checker import check_answer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("grcs.engine")


class GRCSEngine:
    """
    Initializes the GRCS Engine by loading a map and an embedding model.
    """

    def __init__(self, map_path: str, model_name: Optional[str] = None):
        self.map_path = Path(map_path)
        if not self.map_path.exists():
            raise FileNotFoundError(f"Map not found at {map_path}")

        with open(self.map_path, "r", encoding="utf-8") as f:
            self.map_data = json.load(f)

        self.grcs_version = self.map_data["metadata"].get("grcs_version", "1.0")
        self.model_name = model_name or self.map_data["metadata"].get("model_name")
        
        # Robust Alpha Loading
        self.alpha = self.map_data["metadata"].get("alpha")
        if self.alpha is None:
            self.alpha = self.map_data.get("config", {}).get("alpha", 0.1)

        self.expected_type = self.map_data["metadata"].get("expected_type", "html")
        
        # Robust Anchor Loading
        anchor_data = self.map_data.get("anchor", {})
        self.anchor = check_answer(anchor_data.get("completion", ""))

        logger.info(f"[*] Initializing GRCS v{self.grcs_version} Engine")
        logger.info(f"[*] Loading embedding model: {self.model_name}")
        self.embedder = SentenceTransformer(self.model_name)

        # Load centroids as numpy arrays (Support for both root-level and "essay" key)
        if "essay" in self.map_data:
            self.pos_centroids = np.array(self.map_data["essay"]["pos_centroids"])
            self.neg_centroids = np.array(self.map_data["essay"]["neg_centroids"])
        elif "pos_centroids" in self.map_data:
            self.pos_centroids = np.array(self.map_data["pos_centroids"])
            self.neg_centroids = np.array(self.map_data["neg_centroids"])
        else:
            raise KeyError("Neither 'essay' nor 'pos_centroids' key found in GRCS map.")

    def get_steered_prompt(self) -> str:
        """
        Constructs the system prompt using the file-based SYSTEM_PROMPT and the Fixed Anchor.
        """
        steered_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"The following high-quality reference is the industrial choice, clean code."
            f"{self.anchor}\n"

        )
        return steered_prompt

    def score_completions(self, completions: List[str], alpha: Optional[float] = None, expected_type: str = "html") -> List[float]:
        """
        Contrastive Scoring with Chunking: Each chunk is treated as an isolated document (Query instruction).
        Score of a completion is the MAXIMUM score of its chunks.
        """
        if alpha is None:
            alpha = self.alpha

        logger.info(f"[*] Scoring {len(completions)} completions (alpha={alpha}) as isolated 'query' documents...")
        
        scores = []
        for i, text in enumerate(completions):
            # Safety cleaning before scoring
            text = check_answer(text, expected_type=expected_type)
            chunks = chunk_text(text, max_tokens=512)
            if not chunks:
                scores.append(0.0)
                continue
                
            # 1. Embed chunks (Inference uses no specific prompt name)
            if hasattr(self.embedder, "encode_document"):
                chunk_embeddings = self.embedder.encode_document(chunks)
            else:
                chunk_embeddings = self.embedder.encode(chunks)

            chunk_scores = []
            for emb in chunk_embeddings:
                emb = emb.reshape(1, -1)
                
                # 2. Similarity to Positives
                pos_sims = cosine_similarity(emb, self.pos_centroids)[0]
                max_pos_sim = np.max(pos_sims)

                # 3. Similarity to Negatives
                neg_sims = cosine_similarity(emb, self.neg_centroids)[0]
                max_neg_sim = np.max(neg_sims)

                # 4. Chunk Score
                chunk_score = max_pos_sim - (alpha * max_neg_sim)
                chunk_scores.append(chunk_score)

            # 5. Completion Score (Maximum among its chunks)
            final_score = float(np.max(chunk_scores))
            scores.append(final_score)
            logger.info(f"  [+] Sample {i+1} ({len(chunks)} chunks): BestChunkScore={final_score:.4f}")

        return scores

    def run_steered_inference(
        self,
        prompt: str,
        generator_backend: str = "lmstudio",
        k: int = 3,
        max_workers: int = 5,
        expected_type: str = "html",
        **generator_kwargs
    ) -> Dict[str, Any]:
        """
        Full inference loop: Generate k -> Score -> Select Best.
        """
        system_prompt = self.get_steered_prompt()
        
        # Initialize generator with steered system prompt and worker count
        if generator_backend == "lmstudio":
            generator = LMStudioGenerator(
                system_prompt=system_prompt, 
                max_workers=max_workers,
                **generator_kwargs
            )
        elif generator_backend == "openai":
            generator = OpenAIGenerator(
                system_prompt=system_prompt, 
                max_workers=max_workers,
                **generator_kwargs
            )
        else:
            raise ValueError(f"Unknown generator backend: {generator_backend}")

        logger.info(f"[*] Generating {k} completions using {generator_backend}...")
        completions = list(generator.generate(prompt, k=k, expected_type=expected_type))

        if not completions:
            logger.error("[!] No completions were generated. Check backend connectivity and logs.")
            raise RuntimeError("Backend failed to generate any completions after retries.")

        # Score completions
        scores = self.score_completions(completions, expected_type=expected_type)
        best_idx = np.argmax(scores)
        
        logger.info(f"[!] Selected best completion with score {scores[best_idx]:.4f}")

        return {
            "prompt": prompt,
            "best_completion": completions[best_idx],
            "all_scores": scores,
            "best_idx": int(best_idx),
            "samples": completions,
        }



def run_engine_inference(
    map_path,
    prompt,
    k=3,
    alpha=None,
    backend="lmstudio",
    model="local-model",
    base_url="http://localhost:1234/v1",
    workers=4,
    expected_type="html",
):
    engine = GRCSEngine(map_path=map_path)
    result = engine.run_steered_inference(
        prompt=prompt,
        generator_backend=backend,
        k=k,
        max_workers=workers,
        model=model,
        base_url=base_url,
        expected_type=expected_type
    )
    
    print("\n" + "="*50)
    print("GRCS STEERED INFERENCE RESULT")
    print("="*50)
    print(f"PROMPT: {prompt[:100]}...")
    print(f"BEST SCORE: {result['all_scores'][result['best_idx']]:.4f}")
    print(f"OUTPUT LENGTH: {len(result['best_completion'])} characters")
    print("="*50)
    
    # Save output for viewing
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"steered_result_{result['best_idx']}.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(result['best_completion'])
    
    print(f"[+] Full HTML saved to {out_file}")
    return result
