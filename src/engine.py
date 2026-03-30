import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.generator import BaseGenerator, LMStudioGenerator, OpenAIGenerator, SYSTEM_PROMPT
from src.utils import chunk_text
from src.checker import check_answer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ceps.engine")


class CEPSEngine:
    def __init__(self, patch_path: str, model_name: Optional[str] = None):
        """
        Initializes the CEPS Engine by loading a patch and an embedding model.
        """
        self.patch_path = Path(patch_path)
        if not self.patch_path.exists():
            raise FileNotFoundError(f"Patch not found at {patch_path}")

        with open(self.patch_path, "r", encoding="utf-8") as f:
            self.patch = json.load(f)

        self.ceps_version = self.patch["metadata"]["ceps_version"]
        self.model_name = model_name or self.patch["metadata"]["model_name"]
        self.alpha = self.patch["config"]["alpha"]
        self.anchor = check_answer(self.patch["anchor"]["completion"])

        logger.info(f"[*] Initializing CEPS v{self.ceps_version} Engine")
        logger.info(f"[*] Loading embedding model: {self.model_name}")
        self.embedder = SentenceTransformer(self.model_name)

        # Load centroids as numpy arrays
        self.pos_centroids = np.array(self.patch["essay"]["pos_centroids"])
        self.neg_centroids = np.array(self.patch["essay"]["neg_centroids"])

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
    patch_path: str,
    prompt: str,
    k: int = 3,
    alpha: Optional[float] = None,
    backend: str = "lmstudio",
    model: str = "local-model",
    base_url: str = "http://localhost:1234/v1",
    workers: int = 5,
    expected_type: str = "html",
):
    engine = CEPSEngine(patch_path=patch_path)
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
    print("CEPS STEERED INFERENCE RESULT")
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
