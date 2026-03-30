This is a solid, modular plan. Since we are targeting **UI Experts** first, the "Judgment" phase is crucial because you need to *see* the rendered HTML/CSS to decide if it's "Expert" or "Substandard."

Here is the architectural blueprint for **GRCS v1.0**.

---

### Project Structure
```text
./
├── data/               # Raw and labeled JSONL files
├── maps/               # Compiled .grcs files (Vector Essays + Anchors)
├── grcs/
│   ├── generator.py    # Batch generation logic (OpenAI/vLLM/Local)
│   ├── judge_ui.py     # Simple local web-server to view and P/N label UI
│   ├── builder.py      # Embedding, Clustering, and Anchor selection
│   ├── engine.py       # The inference wrapper (The "Patch" runner)
│   ├── checker.py      # Content cleaning and validation
│   └── utils.py        # Shared utilities
└── main.py             # CLI entry point
```

---

### Phase 1: The Collector (Generation & Labeling)
**Goal:** Generate $k=3$ diverse UI components and save them for human judgment.

*   **Generator Module:** Needs to be an abstract class so we can swap out the backend (e.g., `OAI_Generator`, `Llama_Generator`). (Build the LM studio first.)
*   **Data Format (`samples.jsonl`):**
    ```json
    {"id": "001", "prompt": "Glassmorphism login card", "completion": "<html>...</html>", "label": null}
    ```
*   **UI Judge Script:** A lightweight Flask or FastAPI script that:
    1. Reads the `jsonl`.
    2. Renders the 3 completions side-by-side in an iframe.
    3. Provides **[P]** and **[N]** buttons.
    4. Updates the `jsonl` with labels.

---

### Phase 2: The Builder (Embedding & Clustering)
**Goal:** Turn the labeled data into a "Vector Essay" and find the "Expert Anchor."

1.  **Embedding:** Use this
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")

# Run inference with queries and documents
query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
query_embeddings = model.encode_query(query)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# (768,) (4, 768)

# Compute similarities to determine a ranking
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.3011, 0.6359, 0.4930, 0.4889]])
```

2.  **Clustering:** 
    *   Run **K-Means** on the **Positive (P)** set.
    *   The goal isn't just to group them, but to find the **Medoid** (the actual sample closest to the cluster center).
    *   **The Anchor:** The sample closest to the center of the largest/highest-quality cluster becomes the **Static Anchor** used in the prompt instructions.
3.  **Vector Essay:**
    *   Store the cluster centroids for both Pos and Neg sets. This keeps the `.grcs` file small (you only store 10–20 centroids rather than 1,000 raw embeddings).
4.  **Storage:** Save as a `.grcs` (JSON). Also need to save metadata like "default prompt" (basic role steering).

---

### Phase 3: The Engine (Inference)
**Goal:** Use the patch to generate and select the best UI.

1.  **Priming:** Construct the system instruction:
    ```text
    You are a UI Expert. The following high-quality references are the industrial choices:
    {Fixed_Anchor_From_Patch}
    ```
2.  **Batching:** Generate $k=3$ to $k=6$ completions. (configurable)
3.  **Contrastive Scoring:**
    *   Embed each of the new $k$ completions.
    *   Calculate similarity to the **Positive Centroids** and **Negative Centroids**.
    *   `Score = max(Sim_Pos) - (alpha * max(Sim_Neg))` (alpha small first, let say 0.1) (configurable but default inside grcs metadata)
4.  **Selection:** Return the highest-scoring candidate.
