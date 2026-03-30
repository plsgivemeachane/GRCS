# GRCS: Group Relative Completion Selection

**Getting better answers by picking the best one from a group.**

GRCS is a training-free way to improve model outputs. Instead of relying on a single prompt and hoping for the best, GRCS generates a group of several completions at once. It then uses a comparison math to pick the one that looks most like a successful result and least like a failure.

## How it Works

### 1. Proven Anchors
Instead of writing instructions like "Act like a professional," GRCS uses **Proven Anchors**. These are actual successful answers from the GRCS files that are hard-coded into the model's context. This shows the model a clear target to follow based on real data rather than subjective descriptions.

### 2. Group Sampling
Most models only give you one answer at a time. GRCS asks for a group of 3 to 6 answers. By exploring more of what the model can do in a single pass, we increase the statistical chance that at least one of those answers is a high-quality outlier.

### 3. Relative Selection
Once we have a group of answers, we need to pick the winner. GRCS compares every answer in the group to a "Behavior Map." This map is built from two things:
*   **The Positive Map:** What a correct, high-quality completion looks like.
*   **The Negative Map:** What a bad or generic completion looks like.

The system selects the answer that is closest to the positive map and farthest from the negative map. This actively filters out "safe" or mediocre responses.

## Key Advantages

*   **No Training Needed:** You don't have to update model weights or run expensive fine-tuning. It works on any frozen model.
*   **Swappable Behavior:** You can save these Behavior Maps as small files. You can switch from a "Coding Style" to a "Creative Writing Style" instantly by just loading a different file.
*   **Data-Driven:** It uses real examples of what worked in the past to guide the selection, making it more reliable than standard prompt engineering.
*   **Better Quality Floor:** By picking the best of 6 instead of the first of 1, the "quality floor" of your application stays much higher.

## Performance and Compute

| Feature | Standard Prompting | Fine-Tuning (LoRA) | **GRCS (Ours)** |
| :--- | :--- | :--- | :--- |
| **Logic Source** | Manual Instructions | Changed Weights | **Proven Anchors** |
| **Setup Cost** | Zero | High | **Low (Labeling data)** |
| **Running Cost** | 1x (One answer) | ~1.1x | **3x – 6x (Group of answers)** |
| **Portability** | Text Snippets | Large Weight Files | **Small Map Files** |
| **Success Rate** | Variable | High | **High (Verified Top Pick)** |

## Quickstart (CLI Only)

### 1. Installation
Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Candidate Data
Generate multiple completions for labeling:
```bash
python -m grcs generate --k 3 --output data/samples.jsonl
```

### 3. Label with Judge UI
Run the local labeling tool to mark completions as Positive (P) or Negative (N):
```bash
python -m grcs judge
```

### 4. Build the Behavior Map
Analyze your labels and build a `.grcs` map file:
```bash
python -m grcs build --input data/samples.jsonl --output maps/v1.grcs
```
(Alternatively: `python -m grcs.build --data ./data/samples.jsonl --output ./maps/v1.grcs`)

### 5. Run Steered Inference
Run the engine to select the best answer using your behavior map:
```bash
python -m grcs run --map maps/v1.grcs --prompt "Build a dark theme hero section" --k 3
```

## The Compute Trade-off
GRCS is built for situations where the quality of the answer is more important than the cost of the tokens. By trading extra compute power for a group selection process, it provides a "verified" output that is much more consistent than the random nature of standard LLM calls.