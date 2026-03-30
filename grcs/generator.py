"""
GRCS Phase 1: Generator Module

Abstract base class for batch UI generation with pluggable backends.
Currently implements LM Studio (OpenAI-compatible API).
"""

import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import requests
import lmstudio as lms
from concurrent.futures import ThreadPoolExecutor, as_completed
from grcs.checker import check_answer


# System prompt logic: Load from file or fallback to hardcoded default
def load_system_prompt(path: str = "systemprompt.md") -> str:
    default_prompt = (
        "You are an expert frontend developer specializing in modern CSS and UI design. "
        "Generate complete, self-contained HTML files with inline CSS and JavaScript. "
        "The HTML should be visually polished, production-grade, and render correctly "
        "when opened directly in a browser. Use modern CSS features (flexbox, grid, "
        "custom properties, animations). Do not include markdown fences or explanations—"
        "output ONLY raw HTML."
    )
    p = Path(path)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
    return default_prompt


SYSTEM_PROMPT = load_system_prompt()


class BaseGenerator(ABC):
    """Abstract generator: produces k completions for a given prompt."""

    @abstractmethod
    def generate(self, prompt: str, k: int = 3, expected_type: str = "html"):
        """Yield k cleaned completions for the given prompt."""
        ...


class LMStudioGenerator(BaseGenerator):
    """Backend for LM Studio's OpenAI-compatible local API."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "local-model",
        temperature: float = 0.9,
        max_tokens: int = 16384,
        timeout: int = 60 * 60 * 24, # A days
        system_prompt: str = SYSTEM_PROMPT,
        max_workers: int = 5,
        retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.max_workers = max_workers
        self.retries = retries
        self.backoff_factor = backoff_factor
        self._endpoint = f"{self.base_url}/chat/completions"

    def _make_request(self, prompt: str) -> str:
        """Helper for a single synchronous completion request with retries using LM Studio SDK."""
        last_exception = None
        for attempt in range(self.retries + 1):
            if attempt > 0:
                sleep_time = self.backoff_factor ** attempt
                time.sleep(sleep_time)
                logging.getLogger("grcs.generator").info(f"Retrying request (attempt {attempt}/{self.retries})...")
            
            try:
                # The SDK uses a different host format usually (host:port)
                # but our base_url is http://localhost:1234/v1
                # We need to extract the host:port for the SDK
                from urllib.parse import urlparse
                parsed = urlparse(self.base_url)
                host_port = parsed.netloc if parsed.netloc else self.base_url.replace("http://", "").split("/")[0]

                with lms.Client(host_port) as client:
                    # Get the model. If self.model is "local-model" or "default", 
                    # we might just want whatever is loaded.
                    # The SDK allows specifying model identifier if needed.
                    model = client.llm.model(self.model)
                    
                    # Create a prediction
                    response = model.respond(
                        {
                            "messages": [
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": prompt}
                            ]
                        },
                        config={
                            "temperature": self.temperature,
                            "maxTokens": self.max_tokens,
                        }
                    )
                    return response.content
            except Exception as e:
                last_exception = e
                logging.getLogger("grcs.generator").error(f"LM Studio SDK request failed (attempt {attempt+1}): {e}")
        
        raise last_exception or Exception("Unknown error in _make_request via SDK")

    def generate(self, prompt: str, k: int = 3, expected_type: str = "html"):
        n_workers = min(k, self.max_workers)
        logger_gen = logging.getLogger("grcs.generator")
        logger_gen.info(f"[*] Dispatching {k} parallel requests via LM Studio SDK (workers={n_workers})...")
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {executor.submit(self._make_request, prompt): i for i in range(k)}
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    content = future.result()
                    # Clean the answer using the checker
                    cleaned_content = check_answer(content, expected_type=expected_type)
                    print(f"  [+] Completion {idx+1} received ({len(cleaned_content)} chars cleaned)")
                    yield cleaned_content
                except Exception as e:
                    print(f"  [!] Completion {idx+1} failed after retries: {e}")


class OpenAIGenerator(BaseGenerator):
    """Backend for OpenAI API (used as a fallback or for cloud generation)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.9,
        max_tokens: int = 16384,
        timeout: int = 120,
        system_prompt: str = SYSTEM_PROMPT,
        max_workers: int = 5,
        retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.max_workers = max_workers
        self.retries = retries
        self.backoff_factor = backoff_factor
        self._endpoint = "https://api.openai.com/v1/chat/completions"

    def _make_request(self, prompt: str) -> str:
        """Helper for a single synchronous completion request with retries."""
        last_exception = None
        for attempt in range(self.retries + 1):
            if attempt > 0:
                sleep_time = self.backoff_factor ** attempt
                time.sleep(sleep_time)
                logging.getLogger("grcs.generator").info(f"Retrying request (attempt {attempt}/{self.retries})...")
                
            try:
                resp = requests.post(
                    self._endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_exception = e
                logging.getLogger("grcs.generator").error(f"Request failed (attempt {attempt+1}): {e}")
        
        raise last_exception or Exception("Unknown error in _make_request")

    def generate(self, prompt: str, k: int = 3, expected_type: str = "html"):
        n_workers = min(k, self.max_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {executor.submit(self._make_request, prompt): i for i in range(k)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    content = future.result()
                    cleaned_content = check_answer(content, expected_type=expected_type)
                    print(f"  [+] Completion {idx+1} received ({len(cleaned_content)} chars cleaned)")
                    yield cleaned_content
                except Exception as e:
                    print(f"  [!] Completion {idx+1} failed after retries: {e}")


def run_generation(
    prompts: list[str],
    generator: BaseGenerator,
    output_path: str = "data/samples.jsonl",
    k: int = 3,
    expected_type: str = "html",
) -> str:
    """
    Run batch generation for a list of prompts.

    Appends to existing JSONL if present; skips prompts that already have
    k completions in the file (idempotent re-runs).

    Returns the output file path.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load existing samples to support idempotent re-runs
    existing: dict[str, list[dict]] = {}
    if output.exists():
        with open(output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                pid = sample["prompt"]
                existing.setdefault(pid, []).append(sample)

    total_new = 0
    with open(output, "a", encoding="utf-8") as f:
        for prompt_text in prompts:
            current = existing.get(prompt_text, [])
            if len(current) >= k:
                print(
                    f"[SKIP] Prompt already has {len(current)} completions: {prompt_text[:60]}..."
                )
                continue

            needed = k - len(current)
            print(f"[GEN] {needed} more needed for: {prompt_text[:60]}...")
            
            # Save on the fly by iterating over the generator
            for html in generator.generate(prompt_text, k=needed, expected_type=expected_type):
                sample = {
                    "id": uuid.uuid4().hex[:8],
                    "prompt": prompt_text,
                    "completion": html,
                    "label": None,
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f.flush()  # Force OS write to prevent data loss on crash
                total_new += 1

    print(f"\n[DONE] {total_new} new samples written to {output}")
    return str(output)
