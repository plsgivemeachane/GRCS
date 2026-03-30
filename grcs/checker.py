import re
import logging

logger = logging.getLogger("grcs.checker")


def strip_think_tags(content: str) -> str:
    """
    Strips <think>...</think> blocks emitted by reasoning models via LM Studio's raw API.

    Handles:
      - Multiple <think> blocks in a single response.
      - Unclosed <think> tags (model cut off mid-thought) — removes from <think> to end.
      - Nested whitespace and newlines around the tags.
    """
    if not content:
        return ""

    # Pass 1: Remove fully closed <think>...</think> blocks (greedy within each pair).
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    # Pass 2: Handle unclosed <think> tag (model hit token limit mid-reasoning).
    # Everything from the orphaned <think> onward is reasoning garbage.
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)

    return cleaned.strip()


def strip_markdown_fences(content: str) -> str:
    """
    Strips markdown code blocks like ```html ... ``` or ``` ... ```.
    Handles multiple languages or no language specified.
    """
    if not content:
        return ""
        
    # Pattern explanation:
    # ```      : starting backticks
    # (?:\w+)? : optional language identifier (non-capturing)
    # \s*      : optional whitespace
    # (.*?)    : the actual content (non-greedy, capturing group 1)
    # \s*      : optional whitespace
    # ```      : ending backticks
    pattern = r"```(?:\w+)?\s*(.*?)\s*```"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


def validate_content(content: str, expected_type: str = "html") -> bool:
    """
    Validates if the content meets criteria for the expected type.
    """
    if not content:
        return False
    
    if expected_type == "html":
        content_lower = content.lower()
        has_html_structure = (
            "<html" in content_lower or 
            "<!doctype" in content_lower or 
            "<div" in content_lower or
            "<body" in content_lower
        )
        if not has_html_structure:
            logger.warning("Content failed HTML structural validation.")
            return False
            
    # For future: add validation for json, python, etc.
    return True


def check_answer(content: str, expected_type: str = "html") -> str:
    """
    The main entry point for checking and cleaning an 'answer' (completion).
    
    Pipeline order:
      1. Strip <think> reasoning blocks (LM Studio raw API artifact).
      2. Strip markdown code fences.
      3. Validate structural integrity.
    """
    if content is None:
        return ""

    # Step 1: Remove reasoning tags before any other processing.
    content = strip_think_tags(content)

    # Step 2: Remove markdown fences.
    cleaned = strip_markdown_fences(content)
    
    if not validate_content(cleaned, expected_type):
        logger.warning(f"Validation failed for content type: {expected_type}")
        
    return cleaned
