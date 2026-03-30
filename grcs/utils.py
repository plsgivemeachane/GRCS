import tiktoken
from typing import List

def chunk_text(text: str, max_tokens: int = 512, model: str = "gpt-4") -> List[str]:
    """
    Splits long strings into smaller chunks using tiktoken for granular embedding.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
    return chunks
