import numpy as np
from google import genai
from google.genai import errors
import time
import config


def get_embedding(text: str) -> np.ndarray:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment or .env file")
    
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    result = client.models.embed_content(
        model=config.GEMINI_EMBEDDING_MODEL,
        contents=text
    )
    
    return np.array(result.embeddings[0].values, dtype=np.float32)


def get_query_embedding(text: str) -> np.ndarray:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment or .env file")
    
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    result = client.models.embed_content(
        model=config.GEMINI_EMBEDDING_MODEL,
        contents=text
    )
    
    return np.array(result.embeddings[0].values, dtype=np.float32)


def get_embeddings_batch(texts: list, max_retries: int = 10, base_delay: float = 1.0) -> np.ndarray:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment or .env file")
    
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    embeddings = []
    total = len(texts)
    
    for i, text in enumerate(texts):
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                print(f"  Embedding chunk {i+1}/{total}...", end="\r")
                result = client.models.embed_content(
                    model=config.GEMINI_EMBEDDING_MODEL,
                    contents=text
                )
                embeddings.append(result.embeddings[0].values)
                success = True
                
            except errors.ClientError as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    delay = base_delay * (2 ** retry_count)
                    print(f"  Rate limited. Waiting {delay:.1f}s before retry ({retry_count+1}/{max_retries})...")
                    time.sleep(delay)
                    retry_count += 1
                else:
                    raise e
        
        if not success:
            raise Exception(f"Failed to embed chunk {i+1} after {max_retries} retries")
    
    print(f"  Embedded {total} chunks complete!{' ' * 20}")
    return np.array(embeddings, dtype=np.float32)
