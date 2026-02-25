from google import genai
from google.genai import errors
import time
import config


def summarize_chunk(text: str, model: str = None, max_retries: int = 10, base_delay: float = 1.0) -> str:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    model_name = model or config.GEMINI_GENERATION_MODEL
    
    prompt = f"""Summarize the following text concisely, capturing the key points:

{text}

Summary:"""
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
        
        except errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                delay = base_delay * (2 ** retry_count)
                print(f"  Rate limited. Waiting {delay:.1f}s before retry ({retry_count+1}/{max_retries})...")
                time.sleep(delay)
                retry_count += 1
            else:
                raise e
    
    raise Exception(f"Failed to summarize after {max_retries} retries")


def summarize_chunks(texts: list, model: str = None) -> list:
    summaries = []
    total = len(texts)
    for i, text in enumerate(texts):
        print(f"  Summarizing chunk {i+1}/{total}...", end="\r")
        summary = summarize_chunk(text, model)
        summaries.append(summary)
    print(f"  Summarized {total} chunks complete!{' ' * 20}")
    return summaries
