import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import litellm
from litellm import completion
import time
import config

litellm.suppress_debug_info = True

os.environ["OPENROUTER_API_KEY"] = config.OPENROUTER_API_KEY or ""
os.environ["OPENROUTER_API_BASE"] = config.OPENROUTER_API_BASE


def summarize_chunk(
    text: str, model: str = None, max_retries: int = 10, base_delay: float = 1.0
) -> str:
    if not config.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    model_name = model or config.GENERATION_MODEL

    messages = [
        {
            "role": "system",
            "content": "Summarize the following text very very concisely, focusing on the most important information. skip any unimportant details and unknown characters. The summary should be short and to the point.",
        },
        {"role": "user", "content": f"Text:\n{text}\n\nSummary:"},
    ]

    retry_count = 0
    while retry_count < max_retries:
        try:
            response = completion(model=model_name, messages=messages)
            return response.choices[0].message.content

        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                delay = base_delay * (2**retry_count)
                print(
                    f"  Rate limited. Waiting {delay:.1f}s before retry ({retry_count+1}/{max_retries})..."
                )
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
