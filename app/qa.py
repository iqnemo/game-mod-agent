from __future__ import annotations

import argparse
import os

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from retriever import retrieve

load_dotenv(find_dotenv())

DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_MAX_CONTEXT_CHARS = 8000


def get_client() -> OpenAI:
    api_key = os.getenv("RAG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing LLM API key. Set RAG_LLM_API_KEY (or OPENAI_API_KEY) in .env."
        )

    base_url = os.getenv("RAG_LLM_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def answer_question(question: str, k: int, source_type: str | None) -> str:
    docs = retrieve(query=question, k=k, source_type=source_type)
    if not docs:
        return "I could not find relevant context in the knowledge base yet."

    context_blocks = []
    for idx, doc in enumerate(docs, start=1):
        uri = doc.metadata.get("canonical_uri") or "unknown"
        source_type_value = doc.metadata.get("source_type") or "unknown"
        author_name = doc.metadata.get("author_name")
        source_line = f"[{idx}] source={source_type_value} uri={uri}"
        if author_name:
            source_line += f" author={author_name}"
        context_blocks.append(f"{source_line}\n{doc.page_content}")

    joined_context = "\n\n".join(context_blocks)
    max_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", str(DEFAULT_MAX_CONTEXT_CHARS)))
    trimmed_context = joined_context[:max_chars]

    system_prompt = (
        "You are a Terraria/Calamity assistant using retrieved context. "
        "Answer using only the supplied context. If context is incomplete, say that explicitly. "
        "Always cite sources as [1], [2], etc in your answer."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{trimmed_context}"
    )

    model = os.getenv("RAG_LLM_MODEL", DEFAULT_MODEL)
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple RAG QA")
    parser.add_argument("question", help="User question")
    parser.add_argument("--k", type=int, default=6, help="How many chunks to retrieve")
    parser.add_argument(
        "--source-type",
        choices=["wiki", "discord", "youtube"],
        default=None,
        help="Optional source filter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    answer = answer_question(args.question, k=args.k, source_type=args.source_type)
    print(answer)


if __name__ == "__main__":
    main()
