from __future__ import annotations

import os
import pathlib
import sys
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.ingest.storage import get_db_connection, list_available_filters
from app.rag.qa import answer_question_details
from app.rag.query import RetrievalFilters
from app.rag.retriever import retrieve
from app.settings import DEFAULT_WEB_HOST, DEFAULT_WEB_PORT, get_int_env

STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    question: str
    game: Optional[str] = None
    mods: list[str] = Field(default_factory=list)
    include_base_game: bool = True
    source_type: Optional[str] = None
    k: int = 6


class RetrieveRequest(BaseModel):
    query: str
    game: Optional[str] = None
    mods: list[str] = Field(default_factory=list)
    include_base_game: bool = True
    source_type: Optional[str] = None
    k: int = 6


app = FastAPI(title="game-mod-agent")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _build_filters(
    *,
    game: Optional[str],
    mods: list[str],
    include_base_game: bool,
    source_type: Optional[str],
) -> RetrievalFilters:
    return RetrievalFilters.from_inputs(
        game=game,
        mods=mods,
        include_base_game=include_base_game,
        source_type=source_type,
    )


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/filters")
def filters() -> dict:
    conn = get_db_connection()
    try:
        items = list_available_filters(conn)
    finally:
        conn.close()
    return {"games": items}


@app.post("/api/retrieve")
def retrieve_endpoint(request: RetrieveRequest) -> dict:
    filters = _build_filters(
        game=request.game,
        mods=request.mods,
        include_base_game=request.include_base_game,
        source_type=request.source_type,
    )
    docs = retrieve(request.query, k=request.k, filters=filters)
    return {
        "results": [
            {
                "index": index,
                "metadata": dict(doc.metadata),
                "text": doc.page_content,
            }
            for index, doc in enumerate(docs, start=1)
        ],
        "filters": filters.to_dict(),
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    filters = _build_filters(
        game=request.game,
        mods=request.mods,
        include_base_game=request.include_base_game,
        source_type=request.source_type,
    )
    return answer_question_details(request.question, k=request.k, filters=filters)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def main() -> None:
    host = (os.getenv("WEB_HOST") or DEFAULT_WEB_HOST).strip() or DEFAULT_WEB_HOST
    port = get_int_env("WEB_PORT", DEFAULT_WEB_PORT, minimum=1)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
