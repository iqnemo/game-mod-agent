from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Optional

from ..settings import DEFAULT_RETRIEVAL_K

SOURCE_TYPE_CHOICES = ("wiki", "discord", "youtube")


def normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_mods(values: Optional[Iterable[object]]) -> tuple[str, ...]:
    if values is None:
        return ()

    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        text = normalize_text(value)
        if text is None:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return tuple(normalized)


def parse_mods_csv(value: object) -> tuple[str, ...]:
    text = normalize_text(value)
    if text is None:
        return ()
    return normalize_mods(text.split("|"))


def metadata_mods(metadata: dict) -> tuple[str, ...]:
    mods = metadata.get("mods")
    if isinstance(mods, (list, tuple, set)):
        return normalize_mods(mods)

    mods_csv = metadata.get("mods_csv")
    parsed = parse_mods_csv(mods_csv)
    if parsed:
        return parsed

    primary_mod = normalize_text(metadata.get("mod"))
    if primary_mod is None:
        return ()
    return (primary_mod,)


@dataclass(frozen=True)
class RetrievalFilters:
    game: Optional[str] = None
    mods: tuple[str, ...] = ()
    include_base_game: bool = True
    source_type: Optional[str] = None

    @classmethod
    def from_inputs(
        cls,
        *,
        game: object = None,
        mods: Optional[Iterable[object]] = None,
        include_base_game: bool = True,
        source_type: object = None,
    ) -> "RetrievalFilters":
        return cls(
            game=normalize_text(game),
            mods=normalize_mods(mods),
            include_base_game=include_base_game,
            source_type=normalize_text(source_type),
        )

    def matches_metadata(self, metadata: dict) -> bool:
        if self.source_type is not None:
            metadata_source_type = normalize_text(metadata.get("source_type"))
            if metadata_source_type is None or metadata_source_type.casefold() != self.source_type.casefold():
                return False

        if self.game is not None:
            metadata_game = normalize_text(metadata.get("game"))
            if metadata_game is None or metadata_game.casefold() != self.game.casefold():
                return False

        if not self.mods:
            return True

        selected_mods = {mod.casefold() for mod in self.mods}
        doc_mods = metadata_mods(metadata)
        if not doc_mods:
            return self.include_base_game

        return any(mod.casefold() in selected_mods for mod in doc_mods)

    def prompt_scope(self) -> Optional[str]:
        parts: list[str] = []
        if self.game:
            parts.append(f"game={self.game}")
        if self.mods:
            parts.append(f"mods={', '.join(self.mods)}")
            parts.append(f"include_base_game={'yes' if self.include_base_game else 'no'}")
        if self.source_type:
            parts.append(f"source_type={self.source_type}")
        if not parts:
            return None
        return "; ".join(parts)

    def vector_store_filter(self) -> dict[str, str] | None:
        filter_data: dict[str, str] = {}
        if self.game:
            filter_data["game"] = self.game
        if self.source_type:
            filter_data["source_type"] = self.source_type
        return filter_data or None

    def to_dict(self) -> dict:
        return {
            "game": self.game,
            "mods": list(self.mods),
            "include_base_game": self.include_base_game,
            "source_type": self.source_type,
        }


def add_retrieval_args(
    parser: argparse.ArgumentParser,
    *,
    query_arg_name: str = "query",
    query_help: str = "Question or search query",
) -> None:
    parser.add_argument(query_arg_name, help=query_help)
    parser.add_argument("--k", type=int, default=DEFAULT_RETRIEVAL_K, help="Number of chunks to return")
    parser.add_argument(
        "--source-type",
        choices=SOURCE_TYPE_CHOICES,
        default=None,
        help="Optional source filter",
    )
    parser.add_argument("--game", default=None, help="Optional game filter")
    parser.add_argument("--mod", action="append", default=[], help="Repeatable mod filter")
    parser.add_argument(
        "--exclude-base-game",
        action="store_true",
        help="When using --mod, exclude base game results.",
    )


def filters_from_args(args: argparse.Namespace) -> RetrievalFilters:
    return RetrievalFilters.from_inputs(
        game=args.game,
        mods=args.mod,
        include_base_game=not args.exclude_base_game,
        source_type=args.source_type,
    )
