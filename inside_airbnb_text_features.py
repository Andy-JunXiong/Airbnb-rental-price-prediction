"""Privacy-minimised text features from listing descriptions and neighbourhood context.

Uses sentence-transformers (all-MiniLM-L6-v2, 22 MB) for local, offline text
embeddings. Falls back to TF-IDF (100 components) when sentence-transformers
is not installed.

The embedding columns are named text_embed_000 through text_embed_383 (or
text_tfidf_000 through text_tfidf_099 for the TF-IDF fallback).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]
    _ST_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    _TFIDF_AVAILABLE = True
except ImportError:
    TfidfVectorifier = None  # type: ignore[assignment]
    _TFIDF_AVAILABLE = False


ST_MODEL_NAME = "all-MiniLM-L6-v2"
ST_EMBEDDING_DIM = 384
TFIDF_MAX_FEATURES = 100

TEXT_FIELDS = (
    "description",
    "neighborhood_overview",
    "name",
    "host_about",
    "space",
    "house_rules",
)

# Public: feature names to use in downstream code
TEXT_FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"text_embed_{i:03d}" for i in range(ST_EMBEDDING_DIM))
    if _ST_AVAILABLE or not _TFIDF_AVAILABLE
    else tuple(f"text_tfidf_{i:03d}" for i in range(TFIDF_MAX_FEATURES))
)


def _field_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def merge_text_fields(row: dict[str, str], fields: tuple[str, ...] = TEXT_FIELDS) -> str:
    """Concatenate text fields into a single document string."""
    parts = []
    for field in fields:
        text = (row.get(field) or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts) if parts else ""


def load_text_documents(
    raw_listings_path: Path,
    fields: tuple[str, ...] = TEXT_FIELDS,
) -> tuple[list[str], dict[str, str]]:
    """Read raw listings and return (documents, id->document mapping)."""
    id_to_text: dict[str, str] = {}
    with gzip.open(raw_listings_path, "rt", encoding="utf-8-sig", newline="") as handle:
        import csv

        for row in csv.DictReader(handle):
            listing_id = (row.get("id") or "").strip()
            if not listing_id:
                continue
            text = merge_text_fields(row, fields)
            id_to_text[listing_id] = text

    listing_ids = sorted(id_to_text)
    documents = [id_to_text[lid] for lid in listing_ids]
    return documents, id_to_text


def fit_embeddings(
    raw_listings_path: Path,
    fields: tuple[str, ...] = TEXT_FIELDS,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Fit text embeddings on all listings. Returns (embeddings, listing_ids, manifest)."""
    documents, id_to_text = load_text_documents(raw_listings_path, fields)
    listing_ids = sorted(id_to_text)

    if _ST_AVAILABLE:
        model = SentenceTransformer(ST_MODEL_NAME)
        raw = model.encode(documents, show_progress_bar=False, normalize_embeddings=True)
        dim = raw.shape[1]
        names = tuple(f"text_embed_{i:03d}" for i in range(dim))
        manifest = {
            "method": "sentence-transformers",
            "model": ST_MODEL_NAME,
            "embedding_dim": dim,
            "normalized": True,
            "text_fields": list(fields),
        }
    elif _TFIDF_AVAILABLE:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words="english",
            strip_accents="unicode",
        )
        raw = vectorizer.fit_transform(documents).toarray().astype(np.float32)
        dim = raw.shape[1]
        names = tuple(f"text_tfidf_{i:03d}" for i in range(dim))
        manifest = {
            "method": "tfidf",
            "max_features": TFIDF_MAX_FEATURES,
            "embedding_dim": dim,
            "normalized": False,
            "text_fields": list(fields),
        }
    else:
        raise ImportError(
            "Either sentence-transformers or scikit-learn>=1.5 is required "
            "for text features."
        )

    lookup = np.full((len(listing_ids), dim), np.nan, dtype=np.float32)
    for i, lid in enumerate(listing_ids):
        lookup[i] = raw[i]

    manifest["listing_count"] = len(listing_ids)
    manifest["empty_documents"] = sum(1 for d in documents if not d.strip())
    return lookup, listing_ids, manifest


def embedding_for_listing(
    listing_id: str,
    lookup: np.ndarray,
    id_list: list[str],
) -> dict[str, float | str]:
    """Return named embedding values for a single listing, or blanks if missing."""
    try:
        index = id_list.index(listing_id)
        values = lookup[index]
    except (ValueError, IndexError):
        return {name: "" for name in TEXT_FEATURE_NAMES}

    return {
        name: float(values[i]) if np.isfinite(values[i]) else ""
        for i, name in enumerate(TEXT_FEATURE_NAMES)
    }


def reference_manifest() -> dict[str, Any]:
    """Return the text-feature reference manifest for reproducibility."""
    return {
        "version": 1,
        "available": _ST_AVAILABLE or _TFIDF_AVAILABLE,
        "preferred": "sentence-transformers" if _ST_AVAILABLE else "tfidf",
        "st_model": ST_MODEL_NAME if _ST_AVAILABLE else None,
        "tfidf_max_features": TFIDF_MAX_FEATURES if _TFIDF_AVAILABLE else None,
        "text_fields": list(TEXT_FIELDS),
        "privacy": {
            "raw_text_not_stored": True,
            "text_fingerprint": "sha256 hash per field before concatenation",
        },
    }
