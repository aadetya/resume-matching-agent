"""Local source indexing and contextual candidate retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk, chunk_text
from .contracts import Candidate, Evidence, Requirements, RetrievalBatch
from .corpus import MANIFEST_PATH, load_catalog, source_paths
from .metadata import (
    AS_OF_DATE,
    EXTRACTION_VERSION,
    MetadataExtractor,
    SkillExtractor,
    redact_identity,
)
from .neural_ranking import RERANKER_REVISION, load_ranker
from .policy import ENGINE_VERSION, fingerprint, load_policy, requirements_fingerprint
from .roles import ROLE_LABELS

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
INDEX_VERSION = "candidate-spans-v3"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def extraction_code_fingerprint() -> str:
    return fingerprint(
        {
            name: _sha(Path(__file__).with_name(name))
            for name in ("metadata.py", "roles.py", "chunking.py", "contracts.py")
        }
    )


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w#+.]+", text.casefold())


def tied_ranks(scores: dict[str, float]) -> dict[str, int]:
    """Competition ranks: identical evidence receives identical points."""
    output, previous, rank = {}, None, 0
    for position, (cid, score) in enumerate(sorted(scores.items(), key=lambda item: -item[1]), 1):
        if previous is None or score != previous:
            rank = position
        output[cid], previous = rank, score
    return output


@lru_cache(maxsize=1)
def _embedding():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cpu", trust_remote_code=False
    )


@lru_cache(maxsize=1)
def _reranker():
    return load_ranker()


def build_index(
    root: Path,
    *,
    ner_backend: str = "spacy",
    semantic: bool = True,
    window: int = 100,
    overlap: int = 25,
    index_dir: Path | None = None,
    strict: bool = True,
) -> dict:
    """Read only resume source files; evaluation labels never enter this function."""
    from .file_tools import read_file

    root = Path(root).resolve()
    destination = Path(index_dir) if index_dir else root / "artifacts/index"
    destination.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(root)
    snapshot_date = catalog.snapshot_date if catalog else AS_OF_DATE
    extractor = MetadataExtractor(root, backend=ner_backend, snapshot_date=snapshot_date)
    source_entries = {entry["path"]: entry for entry in catalog.files} if catalog else {}
    paths = [root / entry["path"] for entry in catalog.files] if catalog else source_paths(root)
    candidates, chunks, documents, errors, source_files, observed_files = [], [], {}, [], [], []
    for path in paths:
        observed_files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha(path) if path.is_file() and not path.is_symlink() else None,
            }
        )
        if path.is_symlink() or not path.resolve().is_relative_to(root / "data/resumes"):
            errors.append(
                {
                    "path": path.name,
                    "error": {"code": "path_escape", "message": "Resume symlinks are not indexed."},
                }
            )
            continue
        result = read_file(str(path))
        relative = path.relative_to(root).as_posix()
        if not result["ok"] or not result.get("content", "").strip():
            errors.append(
                {
                    "path": relative,
                    "error": result.get(
                        "error",
                        {
                            "code": "ocr_required",
                            "message": "No extractable text; OCR or a text document is needed.",
                        },
                    ),
                }
            )
            continue
        text = result["content"]
        identity = source_entries.get(relative, {})
        cid = identity.get("candidate_id", path.stem)
        if cid in documents:
            errors.append(
                {
                    "path": relative,
                    "error": {
                        "code": "duplicate_candidate_id",
                        "message": "Candidate IDs must have one source document.",
                    },
                }
            )
            continue
        sha = result["metadata"]["sha256"]
        if catalog and sha != identity["sha256"]:
            raise ValueError(f"Corpus source changed during ingestion: {relative}")
        candidate = extractor.extract(
            cid,
            text,
            relative,
            sha,
            source_id=identity.get("source_id"),
            display_name=identity.get("display_name"),
        )
        candidate.warnings.extend(result["warnings"])
        candidates.append(candidate)
        chunks.extend(chunk_text(text, cid, window=window, overlap=overlap))
        documents[cid] = {
            "source_id": candidate.source_id,
            "text": text,
            "source_path": relative,
            "sha256": sha,
            "segments": result["segments"],
            "extraction": result["metadata"],
        }
        # Runtime artifacts contain portable paths only.
        documents[cid]["extraction"]["path"] = relative
        source_files.append(
            {"candidate_id": cid, "source_id": candidate.source_id, "path": relative, "sha256": sha}
        )
    _write(destination / "ingestion_report.json", {"accepted": len(candidates), "errors": errors})
    if errors and strict:
        raise ValueError(
            f"{len(errors)} source document(s) failed ingestion. See {destination / 'ingestion_report.json'}"
        )
    if not candidates:
        raise ValueError("The resume corpus is empty. Run scripts/generate_dataset.py first.")
    if catalog and load_catalog(root) != catalog:
        raise ValueError("Corpus identity or snapshot changed during ingestion; start a new build.")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if semantic:
        model = _embedding()
        values = [
            redact_identity(chunk.text, candidate_by_id[chunk.candidate_id]) for chunk in chunks
        ]
        oversized = [
            chunk.chunk_id
            for chunk, value in zip(chunks, values, strict=True)
            if len(model.tokenizer.encode(value, add_special_tokens=True)) > model.max_seq_length
        ]
        if oversized:
            raise ValueError(
                f"Embedding token budget exceeded for {oversized[:3]}; rebuild with a smaller window."
            )
        vectors = model.encode(
            values,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding model returned non-finite values")
        np.save(
            destination / "vectors.npy", np.asarray(vectors, dtype=np.float32), allow_pickle=False
        )
    _write(destination / "candidates.json", [candidate.model_dump() for candidate in candidates])
    _write(destination / "chunks.json", [chunk.model_dump() for chunk in chunks])
    _write(destination / "documents.json", documents)
    manifest = {
        "version": INDEX_VERSION,
        "extraction_version": EXTRACTION_VERSION,
        "extraction_code_sha256": extraction_code_fingerprint(),
        "snapshot_date": snapshot_date.isoformat(),
        "corpus_manifest": catalog.provenance if catalog else None,
        "ner_backend": ner_backend,
        "ner_model": "en_core_web_sm==3.8.0" if ner_backend == "spacy" else None,
        "embedding_model": EMBEDDING_MODEL if semantic else None,
        "embedding_revision": EMBEDDING_REVISION if semantic else None,
        "candidate_count": len(candidates),
        "chunk_count": len(chunks),
        "window": window,
        "overlap": overlap,
        "taxonomy_sha256": _sha(root / "config/skills.yaml"),
        "files": source_files,
        "observed_files": observed_files,
        "artifact_hashes": {
            name: _sha(destination / name)
            for name in ["candidates.json", "chunks.json", "documents.json"]
            + (["vectors.npy"] if semantic else [])
        },
    }
    # Manifest commits last, so an interrupted build is detected on next load.
    _write(destination / "manifest.json", manifest)
    return manifest


class ResumeIndex:
    def __init__(
        self, root: str | Path, backend: str = "semantic", *, index_dir: str | Path | None = None
    ):
        if backend not in {"semantic", "dense", "bm25"}:
            raise ValueError("backend must be semantic, dense, or bm25")
        self.root = Path(root).resolve()
        self.backend = backend
        self.index_dir = Path(index_dir) if index_dir else self.root / "artifacts/index"
        if not (self.index_dir / "manifest.json").exists():
            raise RuntimeError("The resume index is missing. Run python scripts/build_index.py.")
        self.manifest = _json(self.index_dir / "manifest.json")
        if (
            self.manifest["version"] != INDEX_VERSION
            or self.manifest["extraction_version"] != EXTRACTION_VERSION
            or self.manifest.get("extraction_code_sha256") != extraction_code_fingerprint()
        ):
            raise RuntimeError("Index version changed; rebuild the resume index.")
        if self.manifest["taxonomy_sha256"] != _sha(self.root / "config/skills.yaml"):
            raise RuntimeError("Skill taxonomy changed; rebuild the resume index.")
        for name, sha in self.manifest["artifact_hashes"].items():
            if _sha(self.index_dir / name) != sha:
                raise RuntimeError(f"Index artifact changed: {name}. Rebuild the resume index.")
        for source in self.manifest["files"]:
            path = self.root / source["path"]
            if not path.exists() or path.is_symlink() or _sha(path) != source["sha256"]:
                raise RuntimeError(
                    f"Resume source changed: {source['path']}. Rebuild the resume index."
                )
        self._check_corpus_manifest()
        actual_sources = self._source_membership()
        if actual_sources != {
            source["path"] for source in self.manifest.get("observed_files", self.manifest["files"])
        }:
            raise RuntimeError("Resume source set changed; rebuild the resume index.")
        self.candidates = {
            row["candidate_id"]: Candidate.model_validate(row)
            for row in _json(self.index_dir / "candidates.json")
        }
        self.chunks = [Chunk.model_validate(row) for row in _json(self.index_dir / "chunks.json")]
        self.documents = _json(self.index_dir / "documents.json")
        self.skills = SkillExtractor(self.root)
        self.embedding_texts = [
            redact_identity(chunk.text, self.candidates[chunk.candidate_id])
            for chunk in self.chunks
        ]
        self.lexical = BM25Okapi([_tokens(text) for text in self.embedding_texts])
        self.vectors = None
        if self.backend != "bm25":
            if (
                self.manifest["embedding_model"] != EMBEDDING_MODEL
                or self.manifest["embedding_revision"] != EMBEDDING_REVISION
            ):
                raise RuntimeError("Semantic model/index mismatch. Rebuild without --lexical-only.")
            self.vectors = np.load(self.index_dir / "vectors.npy", allow_pickle=False)
            if self.vectors.shape[0] != len(self.chunks) or not np.isfinite(self.vectors).all():
                raise RuntimeError("Stored embedding vectors are invalid; rebuild the index.")
        self.policy = load_policy(self.root)
        self.engine_fingerprint = fingerprint(
            {
                "engine": ENGINE_VERSION,
                "manifest": self.manifest,
                "backend": self.backend,
                "policy": self.policy.model_dump(),
                "reranker_revision": RERANKER_REVISION,
                "decision_code": {
                    name: _sha(Path(__file__).with_name(name))
                    for name in (
                        "retrieval.py",
                        "policy.py",
                        "review_integrity.py",
                        "candidate_retrieval.py",
                        "neural_ranking.py",
                        "criteria.py",
                        "inventory.py",
                        "planner.py",
                    )
                },
            }
        )

    def _check_sources(self, ids: list[str] | None = None) -> None:
        """Do not answer a new turn from a source snapshot that silently changed."""
        if load_policy(self.root) != self.policy:
            raise RuntimeError("Screening policy changed; reload the index and start a new review.")
        self._check_corpus_manifest()
        selected = set(ids) if ids is not None else None
        for source in self.manifest["files"]:
            if selected is not None and source["candidate_id"] not in selected:
                continue
            path = self.root / source["path"]
            if not path.exists() or path.is_symlink() or _sha(path) != source["sha256"]:
                raise RuntimeError(
                    f"Resume source changed: {source['path']}. Rebuild the resume index."
                )
        if selected is None:
            actual = self._source_membership()
            if actual != {
                source["path"]
                for source in self.manifest.get("observed_files", self.manifest["files"])
            }:
                raise RuntimeError("Resume source set changed; rebuild the resume index.")
            for source in self.manifest.get("observed_files", []):
                path = self.root / source["path"]
                if source["sha256"] is not None and (
                    not path.is_file() or path.is_symlink() or _sha(path) != source["sha256"]
                ):
                    raise RuntimeError(
                        f"Observed resume source changed: {source['path']}. Rebuild the resume index."
                    )

    def _source_membership(self) -> set[str]:
        return {
            path.relative_to(self.root).as_posix()
            for path in source_paths(
                self.root, recursive=bool(self.manifest.get("corpus_manifest"))
            )
        }

    def _check_corpus_manifest(self) -> None:
        expected = self.manifest.get("corpus_manifest")
        path = self.root / MANIFEST_PATH
        if expected:
            if not path.is_file() or path.is_symlink() or _sha(path) != expected["sha256"]:
                raise RuntimeError(
                    "Corpus identity or snapshot manifest changed; rebuild the resume index."
                )
            if self._source_membership() != {
                source["path"]
                for source in self.manifest.get("observed_files", self.manifest["files"])
            }:
                raise RuntimeError("Resume source set changed; rebuild the resume index.")
        elif path.exists():
            # Legacy manifests are descriptive. Introducing an identity-bearing
            # allowlist must force a rebuild instead of serving filename identities.
            current = _json(path)
            if not isinstance(current, dict) or current.get("schema_version", 1) != 1:
                raise RuntimeError("Corpus identity manifest changed; rebuild the resume index.")

    def _query(self, requirements: Requirements) -> str:
        # Conversation/source text is historical evidence, not an executable
        # query. Removed criteria must stop influencing both retrieval arms.
        parts = ["Find resumes supporting the current requirements."]
        if requirements.semantic_brief.strip():
            parts.append("Current requirements: " + requirements.semantic_brief.strip())
        if requirements.role:
            parts.append(f"Role: {ROLE_LABELS.get(requirements.role, requirements.role)}.")
        for group in requirements.must_have:
            parts.append(
                "Required: " + " or ".join(self.skills.display(skill) for skill in group) + "."
            )
        if requirements.nice_to_have:
            parts.append(
                "Preferred: "
                + ", ".join(self.skills.display(skill) for skill in requirements.nice_to_have)
                + "."
            )
        if requirements.minimum_years:
            parts.append(f"At least {requirements.minimum_years:g} years of total employment.")
        for skill, years in requirements.skill_years.items():
            parts.append(
                f"At least {years:g} years using {self.skills.display(skill)} in dated roles."
            )
        return " ".join(parts)

    def _validate_requirements(self, requirements: Requirements) -> Requirements:
        from .planner import SkillVocabulary, validate_requirements

        return validate_requirements(requirements, SkillVocabulary(self.root))

    def _evidence(self, cid: str, chunk_scores: np.ndarray, limit: int = 3) -> list[Evidence]:
        indices = [index for index, chunk in enumerate(self.chunks) if chunk.candidate_id == cid]
        indices.sort(key=lambda index: (-float(chunk_scores[index]), self.chunks[index].start))
        chosen = []
        for index in indices:
            chunk = self.chunks[index]
            # Drop near-duplicate windows while retaining exact quotes.
            if any(
                max(0, min(chunk.end, item.end) - max(chunk.start, item.start))
                / max(1, min(chunk.end - chunk.start, item.end - item.start))
                > 0.75
                for item in chosen
            ):
                continue
            document = self.documents[cid]
            chosen.append(
                Evidence(
                    candidate_id=cid,
                    source_id=self.candidates[cid].source_id,
                    source_path=document["source_path"],
                    start=chunk.start,
                    end=chunk.end,
                    quote=chunk.text,
                    document_sha256=document["sha256"],
                    kind="retrieval",
                )
            )
            if len(chosen) >= limit:
                break
        # Add the dated evidence that supports tenure claims if no retrieval window contains it.
        for item in self.candidates[cid].evidence:
            if item.kind == "dated_employment" and not any(
                row.start <= item.start and row.end >= item.end for row in chosen
            ):
                chosen.append(item)
        return chosen

    def expand_retrieval(self, batch, requirements, *, progress=None):
        from .candidate_retrieval import expand_retrieval

        requirements = self._validate_requirements(requirements)
        return expand_retrieval(
            self, RetrievalBatch.model_validate(batch), requirements, progress=progress
        )

    def rank(self, batch, requirements, *, top_k=10):
        from .candidate_retrieval import rank_for_review

        requirements = self._validate_requirements(requirements)
        result = rank_for_review(
            self,
            RetrievalBatch.model_validate(batch),
            requirements,
            top_k=top_k,
        )
        return result

    def retrieve(
        self, requirements: Requirements, *, query_text: str | None = None
    ) -> RetrievalBatch:
        requirements = self._validate_requirements(requirements)
        started = time.perf_counter()
        self._check_sources()
        if query_text is not None and not query_text.strip():
            raise ValueError("An explicit retrieval query cannot be empty.")
        query = query_text.strip() if query_text is not None else self._query(requirements)
        lexical = np.asarray(self.lexical.get_scores(_tokens(query)))
        semantic = np.zeros(len(self.chunks), dtype=np.float32)
        if self.vectors is not None:
            model = _embedding()
            if (
                len(model.tokenizer.encode(QUERY_INSTRUCTION + query, add_special_tokens=True))
                > model.max_seq_length
            ):
                raise ValueError(
                    "The retrieval query exceeds the embedding token budget; shorten the requirements without dropping mandatory criteria."
                )
            vector = model.encode(
                [QUERY_INSTRUCTION + query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            semantic = self.vectors @ vector
        per_candidate = {}
        for cid in self.candidates:
            positions = [
                index for index, chunk in enumerate(self.chunks) if chunk.candidate_id == cid
            ]
            per_candidate[cid] = (
                max(float(semantic[index]) for index in positions),
                max(float(lexical[index]) for index in positions),
            )
        selected = (
            semantic
            if self.backend == "dense"
            else lexical
            if self.backend == "bm25"
            else semantic + lexical / max(float(lexical.max()), 1)
        )
        return RetrievalBatch(
            requirements_fingerprint=requirements_fingerprint(requirements),
            engine_fingerprint=self.engine_fingerprint,
            candidate_scores=per_candidate,
            evidence_scores=selected.tolist(),
            corpus_size=len(self.candidates),
            retrieved_count=len(self.candidates),
            timings={"retrieval_seconds": round(time.perf_counter() - started, 4)},
            backend=self.backend,
        )
