"""Embed (DESIGN.md §3): text vectors for scene summaries and phrasings, keyed by the
embedding model. Batched; each batch is a task keyed by its texts, the model, the template
and normalization, so unchanged texts are never embedded twice."""

import math
from typing import Any

from framekit.contracts import EmbeddingRequest
from framekit.extraction.runner import CallRecord
from framekit.extraction.schemas.common import digest
from framekit.extraction.stages.context import Attempts, StageContext

EMBED_VERSION = "embed@1"


def _normalized(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else list(vector)


async def embed_texts(
    ctx: StageContext, unit_id: str, items: list[tuple[str, str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Vectors for (target_type, target_id, text) items: (`vectors` rows, run records)."""
    settings = ctx.run.plan.embedding
    text_settings = settings.text
    if text_settings is None or not items:
        return [], []
    binding = ctx.run.roles["embed_text"]
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    size = text_settings.batch_size
    for start in range(0, len(items), size):
        batch = items[start : start + size]
        texts = [text_settings.template.replace("{text}", text) for _, _, text in batch]
        inputs = {
            "version": EMBED_VERSION,
            "model": binding.identity(),
            "normalize": settings.normalize,
            "texts": digest(texts),
        }

        async def compute(
            _: str, texts: list[str] = texts, start: int = start
        ) -> tuple[Any, dict[str, Any]]:
            request = EmbeddingRequest(tuple(texts), model_id=binding.model_id)
            result = await ctx.ex.embed("embed", binding.provider.name, request)
            vectors = [list(v) for v in result.vectors]
            if settings.normalize:
                vectors = [_normalized(v) for v in vectors]
            attempts = Attempts([CallRecord.of_embedding(result)])
            output = {
                "vectors": vectors,
                "model_id": result.model_id,
                "dimension": result.dimension,
            }
            return output, attempts.fields(unit_id=unit_id, target_id=f"{unit_id}:batch{start}")

        output, record = await ctx.ex.task("embed", inputs, compute)
        records.append(record)
        for (target_type, target_id, text), vector in zip(batch, output["vectors"], strict=True):
            rows.append(
                {
                    "target_type": target_type,
                    "target_id": target_id,
                    "unit_id": unit_id,
                    "embedding_model": output["model_id"],
                    "dimension": len(vector),
                    "vector": vector,
                    "text_sha256": digest(text),
                    "run_id": record["run_id"],
                }
            )
    return rows, records


__all__ = ["EMBED_VERSION", "embed_texts"]
