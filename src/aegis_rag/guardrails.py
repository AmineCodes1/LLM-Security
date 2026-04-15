from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(slots=True)
class ContextScanResult:
    flagged_chunks: list[int]
    reasons: list[str]


@dataclass(slots=True)
class OutputValidationResult:
    is_safe: bool
    reasons: list[str]


class Shield:
    """Keep guardrails separate from RAG logic to allow easy A/B comparison."""

    def __init__(self, risky_keywords: list[str] | None = None) -> None:
        self.risky_keywords = risky_keywords or [
            "ignore",
            "system update",
            "command",
            "api key",
            "password",
            "credentials",
        ]

    def scan_context(self, documents: list[Document]) -> ContextScanResult:
        flagged: list[int] = []
        reasons: list[str] = []

        for index, doc in enumerate(documents):
            lowered = doc.page_content.lower()
            hits = [kw for kw in self.risky_keywords if kw in lowered]
            if hits:
                flagged.append(index)
                reasons.append(f"Chunk {index + 1} contains risky terms: {', '.join(hits)}")

        return ContextScanResult(flagged_chunks=flagged, reasons=reasons)

    def validate_output(self, answer: str) -> OutputValidationResult:
        lowered = answer.lower()
        suspect_tokens = [
            "api key",
            "password",
            "token",
            "secret",
            "credentials",
        ]

        hits = [token for token in suspect_tokens if token in lowered]
        if hits:
            return OutputValidationResult(
                is_safe=False,
                reasons=[f"Response requests or mentions sensitive data: {', '.join(hits)}"],
            )

        return OutputValidationResult(is_safe=True, reasons=[])
