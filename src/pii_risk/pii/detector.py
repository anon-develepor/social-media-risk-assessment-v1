from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import re


class PIIType:
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    URL = "URL"
    IPV4 = "IPV4"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    DOB = "DOB"
    ADDRESS_HINT = "ADDRESS_HINT"


@dataclass(frozen=True)
class PIISpan:
    type: str
    start: int
    end: int
    match: str


PII_PATTERNS: dict[str, re.Pattern[str]] = {
    PIIType.EMAIL: re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    PIIType.PHONE: re.compile(
        r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"
    ),
    PIIType.URL: re.compile(r"\bhttps?://[^\s]+\b"),
    PIIType.IPV4: re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    PIIType.SSN: re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    PIIType.CREDIT_CARD: re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
    PIIType.DOB: re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    PIIType.ADDRESS_HINT: re.compile(
        r"\b\d{1,5}\s+\w+(?:\s+\w+){0,3}\s+"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b",
        re.IGNORECASE,
    ),
}


def _iter_spans(text: str) -> Iterable[PIISpan]:
    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            yield PIISpan(
                type=pii_type,
                start=match.start(),
                end=match.end(),
                match=match.group(0),
            )


def detect_pii_spans(text: str | None) -> list[PIISpan]:
    if not text:
        return []

    spans = sorted(_iter_spans(text), key=lambda span: (span.start, span.end))
    filtered: list[PIISpan] = []

    for span in spans:
        if not filtered:
            filtered.append(span)
            continue

        last = filtered[-1]
        if span.start < last.end:
            if (span.end - span.start) > (last.end - last.start):
                filtered[-1] = span
            continue

        filtered.append(span)

    return filtered


def redact_text(text: str, spans: list[PIISpan]) -> str:
    if not text or not spans:
        return text

    redacted = text
    for span in sorted(spans, key=lambda item: (item.start, item.end), reverse=True):
        redacted = (
            f"{redacted[:span.start]}[REDACTED:{span.type}]{redacted[span.end:]}"
        )
    return redacted
