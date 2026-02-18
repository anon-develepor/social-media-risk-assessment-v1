from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pii_risk.ml.combine import combined_score
from pii_risk.ml.predict import predict_risk
from pii_risk.pii.detector import detect_pii_spans, redact_text
from pii_risk.pii.scoring import score_record


REQUIRED_MODEL_ARTIFACTS = ("pii_risk_model.pkl", "vectorizer.pkl")
DIVIDER = "-" * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive demo for PII/risk scoring on social media captions.",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory containing ML model artifacts (default: models)",
    )
    parser.add_argument(
        "--text",
        help="Run once on a single caption and exit.",
    )
    parser.add_argument(
        "--show-spans",
        action="store_true",
        help="Show detailed detected PII spans.",
    )
    return parser.parse_args()


def validate_model_dir(models_dir: Path) -> None:
    if not models_dir.exists() or not models_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found or is not a directory: {models_dir}"
        )

    missing = [name for name in REQUIRED_MODEL_ARTIFACTS if not (models_dir / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required model artifacts in {models_dir}: {missing_list}"
        )


def evaluate_caption(text: str, models_dir: Path) -> dict:
    spans = detect_pii_spans(text)
    redacted = redact_text(text, spans)
    rule_result = score_record(text)
    ml_result = predict_risk(text, models_dir=models_dir)
    combined = combined_score(rule_result["score"], ml_result["p_risk"])

    pii_types = sorted({span.type for span in spans})

    return {
        "original": text,
        "redacted": redacted,
        "spans": spans,
        "pii_types": pii_types,
        "rule_score": combined["rule_score"],
        "p_risk": ml_result["p_risk"],
        "final_score": combined["final_score"],
    }


def print_report(result: dict, show_spans: bool) -> None:
    print(DIVIDER)
    print("Original:")
    print(result["original"])
    print()

    print("Redacted:")
    print(result["redacted"])
    print()

    pii_types = ", ".join(result["pii_types"]) if result["pii_types"] else "None"
    print(f"PII types: {pii_types}")

    if show_spans:
        for span in result["spans"]:
            print(f'- {span.type} [{span.start}:{span.end}] "{span.match}"')

    print(f'rule_score: {result["rule_score"]}')
    print(f'p_risk: {result["p_risk"]:.4f}')
    print(f'final_score: {result["final_score"]}')
    print(DIVIDER)


def print_help() -> None:
    print("Commands:")
    print("  :help   Show this help message")
    print("  :q      Quit")
    print("  :quit   Quit")
    print("  (empty input to re-prompt)")


def repl(models_dir: Path, show_spans: bool) -> None:
    print("Interactive caption demo. Type :help for commands.")
    while True:
        try:
            line = input("caption> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not line:
            continue
        if line in {":q", ":quit"}:
            break
        if line == ":help":
            print_help()
            continue

        result = evaluate_caption(line, models_dir)
        print_report(result, show_spans)


def main() -> int:
    args = parse_args()
    models_dir = Path(args.model_dir)

    try:
        validate_model_dir(models_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.text is not None:
        result = evaluate_caption(args.text, models_dir)
        print_report(result, args.show_spans)
        return 0

    repl(models_dir, args.show_spans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
