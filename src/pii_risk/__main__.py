from __future__ import annotations

import typer

from pii_risk.eval.audit import audit_records
from pii_risk.pii.detector import detect_pii_spans, redact_text
from pii_risk.pii.scoring import score_record

from pii_risk.ingest.mastodon import ingest_mastodon
from pii_risk.ingest.reddit import ingest_reddit
from pii_risk.ml.combine import combined_score
from pii_risk.ml.predict import predict_risk
from pii_risk.ml.train import train_model

app = typer.Typer(help="PII risk assessment tools.")


@app.command("ingest-reddit")
def ingest_reddit_command(
    input: str = typer.Option(..., "--input", help="Path to JSONL or CSV file."),
    output: str = typer.Option(..., "--output", help="Output directory."),
    max_rows: int | None = typer.Option(None, "--max-rows", help="Max rows to ingest."),
) -> None:
    ingest_reddit(input, output, max_rows)


@app.command("ingest-mastodon")
def ingest_mastodon_command(
    input: str = typer.Option(..., "--input", help="Path to JSONL or CSV file."),
    output: str = typer.Option(..., "--output", help="Output directory."),
    max_rows: int | None = typer.Option(None, "--max-rows", help="Max rows to ingest."),
) -> None:
    ingest_mastodon(input, output, max_rows)


@app.command("analyze-text")
def analyze_text_command(
    text: str = typer.Option(..., "--text", help="Text to analyze."),
) -> None:
    result = score_record(text)
    spans = detect_pii_spans(text)
    redacted = redact_text(text, spans)

    typer.echo(f"score: {result['score']}")
    typer.echo(f"counts_by_type: {result['counts_by_type']}")
    typer.echo(f"redacted_text: {redacted}")


@app.command("train-ml")
def train_ml_command(
    input: str = typer.Option(..., "--input", help="Path to Parquet dataset."),
    max_rows: int | None = typer.Option(None, "--max-rows", help="Max rows to ingest."),
) -> None:
    train_model(input, max_rows=max_rows)


@app.command("analyze-text-ml")
def analyze_text_ml_command(
    text: str = typer.Option(..., "--text", help="Text to analyze."),
) -> None:
    rules = score_record(text)
    spans = detect_pii_spans(text)
    redacted = redact_text(text, spans)
    pii_types = sorted({span.type for span in spans})

    ml_result = predict_risk(text)
    combined = combined_score(rules["score"], ml_result["p_risk"])

    typer.echo(f"rule_score: {rules['score']}")
    typer.echo(f"p_risk: {ml_result['p_risk']:.3f}")
    typer.echo(f"final_score: {combined['final_score']}")
    typer.echo(f"interpretation: {combined['interpretation']}")
    typer.echo(f"detected_pii_types: {pii_types}")
    typer.echo(f"redacted_text: {redacted}")
    typer.echo(f"top_terms: {ml_result['top_terms']}")


@app.command("audit-ml")
def audit_ml_command(
    input: str = typer.Option(..., "--input", help="Path to Parquet dataset."),
    model: str = typer.Option(..., "--model", help="Path to model artifact or folder."),
    out: str = typer.Option(..., "--out", help="Output CSV path."),
    max_rows: int | None = typer.Option(None, "--max-rows", help="Max rows to scan."),
    seed: int = typer.Option(0, "--seed", help="Seed for reproducibility."),
) -> None:
    audit_records(input, model, out, max_rows=max_rows, seed=seed)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
