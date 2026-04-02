#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UNKNOWN_LABEL = "__unknown__"
SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
_PIPELINE = None


@dataclass(frozen=True)
class EvaluationSample:
    industry: str
    expected_logo: str
    image_path: Path


@dataclass
class PredictionRecord:
    industry: str
    expected_logo: str
    image_path: str
    predicted_logo: str | None
    top_candidate_logo: str | None
    matched: bool
    service_matched: bool
    is_correct: bool
    top_candidate_is_correct: bool
    score: float | None
    margin: float | None
    used_full_image_fallback: bool
    error: str | None = None

    @property
    def predicted_label_for_metrics(self) -> str:
        return self.predicted_logo or UNKNOWN_LABEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate logo-classification performance on a dataset organized as "
            "industry/logo_name/image_files."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/jayson/Downloads/logo data/test-data"),
        help="Root folder containing industry/logo_name/image_files.",
    )
    parser.add_argument(
        "--user-id",
        default="user-1",
        help="User ID whose enrolled logo references should be searched in Qdrant.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the summary report as JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to write one row per evaluated image.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of images to evaluate, for quick smoke tests.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any image fails to load or classify.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image progress logs.",
    )
    return parser.parse_args()


def normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def macro_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def discover_samples(dataset_root: Path, limit: int | None = None) -> list[EvaluationSample]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    samples: list[EvaluationSample] = []
    for industry_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for logo_dir in sorted(path for path in industry_dir.iterdir() if path.is_dir()):
            for image_path in sorted(logo_dir.rglob("*")):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                samples.append(
                    EvaluationSample(
                        industry=industry_dir.name.strip(),
                        expected_logo=logo_dir.name.strip(),
                        image_path=image_path,
                    )
                )
                if limit is not None and len(samples) >= limit:
                    return samples

    return samples


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        transposed = ImageOps.exif_transpose(image)
        return transposed.convert("RGB")


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from app.dependencies import get_logo_pipeline_service

        _PIPELINE = get_logo_pipeline_service()
    return _PIPELINE


def classify_sample(sample: EvaluationSample, user_id: str) -> PredictionRecord:
    pipeline = get_pipeline()
    image = load_rgb_image(sample.image_path)
    outcome = pipeline.classify(user_id=user_id, image=image)

    expected_logo = normalize_label(sample.expected_logo)
    top_candidate_logo = (
        normalize_label(outcome.candidates[0].logo_name) if outcome.candidates else None
    )
    predicted_logo = top_candidate_logo or normalize_label(outcome.predicted_logo_name)
    is_correct = predicted_logo == expected_logo

    return PredictionRecord(
        industry=sample.industry,
        expected_logo=expected_logo or sample.expected_logo,
        image_path=str(sample.image_path),
        predicted_logo=predicted_logo,
        top_candidate_logo=top_candidate_logo,
        matched=is_correct,
        service_matched=outcome.matched,
        is_correct=is_correct,
        top_candidate_is_correct=top_candidate_logo == expected_logo,
        score=outcome.score,
        margin=outcome.margin,
        used_full_image_fallback=outcome.used_full_image_fallback,
    )


def ranked_counter(counter: Counter[str], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {"label": label, "count": count}
        for label, count in counter.most_common(limit)
    ]


def build_logo_metrics(
    records: list[PredictionRecord],
    label: str,
) -> dict[str, Any]:
    total = len(records)
    support = sum(1 for record in records if record.expected_logo == label)
    tp = sum(
        1
        for record in records
        if record.expected_logo == label and record.predicted_label_for_metrics == label
    )
    fp = sum(
        1
        for record in records
        if record.expected_logo != label and record.predicted_label_for_metrics == label
    )
    fn = sum(
        1
        for record in records
        if record.expected_logo == label and record.predicted_label_for_metrics != label
    )
    tn = total - tp - fp - fn

    precision = rate(tp, tp + fp)
    recall = rate(tp, tp + fn)
    f1_score = rate(2 * precision * recall, precision + recall) if precision or recall else 0.0
    one_vs_rest_accuracy = rate(tp + tn, total)
    matched_count = sum(
        1 for record in records if record.expected_logo == label and record.matched
    )
    service_matched_count = sum(
        1 for record in records if record.expected_logo == label and record.service_matched
    )
    unknown_count = sum(
        1
        for record in records
        if record.expected_logo == label and record.predicted_label_for_metrics == UNKNOWN_LABEL
    )
    top_candidate_hits = sum(
        1 for record in records if record.expected_logo == label and record.top_candidate_is_correct
    )
    fallback_count = sum(
        1
        for record in records
        if record.expected_logo == label and record.used_full_image_fallback
    )

    misclassifications = Counter(
        record.predicted_label_for_metrics
        for record in records
        if record.expected_logo == label and record.predicted_label_for_metrics != label
    )

    return {
        "images": support,
        "correct": tp,
        "correctness_rate": rate(tp, support),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "one_vs_rest_accuracy": one_vs_rest_accuracy,
        "matched_rate": rate(matched_count, support),
        "service_matched_rate": rate(service_matched_count, support),
        "unknown_rate": rate(unknown_count, support),
        "top_candidate_hit_rate": rate(top_candidate_hits, support),
        "used_full_image_fallback_rate": rate(fallback_count, support),
        "top_misclassifications": ranked_counter(misclassifications),
    }


def build_group_report(
    processed_records: list[PredictionRecord],
    failed_records: list[PredictionRecord],
) -> dict[str, Any]:
    labels = sorted({record.expected_logo for record in processed_records})
    per_logo = {
        label: build_logo_metrics(processed_records, label)
        for label in labels
    }

    correct = sum(1 for record in processed_records if record.is_correct)
    matched_count = sum(1 for record in processed_records if record.matched)
    service_matched_count = sum(1 for record in processed_records if record.service_matched)
    unknown_count = sum(
        1 for record in processed_records if record.predicted_label_for_metrics == UNKNOWN_LABEL
    )
    top_candidate_hits = sum(1 for record in processed_records if record.top_candidate_is_correct)
    fallback_count = sum(1 for record in processed_records if record.used_full_image_fallback)

    overall_misclassifications = Counter(
        f"{record.expected_logo} -> {record.predicted_label_for_metrics}"
        for record in processed_records
        if not record.is_correct
    )

    logo_precisions = [metrics["precision"] for metrics in per_logo.values()]
    logo_recalls = [metrics["recall"] for metrics in per_logo.values()]
    logo_f1_scores = [metrics["f1_score"] for metrics in per_logo.values()]

    discovered_images = len(processed_records) + len(failed_records)
    processed_images = len(processed_records)

    summary = {
        "discovered_images": discovered_images,
        "processed_images": processed_images,
        "failed_images": len(failed_records),
        "processing_success_rate": rate(processed_images, discovered_images),
        "correct": correct,
        "accuracy": rate(correct, processed_images),
        "correctness_rate": rate(correct, processed_images),
        "macro_precision": macro_average(logo_precisions),
        "macro_recall": macro_average(logo_recalls),
        "macro_f1_score": macro_average(logo_f1_scores),
        "matched_rate": rate(matched_count, processed_images),
        "service_matched_rate": rate(service_matched_count, processed_images),
        "unknown_rate": rate(unknown_count, processed_images),
        "top_candidate_hit_rate": rate(top_candidate_hits, processed_images),
        "used_full_image_fallback_rate": rate(fallback_count, processed_images),
        "logo_count": len(labels),
        "top_misclassifications": ranked_counter(overall_misclassifications),
    }

    return {
        "summary": summary,
        "logos": per_logo,
    }


def build_report(records: list[PredictionRecord]) -> dict[str, Any]:
    processed_records = [record for record in records if record.error is None]
    failed_records = [record for record in records if record.error is not None]

    overall = build_group_report(processed_records, failed_records)

    industries: dict[str, dict[str, Any]] = {}
    industry_names = sorted({record.industry for record in records})
    for industry_name in industry_names:
        industry_processed = [
            record
            for record in processed_records
            if record.industry == industry_name
        ]
        industry_failed = [
            record
            for record in failed_records
            if record.industry == industry_name
        ]
        industries[industry_name] = build_group_report(industry_processed, industry_failed)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "industries": industries,
        "errors": [
            {
                "industry": record.industry,
                "expected_logo": record.expected_logo,
                "image_path": record.image_path,
                "error": record.error,
            }
            for record in failed_records
        ],
    }


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, records: list[PredictionRecord]) -> None:
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "industry",
                "expected_logo",
                "image_path",
                "predicted_logo",
                "top_candidate_logo",
                "matched",
                "service_matched",
                "is_correct",
                "top_candidate_is_correct",
                "score",
                "margin",
                "used_full_image_fallback",
                "error",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "industry": record.industry,
                    "expected_logo": record.expected_logo,
                    "image_path": record.image_path,
                    "predicted_logo": record.predicted_logo,
                    "top_candidate_logo": record.top_candidate_logo,
                    "matched": record.matched,
                    "service_matched": record.service_matched,
                    "is_correct": record.is_correct,
                    "top_candidate_is_correct": record.top_candidate_is_correct,
                    "score": record.score,
                    "margin": record.margin,
                    "used_full_image_fallback": record.used_full_image_fallback,
                    "error": record.error,
                }
            )


def print_summary_block(title: str, summary: dict[str, Any]) -> None:
    print(title)
    print(
        "  "
        f"discovered={summary['discovered_images']} "
        f"processed={summary['processed_images']} "
        f"failed={summary['failed_images']} "
        f"success={format_pct(summary['processing_success_rate'])}"
    )
    print(
        "  "
        f"accuracy={format_pct(summary['accuracy'])} "
        f"macro_precision={format_pct(summary['macro_precision'])} "
        f"macro_recall={format_pct(summary['macro_recall'])} "
        f"macro_f1={format_pct(summary['macro_f1_score'])}"
    )
    print(
        "  "
        f"matched={format_pct(summary['matched_rate'])} "
        f"service_matched={format_pct(summary['service_matched_rate'])} "
        f"unknown={format_pct(summary['unknown_rate'])} "
        f"top_candidate_hit={format_pct(summary['top_candidate_hit_rate'])} "
        f"fallback={format_pct(summary['used_full_image_fallback_rate'])}"
    )
    if summary["top_misclassifications"]:
        preview = ", ".join(
            f"{entry['label']} ({entry['count']})"
            for entry in summary["top_misclassifications"]
        )
        print(f"  top_misclassifications={preview}")


def print_logo_table(logos: dict[str, dict[str, Any]]) -> None:
    if not logos:
        print("  No successfully processed logos in this group.")
        return

    print(
        "  "
        "logo | images | correct | correctness | precision | recall | "
        "f1 | matched | service_matched | unknown | top_hit"
    )
    for logo_name, metrics in logos.items():
        print(
            "  "
            f"{logo_name} | "
            f"{metrics['images']} | "
            f"{metrics['correct']} | "
            f"{format_pct(metrics['correctness_rate'])} | "
            f"{format_pct(metrics['precision'])} | "
            f"{format_pct(metrics['recall'])} | "
            f"{format_pct(metrics['f1_score'])} | "
            f"{format_pct(metrics['matched_rate'])} | "
            f"{format_pct(metrics['service_matched_rate'])} | "
            f"{format_pct(metrics['unknown_rate'])} | "
            f"{format_pct(metrics['top_candidate_hit_rate'])}"
        )


def print_report(report: dict[str, Any]) -> None:
    print_summary_block("Overall", report["overall"]["summary"])
    print()
    print("Per-logo (overall)")
    print_logo_table(report["overall"]["logos"])

    for industry_name, industry_report in report["industries"].items():
        print()
        print_summary_block(f"Industry: {industry_name}", industry_report["summary"])
        print_logo_table(industry_report["logos"])

    if report["errors"]:
        print()
        print("Errors")
        for error in report["errors"]:
            print(
                "  "
                f"{error['industry']}/{error['expected_logo']} | "
                f"{error['image_path']} | "
                f"{error['error']}"
            )


def log_progress(message: str, quiet: bool) -> None:
    if quiet:
        return
    print(message, file=sys.stderr, flush=True)


def main() -> int:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    samples = discover_samples(dataset_root, limit=args.limit)
    if not samples:
        raise SystemExit(f"No image files found under {dataset_root}")

    records: list[PredictionRecord] = []
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        try:
            record = classify_sample(sample, user_id=args.user_id)
            log_progress(
                (
                    f"[{index}/{total}] "
                    f"{sample.industry}/{sample.expected_logo}/{sample.image_path.name} -> "
                    f"predicted={record.predicted_label_for_metrics} "
                    f"matched={record.matched} "
                    f"score={record.score}"
                ),
                quiet=args.quiet,
            )
        except Exception as exc:
            record = PredictionRecord(
                industry=sample.industry,
                expected_logo=sample.expected_logo,
                image_path=str(sample.image_path),
                predicted_logo=None,
                top_candidate_logo=None,
                matched=False,
                service_matched=False,
                is_correct=False,
                top_candidate_is_correct=False,
                score=None,
                margin=None,
                used_full_image_fallback=False,
                error=str(exc),
            )
            log_progress(
                (
                    f"[{index}/{total}] "
                    f"{sample.industry}/{sample.expected_logo}/{sample.image_path.name} -> "
                    f"ERROR: {exc}"
                ),
                quiet=args.quiet,
            )
            if args.stop_on_error:
                raise

        records.append(record)

    report = build_report(records)
    report["dataset_root"] = str(dataset_root)
    report["user_id"] = args.user_id

    if args.output_json:
        write_json(args.output_json.expanduser().resolve(), report)

    if args.output_csv:
        write_csv(args.output_csv.expanduser().resolve(), records)

    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
