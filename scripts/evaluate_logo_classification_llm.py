#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

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
    confidence: float | None
    top_candidate_logo: str | None
    matched: bool
    service_matched: bool
    is_correct: bool
    top_candidate_is_correct: bool
    score: float | None
    margin: float | None
    used_full_image_fallback: bool
    token_cost: Any | None = None
    error: str | None = None

    @property
    def predicted_label_for_metrics(self) -> str:
        return self.predicted_logo or UNKNOWN_LABEL

    def as_dict(self) -> dict[str, Any]:
        return {
            "industry": self.industry,
            "expected_logo": self.expected_logo,
            "image_path": self.image_path,
            "predicted_logo": self.predicted_logo,
            "confidence": self.confidence,
            "top_candidate_logo": self.top_candidate_logo,
            "matched": self.matched,
            "service_matched": self.service_matched,
            "is_correct": self.is_correct,
            "top_candidate_is_correct": self.top_candidate_is_correct,
            "score": self.score,
            "margin": self.margin,
            "used_full_image_fallback": self.used_full_image_fallback,
            "token_cost": self.token_cost,
            "error": self.error,
        }


class DatasetHTTPServer:
    def __init__(self, root: Path, host: str, port: int) -> None:
        self.root = root
        self.host = host
        self.port = port
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> str:
        class QuietSimpleHTTPRequestHandler(SimpleHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

        handler = partial(QuietSimpleHTTPRequestHandler, directory=str(self.root))
        self.server = ThreadingHTTPServer((self.host, self.port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        bound_host, bound_port = self.server.server_address[:2]
        return f"http://{bound_host}:{bound_port}"

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate logo-classification performance for the "
            "classify-logo-llm endpoint on a dataset organized as "
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
        help="User ID whose enrolled logo references should be used by the API.",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://127.0.0.1:8000/api/v1",
        help="Base API URL, for example http://127.0.0.1:8000/api/v1.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout in seconds for each classify-logo-llm request.",
    )
    parser.add_argument(
        "--image-base-url",
        help=(
            "Base URL where dataset images are already publicly reachable. "
            "If omitted, the script serves dataset-root over a temporary local HTTP server."
        ),
    )
    parser.add_argument(
        "--image-public-base-url",
        help=(
            "Public base URL for the temporary local HTTP image server. "
            "Useful when the API runs in Docker and must reach the host via a different hostname."
        ),
    )
    parser.add_argument(
        "--local-image-host",
        default="127.0.0.1",
        help="Host for the temporary local HTTP image server when --image-base-url is omitted.",
    )
    parser.add_argument(
        "--local-image-port",
        type=int,
        default=0,
        help="Port for the temporary local HTTP image server. Use 0 for an ephemeral port.",
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
        help="Stop immediately if any image fails to classify.",
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


def average_optional(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    if not available:
        return None
    return sum(available) / len(available)


def format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def extract_token_value(token_cost: Any, key: str) -> float | None:
    if not isinstance(token_cost, dict):
        return None

    value = token_cost.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


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
    label_records = [record for record in records if record.expected_logo == label]
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
    token_usage_samples = sum(1 for record in label_records if record.token_cost is not None)
    average_prompt_tokens = average_optional(
        [extract_token_value(record.token_cost, "prompt_tokens") for record in label_records]
    )
    average_completion_tokens = average_optional(
        [extract_token_value(record.token_cost, "completion_tokens") for record in label_records]
    )
    average_total_tokens = average_optional(
        [extract_token_value(record.token_cost, "total_tokens") for record in label_records]
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
        "token_usage_samples": token_usage_samples,
        "average_prompt_tokens": average_prompt_tokens,
        "average_completion_tokens": average_completion_tokens,
        "average_total_tokens": average_total_tokens,
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
    token_usage_samples = sum(1 for record in processed_records if record.token_cost is not None)
    average_prompt_tokens = average_optional(
        [extract_token_value(record.token_cost, "prompt_tokens") for record in processed_records]
    )
    average_completion_tokens = average_optional(
        [
            extract_token_value(record.token_cost, "completion_tokens")
            for record in processed_records
        ]
    )
    average_total_tokens = average_optional(
        [extract_token_value(record.token_cost, "total_tokens") for record in processed_records]
    )

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
        "token_usage_samples": token_usage_samples,
        "average_prompt_tokens": average_prompt_tokens,
        "average_completion_tokens": average_completion_tokens,
        "average_total_tokens": average_total_tokens,
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
        "records": [record.as_dict() for record in records],
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
                "confidence",
                "top_candidate_logo",
                "matched",
                "service_matched",
                "is_correct",
                "top_candidate_is_correct",
                "score",
                "margin",
                "used_full_image_fallback",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "token_cost",
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
                    "confidence": record.confidence,
                    "top_candidate_logo": record.top_candidate_logo,
                    "matched": record.matched,
                    "service_matched": record.service_matched,
                    "is_correct": record.is_correct,
                    "top_candidate_is_correct": record.top_candidate_is_correct,
                    "score": record.score,
                    "margin": record.margin,
                    "used_full_image_fallback": record.used_full_image_fallback,
                    "prompt_tokens": extract_token_value(record.token_cost, "prompt_tokens"),
                    "completion_tokens": extract_token_value(
                        record.token_cost,
                        "completion_tokens",
                    ),
                    "total_tokens": extract_token_value(record.token_cost, "total_tokens"),
                    "token_cost": (
                        json.dumps(record.token_cost, ensure_ascii=False)
                        if record.token_cost is not None
                        else None
                    ),
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
    if summary["token_usage_samples"]:
        print(
            "  "
            f"avg_total_tokens={format_number(summary['average_total_tokens'])} "
            f"avg_prompt_tokens={format_number(summary['average_prompt_tokens'])} "
            f"avg_completion_tokens={format_number(summary['average_completion_tokens'])} "
            f"token_samples={summary['token_usage_samples']}"
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


def build_sample_image_url(
    *,
    sample: EvaluationSample,
    dataset_root: Path,
    image_base_url: str,
) -> str:
    relative_path = sample.image_path.relative_to(dataset_root)
    encoded_path = "/".join(
        urllib.parse.quote(part)
        for part in relative_path.parts
    )
    return f"{image_base_url.rstrip('/')}/{encoded_path}"


def post_json(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        detail = error_body
        try:
            parsed = json.loads(error_body)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, dict):
                detail = str(parsed.get("detail", parsed))
            else:
                detail = str(parsed)
        raise RuntimeError(f"API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach API: {exc.reason}") from exc

    try:
        parsed_body = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API returned non-JSON response: {body[:500]}") from exc

    if not isinstance(parsed_body, dict):
        raise RuntimeError(f"API returned unexpected payload type: {type(parsed_body).__name__}")
    return parsed_body


def classify_sample(
    *,
    sample: EvaluationSample,
    dataset_root: Path,
    image_base_url: str,
    api_base_url: str,
    user_id: str,
    timeout_seconds: float,
) -> PredictionRecord:
    image_url = build_sample_image_url(
        sample=sample,
        dataset_root=dataset_root,
        image_base_url=image_base_url,
    )
    payload = {
        "user_id": user_id,
        "image_url": image_url,
    }
    response = post_json(
        url=f"{api_base_url.rstrip('/')}/products/classify-logo-llm",
        payload=payload,
        timeout_seconds=timeout_seconds,
    )

    expected_logo = normalize_label(sample.expected_logo)
    predicted_logo = normalize_label(response.get("predicted_logo_name"))
    confidence_value = response.get("confidence")
    confidence = float(confidence_value) if isinstance(confidence_value, int | float) else None
    is_correct = predicted_logo == expected_logo

    return PredictionRecord(
        industry=sample.industry,
        expected_logo=expected_logo or sample.expected_logo,
        image_path=str(sample.image_path),
        predicted_logo=predicted_logo,
        confidence=confidence,
        top_candidate_logo=predicted_logo,
        matched=is_correct,
        service_matched=predicted_logo is not None,
        is_correct=is_correct,
        top_candidate_is_correct=is_correct,
        score=None,
        margin=None,
        used_full_image_fallback=False,
        token_cost=response.get("token_cost"),
    )


def main() -> int:
    args = parse_args()
    if args.image_base_url and args.image_public_base_url:
        raise SystemExit(
            "--image-base-url and --image-public-base-url cannot be used together"
        )

    dataset_root = args.dataset_root.expanduser().resolve()
    samples = discover_samples(dataset_root, limit=args.limit)
    if not samples:
        raise SystemExit(f"No image files found under {dataset_root}")

    records: list[PredictionRecord] = []

    def run_evaluation(image_base_url: str) -> None:
        total = len(samples)
        for index, sample in enumerate(samples, start=1):
            try:
                record = classify_sample(
                    sample=sample,
                    dataset_root=dataset_root,
                    image_base_url=image_base_url,
                    api_base_url=args.api_base_url,
                    user_id=args.user_id,
                    timeout_seconds=args.request_timeout_seconds,
                )
                log_progress(
                    (
                        f"[{index}/{total}] "
                        f"{sample.industry}/{sample.expected_logo}/{sample.image_path.name} -> "
                        f"predicted={record.predicted_label_for_metrics} "
                        f"confidence={format_number(record.confidence)} "
                        f"matched={record.matched}"
                    ),
                    quiet=args.quiet,
                )
            except Exception as exc:
                record = PredictionRecord(
                    industry=sample.industry,
                    expected_logo=sample.expected_logo,
                    image_path=str(sample.image_path),
                    predicted_logo=None,
                    confidence=None,
                    top_candidate_logo=None,
                    matched=False,
                    service_matched=False,
                    is_correct=False,
                    top_candidate_is_correct=False,
                    score=None,
                    margin=None,
                    used_full_image_fallback=False,
                    token_cost=None,
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

    if args.image_base_url:
        run_evaluation(args.image_base_url)
    else:
        with DatasetHTTPServer(
            root=dataset_root,
            host=args.local_image_host,
            port=args.local_image_port,
        ) as local_image_base_url:
            image_base_url = args.image_public_base_url or local_image_base_url
            log_progress(
                (
                    f"Serving dataset images from {local_image_base_url} "
                    f"(using {image_base_url} for API requests)"
                ),
                quiet=args.quiet,
            )
            run_evaluation(image_base_url)

    report = build_report(records)
    report["dataset_root"] = str(dataset_root)
    report["user_id"] = args.user_id
    report["api_base_url"] = args.api_base_url
    report["endpoint"] = "/products/classify-logo-llm"

    if args.output_json:
        write_json(args.output_json.expanduser().resolve(), report)

    if args.output_csv:
        write_csv(args.output_csv.expanduser().resolve(), records)

    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
