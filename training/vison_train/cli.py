"""CLI entrypoints for training, evaluation, and dataset selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import apply_overrides, load_json
from .data.inventory import select_datasets
from .data.manifests import load_manifest, save_manifest, split_manifest, split_summary
from .export import export_onnx
from .runner import evaluate, fit


def _resolved_config(path: str, overrides: list[str]) -> dict:
    return apply_overrides(load_json(path), overrides)


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _cmd_fit(args: argparse.Namespace) -> int:
    config = _resolved_config(args.config, args.override)
    _print_json(fit(config))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    config = _resolved_config(args.config, args.override)
    _print_json(evaluate(config, checkpoint_path=args.checkpoint))
    return 0


def _cmd_select_datasets(args: argparse.Namespace) -> int:
    allowed_statuses = ["approved"]
    if args.allow_restricted:
        allowed_statuses.append("restricted")
    if args.allow_fallback:
        allowed_statuses.append("fallback_only")
    if args.allow_rejected:
        allowed_statuses.append("rejected")
    ranked = select_datasets(
        task=args.task,
        preferred_region=args.preferred_region,
        require_commercial=not args.allow_noncommercial,
        require_modifiable=not args.allow_nonmodifiable,
        allowed_statuses=allowed_statuses,
        path=args.inventory,
    )
    payload = {
        "task": args.task,
        "preferred_region": args.preferred_region,
        "constraints": {
            "commercial_use_ok": not args.allow_noncommercial,
            "modifiable": not args.allow_nonmodifiable,
            "allowed_statuses": allowed_statuses,
        },
        "results": ranked,
    }
    _print_json(payload)
    return 0


def _cmd_split_manifest(args: argparse.Namespace) -> int:
    frame = load_manifest(args.manifest)
    group_cols = _split_csv(args.group_cols)
    splits = split_manifest(
        frame,
        group_cols=group_cols,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_col=args.stratify_col,
        holdout_column=args.holdout_column,
        holdout_values=_split_csv(args.holdout_values),
    )
    save_manifest(splits["train"], args.train_output)
    save_manifest(splits["val"], args.val_output)
    save_manifest(splits["test"], args.test_output)
    payload = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "group_cols": group_cols,
        "summary": split_summary(splits, group_cols=group_cols),
        "outputs": {
            "train": str(Path(args.train_output).expanduser().resolve()),
            "val": str(Path(args.val_output).expanduser().resolve()),
            "test": str(Path(args.test_output).expanduser().resolve()),
        },
    }
    _print_json(payload)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    config = _resolved_config(args.config, args.override)
    _print_json(export_onnx(config, checkpoint_path=args.checkpoint, output_dir=args.output_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vison AI multi-task training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Train a task from a JSON config")
    fit_parser.add_argument("--config", required=True, help="Path to JSON training config")
    fit_parser.add_argument("--override", action="append", default=[], help="Dotted override, e.g. optimization.epochs=20")
    fit_parser.set_defaults(func=_cmd_fit)

    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation from a JSON config")
    eval_parser.add_argument("--config", required=True, help="Path to JSON training config")
    eval_parser.add_argument("--checkpoint", help="Optional checkpoint path")
    eval_parser.add_argument("--override", action="append", default=[], help="Dotted override, e.g. evaluation.report_dir=\"runs/reports\"")
    eval_parser.set_defaults(func=_cmd_evaluate)

    dataset_parser = subparsers.add_parser("select-datasets", help="Rank dataset candidates from the inventory")
    dataset_parser.add_argument("--task", required=True, help="Task id, e.g. verification or passive_pad")
    dataset_parser.add_argument("--preferred-region", default="indonesia", help="Preferred region priority")
    dataset_parser.add_argument("--inventory", help="Optional dataset inventory path")
    dataset_parser.add_argument("--allow-noncommercial", action="store_true", help="Include non-commercial datasets")
    dataset_parser.add_argument("--allow-nonmodifiable", action="store_true", help="Include datasets that disallow derivatives")
    dataset_parser.add_argument("--allow-restricted", action="store_true", help="Include datasets marked restricted")
    dataset_parser.add_argument("--allow-fallback", action="store_true", help="Include datasets marked fallback_only")
    dataset_parser.add_argument("--allow-rejected", action="store_true", help="Include datasets marked rejected")
    dataset_parser.set_defaults(func=_cmd_select_datasets)

    split_parser = subparsers.add_parser("split-manifest", help="Create subject-disjoint train/val/test manifests")
    split_parser.add_argument("--manifest", required=True, help="Source CSV manifest")
    split_parser.add_argument("--group-cols", required=True, help="Comma-separated grouping columns, e.g. subject_id or subject_id,session_id")
    split_parser.add_argument("--train-output", required=True, help="Train CSV output path")
    split_parser.add_argument("--val-output", required=True, help="Validation CSV output path")
    split_parser.add_argument("--test-output", required=True, help="Test CSV output path")
    split_parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    split_parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")
    split_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    split_parser.add_argument("--stratify-col", help="Optional class column for approximate grouped stratification")
    split_parser.add_argument("--holdout-column", help="Optional column whose values should be fully reserved for test")
    split_parser.add_argument("--holdout-values", help="Comma-separated values for the holdout column")
    split_parser.set_defaults(func=_cmd_split_manifest)

    export_parser = subparsers.add_parser("export", help="Export a trained task model to ONNX")
    export_parser.add_argument("--config", required=True, help="Path to JSON training config")
    export_parser.add_argument("--checkpoint", help="Optional checkpoint path")
    export_parser.add_argument("--output-dir", help="Optional export directory")
    export_parser.add_argument("--override", action="append", default=[], help="Dotted override")
    export_parser.set_defaults(func=_cmd_export)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
