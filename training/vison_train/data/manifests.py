"""Manifest helpers for task-specific training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(manifest_path)
    frame.attrs["manifest_path"] = str(manifest_path)
    return frame


def resolve_image_paths(frame: pd.DataFrame, image_columns: list[str], root: str | Path | None = None) -> pd.DataFrame:
    resolved = frame.copy()
    base = Path(root).expanduser().resolve() if root else None
    for column in image_columns:
        if column not in resolved.columns:
            raise KeyError(f"Missing image column '{column}' in manifest")
        resolved[column] = resolved[column].map(
            lambda value: str((base / value).resolve()) if base and not Path(str(value)).is_absolute() else str(Path(str(value)).expanduser().resolve())
        )
    return resolved


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise KeyError(f"Manifest is missing required columns: {missing}")


def split_manifest(
    frame: pd.DataFrame,
    group_cols: list[str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_col: str | None = None,
    holdout_column: str | None = None,
    holdout_values: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    if val_ratio < 0.0 or test_ratio < 0.0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to less than 1")
    if not group_cols:
        raise ValueError("At least one group column is required for leakage-safe splitting")

    required = list(group_cols)
    if stratify_col:
        required.append(stratify_col)
    if holdout_column:
        required.append(holdout_column)
    require_columns(frame, required)

    working = frame.copy()
    working["_group_key"] = working[group_cols].astype(str).agg("::".join, axis=1)

    holdout_mask = pd.Series(False, index=working.index)
    if holdout_column and holdout_values:
        holdout_set = {str(value) for value in holdout_values}
        holdout_mask = working[holdout_column].astype(str).isin(holdout_set)

    fixed_test = working.loc[holdout_mask].copy()
    pool = working.loc[~holdout_mask].copy()
    grouped = pool[["_group_key"]].drop_duplicates().reset_index(drop=True)
    if grouped.empty:
        raise ValueError("No samples left after applying holdout rules")

    if stratify_col:
        group_labels = (
            pool.groupby("_group_key")[stratify_col]
            .agg(lambda values: str(pd.Series(values).mode(dropna=False).iloc[0]))
            .to_dict()
        )
        grouped["_label"] = grouped["_group_key"].map(group_labels)
    else:
        grouped["_label"] = "all"

    rng = np.random.default_rng(seed)
    train_groups: list[str] = []
    val_groups: list[str] = []
    test_groups: list[str] = []
    for _label, group_frame in grouped.groupby("_label"):
        keys = group_frame["_group_key"].to_numpy(copy=True)
        rng.shuffle(keys)
        n_keys = len(keys)
        n_test = int(round(n_keys * test_ratio))
        n_val = int(round(n_keys * val_ratio))
        test_groups.extend(keys[:n_test].tolist())
        val_groups.extend(keys[n_test : n_test + n_val].tolist())
        train_groups.extend(keys[n_test + n_val :].tolist())

    train_frame = pool.loc[pool["_group_key"].isin(set(train_groups))].copy()
    val_frame = pool.loc[pool["_group_key"].isin(set(val_groups))].copy()
    dynamic_test = pool.loc[pool["_group_key"].isin(set(test_groups))].copy()
    test_frame = pd.concat([dynamic_test, fixed_test], ignore_index=True)

    leakage = (
        set(train_frame["_group_key"]).intersection(set(val_frame["_group_key"]))
        | set(train_frame["_group_key"]).intersection(set(test_frame["_group_key"]))
        | set(val_frame["_group_key"]).intersection(set(test_frame["_group_key"]))
    )
    if leakage:
        raise RuntimeError(f"Group leakage detected across splits: {sorted(leakage)[:5]}")

    for split_frame in (train_frame, val_frame, test_frame):
        if "_group_key" in split_frame.columns:
            split_frame.drop(columns=["_group_key"], inplace=True)

    return {
        "train": train_frame.reset_index(drop=True),
        "val": val_frame.reset_index(drop=True),
        "test": test_frame.reset_index(drop=True),
    }


def save_manifest(frame: pd.DataFrame, path: str | Path) -> None:
    manifest_path = Path(path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_path, index=False)


def split_summary(splits: dict[str, pd.DataFrame], group_cols: list[str] | None = None) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for name, frame in splits.items():
        payload: dict[str, object] = {"rows": int(len(frame))}
        if group_cols and all(column in frame.columns for column in group_cols):
            payload["groups"] = int(frame[group_cols].astype(str).agg("::".join, axis=1).nunique())
        summary[name] = payload
    return summary


def summary_json(splits: dict[str, pd.DataFrame], group_cols: list[str] | None = None) -> str:
    return json.dumps(split_summary(splits, group_cols=group_cols), indent=2, sort_keys=True)
