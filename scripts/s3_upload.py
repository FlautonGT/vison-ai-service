#!/usr/bin/env python3
"""Upload logs and artifacts to S3 without requiring awscli."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Iterable

import boto3


def _iter_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(item for item in path.rglob("*") if item.is_file()))
        elif path.is_file():
            files.append(path)
    return files


def _object_key(path: Path, root: Path, key_prefix: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    key = "/".join(part for part in [key_prefix.strip("/"), relative.as_posix()] if part)
    return key


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload files or directories to S3")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default=os.getenv("S3_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-southeast-1")
    parser.add_argument("--root", default="/workspace")
    parser.add_argument("--key-prefix", default=os.getenv("S3_PREFIX", "vison-training"))
    parser.add_argument("--endpoint-url", default=os.getenv("S3_ENDPOINT_URL"))
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--path", action="append", default=[], help="File or directory to upload. May be repeated.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    raw_paths = [Path(item).expanduser().resolve() for item in args.path]
    missing = [str(path) for path in raw_paths if not path.exists()]
    if missing and not args.skip_missing:
        raise FileNotFoundError(f"Missing upload paths: {missing}")

    upload_paths = [path for path in raw_paths if path.exists()]
    files = _iter_files(upload_paths)
    session = boto3.session.Session(region_name=args.region)
    client = session.client("s3", endpoint_url=args.endpoint_url)

    uploaded = []
    for file_path in files:
        key = _object_key(file_path, root=root, key_prefix=args.key_prefix)
        extra_args = {}
        content_type, _encoding = mimetypes.guess_type(str(file_path))
        if content_type:
            extra_args["ContentType"] = content_type
        if extra_args:
            client.upload_file(str(file_path), args.bucket, key, ExtraArgs=extra_args)
        else:
            client.upload_file(str(file_path), args.bucket, key)
        uploaded.append({"path": str(file_path), "s3_key": key, "size_bytes": file_path.stat().st_size})

    print(
        json.dumps(
            {
                "bucket": args.bucket,
                "region": args.region,
                "key_prefix": args.key_prefix,
                "uploaded_files": len(uploaded),
                "missing_paths": missing,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
