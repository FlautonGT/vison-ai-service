#!/usr/bin/env python3
"""Delete all S3 objects under a prefix."""

from __future__ import annotations

import argparse
import json

import boto3


def _batched(items: list[dict[str, str]], size: int = 1000):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete all objects under an S3 prefix")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    client = boto3.client("s3", region_name=args.region)
    paginator = client.get_paginator("list_objects_v2")
    keys: list[dict[str, str]] = []
    for page in paginator.paginate(Bucket=args.bucket, Prefix=args.prefix):
        for item in page.get("Contents", []):
            keys.append({"Key": item["Key"]})

    deleted = 0
    for batch in _batched(keys):
        if not batch:
            continue
        response = client.delete_objects(Bucket=args.bucket, Delete={"Objects": batch, "Quiet": True})
        deleted += len(response.get("Deleted", []))

    print(json.dumps({"bucket": args.bucket, "prefix": args.prefix, "deleted": deleted}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
