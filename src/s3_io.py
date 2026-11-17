"""Upload and download files between the local machine and Amazon S3."""
from pathlib import Path

import boto3

from . import config


def _client():
    return boto3.client("s3", region_name=config.AWS_REGION)


def upload_file(local_path: Path, key: str) -> str:
    """Upload a local file to s3://<bucket>/<key> and return the S3 URI."""
    if not config.S3_BUCKET:
        raise ValueError("Set S3_BUCKET in your .env file to use S3.")
    _client().upload_file(str(local_path), config.S3_BUCKET, key)
    uri = f"s3://{config.S3_BUCKET}/{key}"
    print(f"  uploaded {local_path.name} -> {uri}")
    return uri


def download_file(key: str, local_path: Path) -> Path:
    """Download s3://<bucket>/<key> to a local path."""
    if not config.S3_BUCKET:
        raise ValueError("Set S3_BUCKET in your .env file to use S3.")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(config.S3_BUCKET, key, str(local_path))
    print(f"  downloaded s3://{config.S3_BUCKET}/{key} -> {local_path}")
    return local_path
