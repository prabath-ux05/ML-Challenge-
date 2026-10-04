"""Full Test Inference runner on AWS SageMaker.

Automates:
1. Code packaging and uploading to S3.
2. Creating and launching the SageMaker Processing Job with ml.m5.4xlarge.
3. Polling and streaming CloudWatch logs until completion.
4. Downloading the full output (candidate_pairs.tsv, matching_results.tsv, pipeline_manifest.json).
5. Running the final SubmissionValidator on the complete 1,732,544 S1 entities.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time
from typing import Any, Dict, Optional
import boto3
from botocore.exceptions import ClientError

from business_entity_resolution.pipeline.validator import SubmissionValidator
from business_entity_resolution.utils.io import read_tsv
from business_entity_resolution.utils.logging import get_logger

logger = get_logger("sagemaker.inference")

AWS_REGION = "ap-southeast-1"
S3_BUCKET = "business-entity-resolution-data-2026"
ROLE_ARN = "arn:aws:iam::483199421282:role/service-role/AmazonSageMaker-ExecutionRole-20260927T144362"
IMAGE_URI = "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:2.1.0-cpu-py310-ubuntu20.04-ec2"
INSTANCE_TYPE = "ml.m5.4xlarge"


def create_code_archive(archive_path: Path) -> None:
    """Package project source code and requirements into a tar.gz archive for SageMaker."""
    project_root = Path(__file__).resolve().parent
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(project_root / "src", arcname="src")
        req_file = project_root / "requirements.txt"
        if req_file.exists():
            tar.add(req_file, arcname="requirements.txt")
        pyproj = project_root / "pyproject.toml"
        if pyproj.exists():
            tar.add(pyproj, arcname="pyproject.toml")
        run_sh = project_root / "run_job.sh"
        if run_sh.exists():
            tar.add(run_sh, arcname="run_job.sh")


def upload_code_to_s3(s3_client: boto3.client, local_tar: Path) -> str:
    """Upload packaged code to S3 bucket."""
    timestamp = datetime.now(datetime.timezone.utc if hasattr(datetime, "timezone") else None).strftime("%Y%m%d-%H%M%S")
    s3_key = f"code/source-{timestamp}.tar.gz"
    logger.info("Uploading code archive to s3://%s/%s ...", S3_BUCKET, s3_key)
    s3_client.upload_file(str(local_tar), S3_BUCKET, s3_key)
    return f"s3://{S3_BUCKET}/{s3_key}"


def launch_processing_job(
    sm_client: boto3.client,
    code_s3_uri: str,
    job_name: Optional[str] = None,
) -> str:
    """Create and trigger the SageMaker Processing Job."""
    timestamp = datetime.now(datetime.timezone.utc if hasattr(datetime, "timezone") else None).strftime("%Y%m%d-%H%M%S")
    job_name = job_name or f"er-full-test-{timestamp}"
    logger.info("Launching SageMaker Processing Job: %s (Instance: %s)...", job_name, INSTANCE_TYPE)

    entrypoint_script = ["bash", "/opt/ml/processing/input/code/run_job.sh"]

    processing_inputs = [
        {
            "InputName": "code",
            "S3Input": {
                "S3Uri": code_s3_uri,
                "LocalPath": "/opt/ml/processing/input/code",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
            },
        },
        {
            "InputName": "source1",
            "S3Input": {
                "S3Uri": f"s3://{S3_BUCKET}/business-entity-resolution/test/test/test_source1.tsv",
                "LocalPath": "/opt/ml/processing/input/source1",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
            },
        },
        {
            "InputName": "source2",
            "S3Input": {
                "S3Uri": f"s3://{S3_BUCKET}/business-entity-resolution/test/test/test_source2.tsv",
                "LocalPath": "/opt/ml/processing/input/source2",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
            },
        },
        {
            "InputName": "source3",
            "S3Input": {
                "S3Uri": f"s3://{S3_BUCKET}/business-entity-resolution/test/test/test_source3.tsv",
                "LocalPath": "/opt/ml/processing/input/source3",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
            },
        },
        {
            "InputName": "model",
            "S3Input": {
                "S3Uri": f"s3://{S3_BUCKET}/models/",
                "LocalPath": "/opt/ml/processing/model",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
            },
        },
    ]

    output_s3_uri = f"s3://{S3_BUCKET}/submissions/{job_name}/"
    processing_outputs = [
        {
            "OutputName": "submission",
            "S3Output": {
                "S3Uri": output_s3_uri,
                "LocalPath": "/opt/ml/processing/output",
                "S3UploadMode": "EndOfJob",
            },
        }
    ]

    sm_client.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=ROLE_ARN,
        AppSpecification={
            "ImageUri": IMAGE_URI,
            "ContainerEntrypoint": entrypoint_script,
        },
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": INSTANCE_TYPE,
                "VolumeSizeInGB": 100,
            }
        },
        ProcessingInputs=processing_inputs,
        ProcessingOutputConfig={"Outputs": processing_outputs},
        StoppingCondition={"MaxRuntimeInSeconds": 86400},  # 24 hours max
    )

    logger.info("Processing job successfully created: %s", job_name)
    logger.info("Outputs will be saved to: %s", output_s3_uri)
    return job_name


def wait_for_job(sm_client: boto3.client, job_name: str, poll_interval_sec: int = 60) -> Dict[str, Any]:
    """Poll processing job status until terminal state."""
    logger.info("Monitoring SageMaker processing job: %s ...", job_name)
    while True:
        desc = sm_client.describe_processing_job(ProcessingJobName=job_name)
        status = desc["ProcessingJobStatus"]
        logger.info("Job Status: %s", status)

        if status == "Completed":
            logger.info("Job completed successfully!")
            return desc
        elif status in ("Failed", "Stopped"):
            failure_reason = desc.get("FailureReason", "No failure reason provided.")
            logger.error("Job ended with status %s: %s", status, failure_reason)
            raise RuntimeError(f"Processing job {job_name} failed: {failure_reason}")

        time.sleep(poll_interval_sec)


def download_outputs(s3_client: boto3.client, job_name: str, local_output_dir: Path) -> None:
    """Download generated output files from S3 to local output directory."""
    local_output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"submissions/{job_name}/"

    logger.info("Downloading final outputs from s3://%s/%s -> %s", S3_BUCKET, prefix, local_output_dir)
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = Path(key).name
            if not fname:
                continue
            dest = local_output_dir / fname
            logger.info("Downloading %s -> %s", key, dest)
            s3_client.download_file(S3_BUCKET, key, str(dest))


def validate_complete_test_submission(local_output_dir: Path) -> None:
    """Execute SubmissionValidator against the complete 1,732,544 test S1 entities."""
    m_path = local_output_dir / "matching_results.tsv"
    c_path = local_output_dir / "candidate_pairs.tsv"

    logger.info("Reading complete test Source 1 entities for validation...")
    project_root = Path(__file__).resolve().parent
    s1_path = project_root / "data" / "raw" / "test_source1.tsv"

    df_s1 = read_tsv(s1_path)
    expected_s1_ids = set(df_s1["entity_id"].astype(str))
    total_expected = len(expected_s1_ids)
    logger.info("Total expected S1 queries: %d", total_expected)

    logger.info("Running SubmissionValidator on complete output...")
    report = SubmissionValidator.validate_matching_results(
        matching_results_path=m_path,
        candidate_pairs_path=c_path,
        expected_s1_ids=expected_s1_ids,
        require_deterministic_order=True,
    )

    print(report.summary())
    if not report.is_valid:
        logger.error("Submission validation failed! %d errors detected.", len(report.errors))
        sys.exit(1)

    logger.info("Submission validation PASSED on all %d S1 entities!", total_expected)


def main():
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    sm_client = boto3.client("sagemaker", region_name=AWS_REGION)

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "source.tar.gz"
        create_code_archive(tar_path)
        code_s3_uri = upload_code_to_s3(s3_client, tar_path)

    try:
        job_name = launch_processing_job(sm_client, code_s3_uri)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        logger.error("Failed to launch SageMaker job [%s]: %s", code, msg)
        if "ResourceLimitExceeded" in code:
            print("\n" + "=" * 80)
            print("ACTION REQUIRED: AWS SAGEMAKER SERVICE QUOTA INCREASE")
            print("=" * 80)
            print(f"Error: {msg}")
            print("\nHow to request quota increase:")
            print("1. Go to AWS Console -> Service Quotas -> AWS Services -> Amazon SageMaker")
            print("   URL: https://ap-southeast-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas")
            print(f"2. Search for: '{INSTANCE_TYPE} for processing job usage'")
            print("3. Click 'Request quota increase' and enter: 1 (or 2)")
            print("4. Once approved by AWS, rerun this script:")
            print("   python run_sagemaker_inference.py")
            print("=" * 80 + "\n")
        sys.exit(1)

    desc = wait_for_job(sm_client, job_name)
    local_out = Path("output")
    download_outputs(s3_client, job_name, local_out)
    validate_complete_test_submission(local_out)


if __name__ == "__main__":
    main()
