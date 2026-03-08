"""
Build job router.

POST /build/full         — trigger full 4-stage pipeline
POST /build/incremental  — trigger incremental update
GET  /build/jobs         — list all jobs
GET  /build/jobs/{id}    — get single job status
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException

from models import TriggerResponse, JobResponse
from build_pipeline.job_store import job_store
from build_pipeline.runner import run_full_build, run_incremental_build
from config.settings import FILES_DIR, FILE_REGISTRY_PATH, SEC_PARSER_URL, SEC_FILES_DIR

router = APIRouter(prefix="/build")

# Single-worker executor so builds run one at a time
_executor = ThreadPoolExecutor(max_workers=1)


@router.post("/full", response_model=TriggerResponse)
async def trigger_full_build():
    """Trigger a full graph build (drop indexes → build graph → index community → chunk index)."""
    # Reject if a build is already running
    running = [j for j in job_store.list_all() if j["status"] == "running"]
    if running:
        raise HTTPException(status_code=409, detail=f"A build is already running: {running[0]['job_id']}")

    job = job_store.create("full")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, run_full_build, job.job_id, SEC_FILES_DIR, SEC_PARSER_URL)
    return TriggerResponse(job_id=job.job_id, message="Full build started")


@router.post("/incremental", response_model=TriggerResponse)
async def trigger_incremental_build():
    """Trigger an incremental update (detects changed files and updates graph)."""
    running = [j for j in job_store.list_all() if j["status"] == "running"]
    if running:
        raise HTTPException(status_code=409, detail=f"A build is already running: {running[0]['job_id']}")

    job = job_store.create("incremental")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, run_incremental_build, job.job_id, FILES_DIR, FILE_REGISTRY_PATH)
    return TriggerResponse(job_id=job.job_id, message="Incremental build started")


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs():
    """List all build jobs."""
    return job_store.list_all()


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get status of a specific job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()
