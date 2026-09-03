"""
Deploy SettleSense to Hugging Face Spaces (Streamlit SDK — FREE tier).

Usage:
    python deploy_to_hf.py

Requires:
    HF_TOKEN in .env (format: hf_...)
    pip install huggingface_hub>=0.20.0

What this does:
    1. Creates a Streamlit Space (not Docker) — works on free HF accounts
    2. Uploads all project files except secrets (.env, *.db, __pycache__)
    3. The Space README.md is taken from README_HF.md (contains HF YAML frontmatter)
"""
import os
import sys
import shutil
import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent
load_dotenv(project_root / ".env")

HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN.startswith("hf_"):
    logger.error("HF_TOKEN not found or invalid format. Set HF_TOKEN=hf_... in .env")
    sys.exit(1)

from huggingface_hub import HfApi, create_repo, upload_folder, whoami

# Authenticate
try:
    user_info = whoami(token=HF_TOKEN)
    username = user_info["name"]
    logger.info(f"Authenticated as: {username}")
except Exception as e:
    logger.error(f"HF authentication failed: {e}")
    sys.exit(1)

SPACE_NAME = "settlesense"
REPO_ID = f"{username}/{SPACE_NAME}"
api = HfApi(token=HF_TOKEN)

# Create Space (Streamlit SDK — free tier)
logger.info(f"Creating/updating Space: {REPO_ID}")
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",      # HF API accepts: gradio | docker | static
        private=False,
        token=HF_TOKEN,
        exist_ok=True,
    )
    logger.info("Space ready.")
except Exception as e:
    logger.error(f"Failed to create Space: {e}")
    sys.exit(1)

# Files/dirs to exclude from upload
IGNORE_PATTERNS = [
    ".env",
    ".env.local",
    ".git",
    ".git/**",
    "__pycache__",
    "**/__pycache__/**",
    "*.pyc",
    "*.db",
    "*.db-shm",
    "*.db-wal",
    "*.pkl",
    ".pytest_cache",
    "deploy_to_hf.py",   # Don't upload deploy script itself
    "data/razorpay_live", # Don't upload live API responses with potential secrets
    "Dockerfile",         # Not needed for Streamlit SDK spaces
    "docker-compose.yml", # Not needed for Streamlit SDK spaces
]

# Stage upload — rename README_HF.md → README.md for the Space
# (HF Space uses README.md as the Space card)
logger.info("Staging files for upload...")
with tempfile.TemporaryDirectory() as tmp_dir:
    tmp = Path(tmp_dir)

    # Copy project files
    for item in project_root.iterdir():
        name = item.name
        # Skip ignored items
        if any(name == p or name.startswith(".git") for p in [
            ".env", ".git", "__pycache__", "deploy_to_hf.py",
            "Dockerfile", "docker-compose.yml", "README_HF.md"
        ]):
            continue
        if name.endswith((".db", ".db-shm", ".db-wal", ".pkl", ".pyc")):
            continue
        if item.is_dir():
            if name in (".pytest_cache", "__pycache__"):
                continue
            shutil.copytree(str(item), str(tmp / name), dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.db",
                                                          "*.db-shm", "*.db-wal", "*.pkl",
                                                          "razorpay_live"))
        else:
            shutil.copy2(str(item), str(tmp / name))

    # Use README_HF.md as the Space README.md
    hf_readme = project_root / "README_HF.md"
    if hf_readme.exists():
        shutil.copy2(str(hf_readme), str(tmp / "README.md"))
        logger.info("Using README_HF.md as HF Space README.md")

    # Upload
    logger.info(f"Uploading to https://huggingface.co/spaces/{REPO_ID} ...")
    upload_folder(
        folder_path=str(tmp),
        repo_id=REPO_ID,
        repo_type="space",
        token=HF_TOKEN,
        commit_message="SettleSense v1.0 — Razorpay AI Buildathon Track 04",
        ignore_patterns=["*.db", "*.db-shm", "*.db-wal", "*.pyc", "__pycache__"],
    )

SPACE_URL = f"https://huggingface.co/spaces/{REPO_ID}"
logger.info("")
logger.info("=" * 60)
logger.info(f"Deployment complete!")
logger.info(f"Space URL  : {SPACE_URL}")
logger.info(f"App URL    : https://{username.replace('_','-')}-{SPACE_NAME}.hf.space")
logger.info("")
logger.info("To enable real AI, add secrets in Space Settings:")
logger.info("  ANTHROPIC_API_KEY = sk-ant-...")
logger.info("  LLM_PROVIDER = anthropic")
logger.info("=" * 60)
print(f"\nSPACE_URL={SPACE_URL}")
