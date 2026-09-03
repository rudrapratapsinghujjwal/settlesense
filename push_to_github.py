"""
Push SettleSense to GitHub (creates repo if needed) and gives Streamlit Cloud URL.

Requires:
    GITHUB_TOKEN in .env (GitHub Personal Access Token with repo scope)
    OR create repo manually at github.com and run:
        git remote add origin https://github.com/USERNAME/settlesense.git
        git push -u origin master
"""
import os, sys, json, subprocess
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent
load_dotenv(project_root / ".env")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "")

if not GITHUB_TOKEN or not GITHUB_USERNAME:
    print("=" * 60)
    print("MANUAL GITHUB SETUP REQUIRED")
    print("=" * 60)
    print()
    print("1. Create a GitHub repo at: https://github.com/new")
    print("   Name: settlesense")
    print("   Visibility: Public")
    print("   DO NOT initialize with README")
    print()
    print("2. Run these commands:")
    print()
    print("   git remote add origin https://github.com/YOUR_USERNAME/settlesense.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    print()
    print("3. Then deploy on Streamlit Cloud:")
    print("   https://share.streamlit.io")
    print("   Connect GitHub -> Select repo -> Set app_file: app.py")
    print()
    print("4. Add secrets in Streamlit Cloud:")
    print("   ANTHROPIC_API_KEY = sk-ant-...")
    print("   LLM_PROVIDER = anthropic")
    print()
    sys.exit(0)

import requests

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

# Create repo
repo_name = "settlesense"
r = requests.post("https://api.github.com/user/repos", headers=headers, json={
    "name": repo_name,
    "description": "AI Finance Controller — Razorpay AI Buildathon Track 04",
    "public": True,
    "has_issues": True,
    "has_wiki": False,
})

if r.status_code in (201, 422):  # 422 = already exists
    if r.status_code == 422:
        print(f"Repo already exists: https://github.com/{GITHUB_USERNAME}/{repo_name}")
    else:
        print(f"Repo created: {r.json()['html_url']}")
else:
    print(f"Error creating repo: {r.status_code} {r.text}")
    sys.exit(1)

remote_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{repo_name}.git"

# Add remote and push
subprocess.run(["git", "remote", "remove", "origin"], cwd=project_root, capture_output=True)
subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=project_root, check=True)
subprocess.run(["git", "branch", "-M", "main"], cwd=project_root, check=True)
result = subprocess.run(["git", "push", "-u", "origin", "main", "--force"],
                       cwd=project_root, capture_output=True, text=True)

if result.returncode == 0:
    print(f"Pushed to GitHub: https://github.com/{GITHUB_USERNAME}/{repo_name}")
    print()
    print("=" * 60)
    print("NEXT: Deploy on Streamlit Cloud")
    print("=" * 60)
    print(f"1. Visit: https://share.streamlit.io")
    print(f"2. 'New app' -> {GITHUB_USERNAME}/{repo_name} -> branch: main")
    print(f"3. App file: app.py")
    print(f"4. Add secrets: ANTHROPIC_API_KEY, LLM_PROVIDER, etc.")
    print(f"")
    print(f"Your app URL will be:")
    print(f"  https://{GITHUB_USERNAME}-settlesense-app-py-XXXX.streamlit.app")
else:
    print("Push failed:", result.stderr)
