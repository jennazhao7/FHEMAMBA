"""
Upload s4d_adding_best.pt to Hugging Face Hub.

Usage:
  python push_to_hf.py --repo_id <user>/<repo> --token <HF_TOKEN>
  python push_to_hf.py --repo_id <user>/<repo> --token <HF_TOKEN> --private

Requires:
  pip install huggingface_hub
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file


def main():
    parser = argparse.ArgumentParser(description="Upload s4d_adding_best.pt to Hugging Face Hub")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repo: <user>/<repo>")
    parser.add_argument("--token", required=True, help="Hugging Face access token")
    parser.add_argument("--file", default="s4d_adding_best.pt", help="Checkpoint file to upload")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument("--commit_message", default="Add s4d_adding_best.pt", help="Commit message")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path.resolve()}")

    api = HfApi()
    create_repo(
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        exist_ok=True,
        repo_type="model",
    )

    upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=file_path.name,
        repo_id=args.repo_id,
        token=args.token,
        repo_type="model",
        commit_message=args.commit_message,
    )

    print(f"Uploaded {file_path.name} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
