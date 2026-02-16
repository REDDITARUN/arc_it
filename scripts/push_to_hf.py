#!/usr/bin/env python3
"""Push a trained ARC-IT checkpoint to Hugging Face Hub.

Usage:
    # Push using .env config:
    python scripts/push_to_hf.py --checkpoint checkpoints/best_stage2.pt

    # Override repo ID:
    python scripts/push_to_hf.py --checkpoint checkpoints/best_stage2.pt --repo-id myuser/arc-it

    # Private repo:
    python scripts/push_to_hf.py --checkpoint checkpoints/best_stage2.pt --private
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from arc_it.utils.hf_upload import upload_checkpoint_to_hf


def main():
    parser = argparse.ArgumentParser(description="Push ARC-IT model to Hugging Face Hub")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--repo-id", default=None, help="HF repo ID (default: from .env HF_REPO_ID)")
    parser.add_argument("--token", default=None, help="HF token (default: from .env HF_TOKEN)")
    parser.add_argument("--message", default="Upload ARC-IT model checkpoint", help="Commit message")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    args = parser.parse_args()

    url = upload_checkpoint_to_hf(
        checkpoint_path=args.checkpoint,
        repo_id=args.repo_id,
        token=args.token,
        commit_message=args.message,
        private=args.private,
    )
    print(f"\nModel available at: {url}")


if __name__ == "__main__":
    main()
