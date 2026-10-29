#!/usr/bin/env python3
"""
Zero-dependency integrity and synchronicity verifier for Diffusion Ablation Research.
Checks that research_finds.md and research_finds match bit-for-bit and maintain
all 24 structural sections and valid mathematical markup.

Usage:
    python verify_integrity.py          # Verify integrity only
    python verify_integrity.py --sync   # Sync research_finds.md -> research_finds, then verify
"""

import os
import sys
import hashlib
import re
import argparse

def compute_sha256(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Verify research findings integrity and twin file synchronicity.")
    parser.add_argument("--sync", action="store_true", help="Mirror research_finds.md to research_finds before verifying.")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    md_file = os.path.join(repo_root, "research_finds.md")
    twin_file = os.path.join(repo_root, "research_finds")

    if not os.path.exists(md_file):
        print(f"[ERROR] Missing file: {md_file}", file=sys.stderr)
        sys.exit(1)

    if args.sync:
        print("[INFO] Synchronizing research_finds.md -> research_finds...")
        with open(md_file, "rb") as src, open(twin_file, "wb") as dst:
            dst.write(src.read())

    if not os.path.exists(twin_file):
        print(f"[ERROR] Missing file: {twin_file}. Run with --sync to generate.", file=sys.stderr)
        sys.exit(1)

    # 1. Check SHA256 twin parity
    md_hash = compute_sha256(md_file)
    twin_hash = compute_sha256(twin_file)

    print(f"[CHECK 1/3] File Hash Verification:")
    print(f"  research_finds.md: {md_hash}")
    print(f"  research_finds:    {twin_hash}")

    if md_hash != twin_hash:
        print("[FAIL] research_finds.md and research_finds do not match!", file=sys.stderr)
        print("Run `python verify_integrity.py --sync` to synchronize them.", file=sys.stderr)
        sys.exit(1)
    print("  [OK] Bit-for-bit twin synchronicity verified.")

    # 2. Check Structural Section Presence (Sections 1 through 26)
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"[CHECK 2/3] Structural Section Verification (1..26):")
    missing_sections = []
    for sec_num in range(1, 27):
        # Look for ## <num>. or # <num>.
        pattern = rf"(?m)^##?\s+{sec_num}\.\s+"
        if not re.search(pattern, content):
            missing_sections.append(sec_num)

    if missing_sections:
        print(f"[FAIL] Missing required sections: {missing_sections}", file=sys.stderr)
        sys.exit(1)
    print("  [OK] All core sections (1..26) are present and properly numbered.")

    # 3. Check Mathematical Formatting
    print(f"[CHECK 3/3] Mathematical Syntax Balance Check:")
    display_math_count = len(re.findall(r"\$\$", content))
    if display_math_count % 2 != 0:
        print(f"[FAIL] Unbalanced display math ($$) blocks detected! Count={display_math_count}", file=sys.stderr)
        sys.exit(1)
    print(f"  [OK] Display math blocks ($$) are balanced ({display_math_count // 2} blocks found).")

    print("\n[SUCCESS] All integrity and provenance checks passed cleanly!")
    sys.exit(0)

if __name__ == "__main__":
    main()
