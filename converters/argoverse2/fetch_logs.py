#!/usr/bin/env python3
"""Download AV2 Sensor logs WITHOUT camera or lidar data.

Pulls only what is needed for world-frame trajectory ground truth:
  annotations.feather, city_SE3_egovehicle.feather, calibration/*, map/*

Anonymous HTTPS against the public argoverse S3 bucket -- no AWS creds,
no s5cmd. ~2 MB per log.

Usage:
  fetch_logs.py --split val --limit 1
  fetch_logs.py --split val --logs <log_id> <log_id>
  fetch_logs.py --split val --all
  fetch_logs.py --split val --list-only
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
import os

BUCKET = "https://argoverse.s3.amazonaws.com"
ROOT = "datasets/av2"
DEST = Path(os.environ.get("TRQS_RAW_DIR", Path(__file__).resolve().parents[2] / "datasets" / "raw")) / "argoverse2"

# Everything under these prefixes is imagery / point clouds -- never fetched.
SKIP_PREFIXES = ("sensors/",)


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _list(prefix: str, delimiter: str = "") -> tuple[list[tuple[str, int]], list[str]]:
    """Return ([(key, size)], [common_prefix]) for an S3 prefix, paginated."""
    keys: list[tuple[str, int]] = []
    prefixes: list[str] = []
    token = ""
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        body = _get(f"{BUCKET}/?{urllib.parse.urlencode(q)}").decode()
        keys += [
            (k, int(s))
            for k, s in re.findall(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", body, re.S)
        ]
        prefixes += re.findall(r"<CommonPrefixes><Prefix>(.*?)</Prefix></CommonPrefixes>", body)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", body)
        if not m:
            return keys, prefixes
        token = m.group(1)


def list_logs(dataset: str, split: str) -> list[str]:
    _, prefixes = _list(f"{ROOT}/{dataset}/{split}/", delimiter="/")
    return sorted(p.rstrip("/").split("/")[-1] for p in prefixes)


def log_files(dataset: str, split: str, log_id: str) -> list[tuple[str, int]]:
    """All keys for a log except sensors/ (cameras + lidar)."""
    base = f"{ROOT}/{dataset}/{split}/{log_id}/"
    out = []
    # Listing per-subdir avoids paging through thousands of image keys.
    for sub in ("", "calibration/", "map/"):
        keys, _ = _list(base + sub, delimiter="/" if sub == "" else "")
        for k, sz in keys:
            rel = k[len(base):]
            if rel and not rel.startswith(SKIP_PREFIXES):
                out.append((k, sz))
    return sorted(set(out))


def fetch_log(dataset: str, split: str, log_id: str, force: bool = False) -> tuple[int, int]:
    base = f"{ROOT}/{dataset}/{split}/{log_id}/"
    dest_log = DEST / dataset / split / log_id
    files = log_files(dataset, split, log_id)
    n_new = n_bytes = 0
    for key, size in files:
        out = dest_log / key[len(base):]
        if out.exists() and out.stat().st_size == size and not force:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        tmp.write_bytes(_get(f"{BUCKET}/{urllib.parse.quote(key)}"))
        tmp.rename(out)
        n_new += 1
        n_bytes += size
    return n_new, n_bytes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sensor", choices=["sensor", "lidar", "tbv"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--logs", nargs="*", help="explicit log ids")
    ap.add_argument("--limit", type=int, help="first N logs of the split")
    ap.add_argument("--all", action="store_true", help="every log in the split")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.logs:
        logs = a.logs
    else:
        all_logs = list_logs(a.dataset, a.split)
        print(f"{len(all_logs)} logs in {a.dataset}/{a.split}", file=sys.stderr)
        if a.list_only:
            print("\n".join(all_logs))
            return 0
        if a.all:
            logs = all_logs
        elif a.limit:
            logs = all_logs[: a.limit]
        else:
            ap.error("pass --logs, --limit, --all, or --list-only")

    total_files = total_bytes = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_log, a.dataset, a.split, lg, a.force): lg for lg in logs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            lg = futs[fut]
            try:
                n, b = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(logs)}] FAIL {lg}: {e}", file=sys.stderr)
                continue
            total_files += n
            total_bytes += b
            print(f"[{i}/{len(logs)}] {lg}  {n} new files  {b/1e6:.2f} MB", file=sys.stderr)

    print(f"\ndone: {total_files} files, {total_bytes/1e6:.1f} MB -> "
          f"{DEST/a.dataset/a.split}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
