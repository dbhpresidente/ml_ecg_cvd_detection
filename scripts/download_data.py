"""Download script for ECG datasets from PhysioNet.

Supports two download methods:
- curl (default): parallel download with resume support, no credentials needed
  for open-access datasets like PTB-XL.
- wfdb: uses the wfdb library, sequential, no resume support.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --workers 8
    python scripts/download_data.py --method wfdb
"""
import argparse
import re
import subprocess
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASETS = {
    "ptb-xl": {
        "physionet_name": "ptb-xl",
        "version": "1.0.3",
        "description": "PTB-XL: 21,837 12-lead ECGs, 71 diagnostic classes",
        "size_gb": 6.6,
        "url": "https://physionet.org/files/ptb-xl/1.0.3/",
    },
    "mitbih": {
        "physionet_name": "mitdb",
        "version": "1.0.0",
        "description": "MIT-BIH Arrhythmia: 48 two-lead 30-min recordings",
        "size_gb": 0.1,
        "url": "https://physionet.org/files/mitdb/1.0.0/",
    },
    "cpsc2018": {
        "physionet_name": "cpsc2018",
        "version": "1.0.0",
        "description": "CPSC 2018: 6,877 12-lead ECGs, 9 rhythm classes",
        "size_gb": 1.7,
        "url": "https://physionet.org/files/cpsc2018/1.0.0/",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ECG datasets from PhysioNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_data.py                        # PTB-XL, 4 workers
  python scripts/download_data.py --workers 8            # faster with more connections
  python scripts/download_data.py --dataset mitbih
  python scripts/download_data.py --method wfdb          # fallback, no parallelism
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="ptb-xl",
        help="Dataset to download (default: ptb-xl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save the dataset (default: data/raw)",
    )
    parser.add_argument(
        "--method",
        choices=["curl", "wfdb"],
        default="curl",
        help="Download method (default: curl)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    parser.add_argument("--user", type=str, default=None, help="PhysioNet username (if required)")
    parser.add_argument("--password", type=str, default=None, help="PhysioNet password (if required)")
    return parser.parse_args()


def _fetch_file_list(
    url: str,
    target_dir: Path,
    auth_flags: list[str],
) -> list[tuple[str, Path]]:
    """Recursively fetch the full list of (url, local_path) pairs from a PhysioNet directory.

    Args:
        url: PhysioNet directory URL to crawl.
        target_dir: Local directory corresponding to this URL.
        auth_flags: curl auth flags (empty list if no credentials needed).

    Returns:
        List of (remote_url, local_path) tuples for all files found.
    """
    result = subprocess.run(
        ["curl", "-s", *auth_flags, "-L", url],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  [warn] could not fetch listing for {url}: {result.stderr.strip()}")
        return []

    hrefs = re.findall(r'href="([^"?#]+)"', result.stdout)
    files: list[tuple[str, Path]] = []

    for href in hrefs:
        if href.startswith(("/", "http", "?", "..")) or href == "/":
            continue

        full_url = url.rstrip("/") + "/" + href
        local_path = target_dir / urllib.parse.unquote(href)

        if href.endswith("/"):
            local_path.mkdir(parents=True, exist_ok=True)
            files.extend(_fetch_file_list(full_url, local_path, auth_flags))
        else:
            files.append((full_url, local_path))

    return files


def _download_file(
    url: str,
    local_path: Path,
    auth_flags: list[str],
) -> tuple[str, bool, str]:
    """Download a single file with resume support.

    Args:
        url: Remote file URL.
        local_path: Local destination path.
        auth_flags: curl auth flags.

    Returns:
        Tuple of (filename, success, message).
    """
    name = local_path.name

    if local_path.exists() and local_path.stat().st_size > 0:
        return name, True, "skipped (exists)"

    local_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "curl",
            *auth_flags,
            "-L",
            "--retry", "10",
            "--retry-delay", "5",
            "--retry-all-errors",
            "-C", "-",
            "-s",               # silent — progress handled by caller
            "-o", str(local_path),
            url,
        ],
        check=False,
    )

    if result.returncode != 0:
        return name, False, f"failed (exit {result.returncode})"
    return name, True, "done"


def download_curl(
    dataset_key: str,
    output_dir: Path,
    workers: int = 4,
    user: str | None = None,
    password: str | None = None,
) -> None:
    """Download a PhysioNet dataset using parallel curl workers.

    First crawls the remote directory tree to collect all file URLs, then
    downloads them in parallel. Already-downloaded files are skipped, so
    re-running resumes an interrupted download.

    Args:
        dataset_key: Key from the DATASETS dictionary.
        output_dir: Root directory where files will be saved.
        workers: Number of parallel download threads.
        user: PhysioNet username (optional for open-access datasets).
        password: PhysioNet password (optional for open-access datasets).
    """
    info = DATASETS[dataset_key]
    target_dir = output_dir / dataset_key
    target_dir.mkdir(parents=True, exist_ok=True)

    auth_flags = ["--user", f"{user}:{password}"] if user and password else []

    print(f"Dataset  : {info['description']}")
    print(f"Size     : ~{info['size_gb']} GB")
    print(f"Target   : {target_dir.resolve()}")
    print(f"Workers  : {workers}")
    print()

    print("Step 1/2 — Crawling remote directory tree...")
    all_files = _fetch_file_list(info["url"], target_dir, auth_flags)
    total = len(all_files)
    print(f"          Found {total} files.\n")

    already_done = sum(1 for _, p in all_files if p.exists() and p.stat().st_size > 0)
    pending = total - already_done
    print(f"Step 2/2 — Downloading {pending} files ({already_done} already cached)...")

    completed = 0
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_file, url, path, auth_flags): path.name
            for url, path in all_files
        }
        for future in as_completed(futures):
            name, success, msg = future.result()
            completed += 1
            status = "ok" if success else "!!"
            print(f"  [{completed:>5}/{total}] {status} {name}  ({msg})")
            if not success:
                failed.append(name)

    print(f"\nDone. {total - len(failed)} / {total} files saved to: {target_dir.resolve()}")
    if failed:
        print(f"\n[warn] {len(failed)} files failed — re-run to retry:")
        for f in failed:
            print(f"  {f}")


def download_wfdb(dataset_key: str, output_dir: Path) -> None:
    """Download a PhysioNet dataset using the wfdb library (sequential, no resume).

    Args:
        dataset_key: Key from the DATASETS dictionary.
        output_dir: Root directory where the dataset will be saved.
    """
    import wfdb

    info = DATASETS[dataset_key]
    target_dir = output_dir / dataset_key

    print(f"Dataset  : {info['description']}")
    print(f"Size     : ~{info['size_gb']} GB")
    print(f"Target   : {target_dir.resolve()}")
    print("Method   : wfdb (sequential, no resume support)")
    print()

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Directory {target_dir} already exists and is not empty. Skipping.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(info["physionet_name"], dl_dir=str(target_dir))
    print(f"\nDone. Dataset saved to: {target_dir.resolve()}")


def main() -> None:
    args = parse_args()

    if args.method == "curl":
        download_curl(args.dataset, args.output_dir, args.workers, args.user, args.password)
    else:
        download_wfdb(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
