"""Download script for ECG datasets from PhysioNet.

Supports two download methods:
- curl (default): recursive download with resume support, requires PhysioNet credentials.
- wfdb: uses the wfdb library, no credentials needed but no resume support.

Usage:
    python scripts/download_data.py --user <physionet_user> --password <physionet_pass>
    python scripts/download_data.py --method wfdb
"""
import argparse
import subprocess
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
  # Download PTB-XL with curl (resumable, requires PhysioNet account):
  python scripts/download_data.py --user myuser --password mypass

  # Download without credentials (no resume support):
  python scripts/download_data.py --method wfdb

  # Download MIT-BIH instead:
  python scripts/download_data.py --dataset mitbih --user myuser --password mypass
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
        help="Download method: curl (resumable, needs credentials) or wfdb (default: curl)",
    )
    parser.add_argument("--user", type=str, default=None, help="PhysioNet username")
    parser.add_argument("--password", type=str, default=None, help="PhysioNet password")
    return parser.parse_args()


def download_curl(
    dataset_key: str,
    output_dir: Path,
    user: str | None = None,
    password: str | None = None,
) -> None:
    """Download a PhysioNet dataset using curl with resume support.

    Uses recursive download (-r), timestamping to skip existing files (-N),
    and resume (-C -) to continue interrupted downloads.

    Args:
        dataset_key: Key from the DATASETS dictionary.
        output_dir: Root directory where files will be saved.
        user: PhysioNet username.
        password: PhysioNet password.
    """
    info = DATASETS[dataset_key]
    target_dir = output_dir / dataset_key
    target_dir.mkdir(parents=True, exist_ok=True)

    url = info["url"]

    print(f"Dataset  : {info['description']}")
    print(f"Size     : ~{info['size_gb']} GB")
    print(f"Target   : {target_dir.resolve()}")
    print(f"Method   : curl (resumable)")
    print()
    print("Tip: if the download is interrupted, run the same command again to resume.\n")

    cmd = [
        "curl",
        "--user", f"{user}:{password}",
        "-r", "-",           # resume from where it left off
        "--retry", "10",     # retry up to 10 times on transient errors
        "--retry-delay", "5",
        "--retry-all-errors",
        "-L",                # follow redirects
        "--create-dirs",
        "-O",                # save with original filename
        "--output-dir", str(target_dir),
        "--progress-bar",
        # recursive mirror flags via wget-style glob — PhysioNet supports listing
        url,
    ]

    # For recursive directory download, curl needs a different approach.
    # We use curl to first fetch the file list, then download each file.
    _curl_recursive(url, target_dir, user, password)


def _curl_recursive(url: str, target_dir: Path, user: str | None, password: str | None) -> None:
    """Recursively download all files from a PhysioNet directory via curl.

    Fetches the HTML index to discover files, then downloads each one
    with resume support (-C -).

    Args:
        url: PhysioNet directory URL.
        target_dir: Local directory to save files.
        user: PhysioNet username.
        password: PhysioNet password.
    """
    import re
    import urllib.parse

    # Fetch directory listing
    auth_flags = ["--user", f"{user}:{password}"] if user and password else []
    result = subprocess.run(
        ["curl", "-s", *auth_flags, "-L", url],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error fetching directory listing: {result.stderr}")
        return

    # Parse hrefs from the HTML listing (PhysioNet uses Apache-style listings)
    hrefs = re.findall(r'href="([^"?#]+)"', result.stdout)

    for href in hrefs:
        # Skip parent directory links and absolute URLs
        if href.startswith(("/", "http", "?", "..")) or href == "/":
            continue

        full_url = url.rstrip("/") + "/" + href
        local_path = target_dir / urllib.parse.unquote(href)

        if href.endswith("/"):
            # It's a subdirectory — recurse
            local_path.mkdir(parents=True, exist_ok=True)
            _curl_recursive(full_url, local_path, user, password)
        else:
            # It's a file — download with resume
            if local_path.exists() and local_path.stat().st_size > 0:
                print(f"  skip (exists): {local_path.relative_to(target_dir.parent.parent)}")
                continue

            print(f"  downloading  : {href}")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "curl",
                    *auth_flags,
                    "-L",
                    "--retry", "10",
                    "--retry-delay", "5",
                    "--retry-all-errors",
                    "-C", "-",
                    "-o", str(local_path),
                    "--progress-bar",
                    full_url,
                ],
                check=False,
            )

    print(f"\nDone. Dataset saved to: {target_dir.resolve()}")


def download_wfdb(dataset_key: str, output_dir: Path) -> None:
    """Download a PhysioNet dataset using the wfdb library.

    No credentials required, but does not support resuming interrupted downloads.

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
    print(f"Method   : wfdb (no resume support)")
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
        download_curl(args.dataset, args.output_dir, args.user, args.password)
    else:
        download_wfdb(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
