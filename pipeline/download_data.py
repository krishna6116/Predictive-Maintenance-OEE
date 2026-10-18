"""Download NASA C-MAPSS into data/raw/ and check every file against data/checksums.sha256.

Run:  python -m pipeline.download_data
Files already present with the right checksum are left alone. Exits 1 if any file does not match,
so results can only ever be reproduced on byte-identical data.
"""
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
RAW = Path("data/raw")
CHECKSUMS = Path("data/checksums.sha256")


def expected() -> dict[str, str]:
    out = {}
    for line in CHECKSUMS.read_text().splitlines():
        if line.strip():
            digest, name = line.split()
            out[name] = digest
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    want = expected()
    RAW.mkdir(parents=True, exist_ok=True)
    missing = [n for n, d in want.items() if not (RAW / n).exists() or sha256(RAW / n) != d]
    if missing:
        print(f"Downloading C-MAPSS ({len(missing)} files missing or different) ...")
        with urllib.request.urlopen(URL, timeout=120) as resp:
            outer = zipfile.ZipFile(io.BytesIO(resp.read()))
        inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
        for member in inner.namelist():
            name = Path(member).name
            if name in want:  # only the files we pin; never write paths taken from the archive
                (RAW / name).write_bytes(inner.read(member))
    bad = [n for n, d in want.items() if not (RAW / n).exists() or sha256(RAW / n) != d]
    for n in want:
        print(("OK   " if n not in bad else "FAIL ") + n)
    print("All files match data/checksums.sha256." if not bad else f"{len(bad)} file(s) do not match.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
