import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

DEFAULT_DATASET = "m5-forecasting-accuracy"  # competition name
DEFAULT_OUTDIR = "data/raw/m5"


def _ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def _kaggle_configured() -> bool:
    """True if ~/.kaggle/kaggle.json exists."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )


def download_m5(dataset: str, outdir: Path, force: bool = False) -> Path:
    """Download the M5 competition zip into outdir and return the zip path."""
    _ensure_dir(outdir)

    zip_path = outdir / "m5.zip"
    if zip_path.exists() and not force:
        print(f"[OK] Zip already exists: {zip_path} (use --force to re-download)")
        return zip_path

    if not _kaggle_configured():
        raise FileNotFoundError(
            "Kaggle credentials not found.\n"
            "Expected: ~/.kaggle/kaggle.json\n\n"
            "Fix:\n"
            "1) Kaggle -> Account -> API -> Create Legacy API Key\n"
            "2) Upload kaggle.json into this workspace\n"
            "3) Run:\n"
            "   mkdir -p ~/.kaggle\n"
            "   mv kaggle.json ~/.kaggle/kaggle.json\n"
            "   chmod 600 ~/.kaggle/kaggle.json\n"
        )

    # M5 is a COMPETITION
    cmd = ["kaggle", "competitions", "download", "-c", dataset, "--path", str(outdir)]
    if force:
        cmd.append("--force")

    print(f"[RUN] {' '.join(cmd)}")
    _run(cmd)

    # Find newest zip and normalize to m5.zip
    zips = sorted(outdir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No .zip files found in {outdir} after download.")

    downloaded_zip = zips[0]
    print(f"[OK] Downloaded: {downloaded_zip}")

    if downloaded_zip.name != "m5.zip":
        shutil.move(str(downloaded_zip), str(zip_path))
        downloaded_zip = zip_path
        print(f"[INFO] Renamed to: {downloaded_zip}")

    return downloaded_zip


def unzip_m5(zip_path: Path, outdir: Path, force: bool = False) -> Path:
    """Unzip the zip into outdir/unzipped and return that folder path."""
    target_dir = outdir / "unzipped"

    if target_dir.exists() and not force:
        print(f"[OK] Already unzipped: {target_dir} (use --force to re-unzip)")
        return target_dir

    if target_dir.exists() and force:
        shutil.rmtree(target_dir)

    _ensure_dir(target_dir)

    print(f"[RUN] Unzipping {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    files = list(target_dir.glob("*"))
    print(
        f"[OK] Unzipped {len(files)} top-level files. Example: {[f.name for f in files[:5]]}"
    )

    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and unzip the M5 dataset from Kaggle."
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    zip_path = download_m5(args.dataset, outdir, force=args.force)
    unzip_dir = unzip_m5(zip_path, outdir, force=args.force)

    print("\n[READY]")
    print("Zip:", zip_path)
    print("Unzipped:", unzip_dir)


if __name__ == "__main__":
    main()
