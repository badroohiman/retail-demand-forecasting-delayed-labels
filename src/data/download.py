import argparse 
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

Default_dataset = "competitions/m5-forecasting-accuracy"
Default_outdir = "data/raw/m5"

def _ensure_dir(path: Path) ->None:
     """Create a directory (and parents) if it does not already exist.

    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exit_ok=True)

def _kaggle_configured() ->bool:
    """Check whether Kaggle credentials are available for the Kaggle CLI.

    Returns:
        True if `~/.kaggle/kaggle.json` exists; otherwise False.
    """
    kaggle_json = Path.home()
    return kaggle_json.exists()


def _run(cmd: list[str]) -> None: 
    result= subprocess.run(cmd, check=False, capture_output=True, text= True)
    if result.returncode != 0: 
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
    
def downlaod_m5(dataset:str, outdir:Path, force:bool=False) ->Path:
     """
    Downloads the M5 dataset zip from Kaggle into outdir.
    Returns the path to the downloaded zip.
    """
    _ensure_dir(outdir)

    zip_path = outdir /"m5.zip"
    if zip_path.exists() and not force: 
        print(f"[OK] Already unzipped: {target_dir} (use --force to re-unzip)")
        return zip_path
    
    if not _kaggle_configured():
        raise FileNotFoundError(
            "Kaggle credentials not found.\n"
            "Expected: ~/.kaggle/kaggle.json\n\n"
            "Fix:\n"
            "1) Create Kaggle API token (Account -> API -> Create New Token)\n"
            "2) Upload kaggle.json into this workspace\n"
            "3) Run:\n"
            "   mkdir -p ~/.kaggle\n"
            "   mv kaggle.json ~/.kaggle/kaggle.json\n"
            "   chmod 600 ~/.kaggle/kaggle.json\n")

    # Download using Kaggle CLI
    # --path sets download folder, --force overwrites, --quiet reduces logs
    cmd= [
        "kaggle","datasets","download","-d", dataset,--"path", str(outdir)]
    if force:
        cmd.append("--force")
    
    print(f"[RUN] {' '.join(cmd)}")
    _run(cmd)

# Kaggle download output file name can vary; find the newest .zip in outdir
zips=sorted(outdir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
if not zips:
    raise FileNotFoundError(f"No .zip files found in {outdir} after download.")

downloaded_zip = zips[0]
print(f"[OK] Downloaded: {downloaded_zip}")
if downloaded_zip.name != "m5.zip":
    target_zip = outdir / "m5.zip"
    shutil.move(str(downloaded_zip), str(target_zip))
    downloaded_zip = target_zip
    print(f"[INFO] Renamed to: {downloaded_zip}")
    return downloaded_zip

def unzip_m5(zip_path:Path, outdir:Path, force:bool=False) ->Path:
    """
    Unzips the M5 dataset zip into outdir/m5.
    Returns the path to the unzipped directory.
    """
    target_dir = outdir / "unzipped"
    if target_dir.exists() and not force:
        print(f"[OK] Already unzipped: {target_dir} (use --force to re-unzip)")
        return

    if target_dir.exists() and force:
        shutil.rmtree(target_dir)

    _ensure_dir(target_dir)

    print(f"[RUN] Unzipping {zip_path} to {target_dir}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)
    
    # Basic check:list first few files
    print(f"[OK] Unzipped {len(files)} top-level files/folders. Example: {[f.name for f in files[:5]]}")

def main()->None:
    parser = argparse.ArgumentParser(description="Download and unzip the M5 dataset from Kaggle.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=Default_dataset,
        help=f"Kaggle dataset identifier (default: {Default_dataset})"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=Default_outdir,
        help=f"Output directory (default: {Default_outdir})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-unzip even if files already exist."
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    zip_path = downlaod_m5(args.dataset, outdir, force=args.force)
    unzip_m5(zip_path, outdir, force=args.force)

    if __name__ == "__main__":
        main()

        