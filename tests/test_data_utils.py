from pathlib import Path
from src.data.download import _ensure_dir

def test_ensure_dir_creates_directory(tmp_path: Path):
    p = tmp_path / "a" / "b"
    _ensure_dir(p)
    assert p.exists()
    assert p.is_dir()
