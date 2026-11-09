"""Inventory every source, excluding Office locks; retain prior subset assets."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "data/documents"
    target = ROOT / "output/kb_full"
    target.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT / "output/kb_v3/file_inventory.json").read_text(encoding="utf-8"))
    known = {r["rel_path"]: r for r in old["files"]}
    included, excluded = [], []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source).as_posix()
        if path.name.startswith("~$") or path.name == ".gitkeep":
            excluded.append({"rel_path": rel, "reason": "office_lock" if path.name.startswith("~$") else "repository_placeholder"})
            continue
        included.append({**known.get(rel, {}), "rel_path": rel,
                         "ext": path.suffix.lower(), "size_kb": round(path.stat().st_size / 1024, 1)})
    payload = {"count": len(included), "files": included, "excluded": excluded}
    (target / "full_inventory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Reuse converted source PDFs without moving or changing the active 91-doc KB.
    converted = target / "converted"
    converted.mkdir(exist_ok=True)
    for path in (ROOT / "output/kb_v3/converted").glob("*.pdf"):
        dest = converted / path.name
        if not dest.exists():
            shutil.copy2(path, dest)
    print(json.dumps({"business_files": len(included), "excluded": excluded}, ensure_ascii=False))


if __name__ == "__main__":
    main()
