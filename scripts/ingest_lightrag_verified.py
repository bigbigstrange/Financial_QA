"""Resume document ingestion using server completion, not HTTP acceptance.

One document per request; delimiter-only chunking preserves KB boundaries.
The manifest is bound to server workspace, source version and content hash.
No delete/clear operations are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.system.lightrag_client import LightRAGClient

DELIMITER = "\n\n<FINANCE_CHUNK_BOUNDARY>\n\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kb-dir", type=Path, default=ROOT / "output/kb_v3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()
    client = LightRAGClient()
    health = client.health()
    workspace = health.get("configuration", {}).get("workspace") or "default"
    manifest = args.kb_dir / ("kg_verified_" + workspace + ".jsonl")
    records = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            records[row["identity"]] = row
    docs = {}
    for line in (args.kb_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        chunk = json.loads(line)
        docs.setdefault((chunk["source_doc"], chunk["document_version"]), []).append(chunk)
    if args.limit:
        docs = dict(list(docs.items())[:args.limit])
    failures = 0
    for number, ((source, version), chunks) in enumerate(docs.items(), 1):
        content = DELIMITER.join(c["content"].strip() for c in chunks)
        digest = hashlib.sha256(content.encode()).hexdigest()
        identity = f"{workspace}:{version}:{digest}"
        previous = records.get(identity, {})
        if previous.get("status") == "processed" and previous.get("track_id"):
            current = client._request("GET", f"/documents/track_status/{previous['track_id']}")
            entries = current.get("documents") or []
            if entries and all(d.get("status") == "processed" for d in entries):
                print(f"[{number}/{len(docs)}] verified skip: {source}", flush=True)
                continue
            previous = {}
        row = {"identity": identity, "source_doc": source, "document_version": version,
               "sha256": digest, "expected_chunks": len(chunks), "workspace": workspace}
        try:
            track = previous.get("track_id")
            if not track:
                response = client.insert_text(content, file_source=source, extra={
                    "chunking": {"strategy": "fixed_token", "params": {
                        "split_by_character": DELIMITER, "split_by_character_only": True,
                        "chunk_token_size": 4096, "chunk_overlap_token_size": 0,
                    }},
                })
                track = response.get("track_id")
                if not track:
                    raise RuntimeError(f"No tracking ID: {response}")
            row.update(track_id=track, status="submitted")
            with manifest.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            deadline = time.monotonic() + args.timeout
            while True:
                result = client._request("GET", f"/documents/track_status/{track}")
                entries = result.get("documents") or []
                states = [d.get("status") for d in entries]
                if entries and all(s == "processed" for s in states):
                    row.update(status="processed", server_documents=entries)
                    break
                if "failed" in states:
                    raise RuntimeError(str(entries)[:1200])
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Still pending: {states}")
                time.sleep(5)
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            failures += 1
        row["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{number}/{len(docs)}] {row['status']}: {source}", flush=True)
        if row.get("error"):
            print(row["error"], flush=True)
            # Surface service/configuration failures before queuing more paid work.
            break
    raise SystemExit(2 if failures else 0)


if __name__ == "__main__":
    main()
