from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_pipeline import NAMESPACED_STORE_ROOT, append_document_to_vector_store


def discover_namespaces() -> list[str]:
    root = Path(NAMESPACED_STORE_ROOT)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed limitations QA knowledge into vector stores.")
    parser.add_argument(
        "--seed-file",
        default=str(Path(__file__).with_name("limitations_qa_seed.txt")),
        help="Path to the seed .txt document.",
    )
    parser.add_argument(
        "--include-namespace",
        action="append",
        default=[],
        help="Namespace to include in addition to discovered namespaces. Can be repeated.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size for append operation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_file = Path(args.seed_file).resolve()
    if not seed_file.exists():
        print(json.dumps({"ok": False, "error": f"Seed file not found: {seed_file}"}, ensure_ascii=True))
        return 1

    discovered = discover_namespaces()
    targets = set(discovered)
    targets.update(ns.strip() for ns in args.include_namespace if ns and ns.strip())
    targets.add("default")

    results: dict[str, dict] = {}
    for ns in sorted(targets):
        namespace_arg = None if ns == "default" else ns
        try:
            stats = append_document_to_vector_store(
                file_path=str(seed_file),
                batch_size=max(1, int(args.batch_size)),
                namespace=namespace_arg,
            )
            results[ns] = {"ok": True, "stats": stats}
        except Exception as exc:  # pragma: no cover - runtime safety
            results[ns] = {"ok": False, "error": str(exc)}

    print(json.dumps({"ok": True, "seed_file": str(seed_file), "targets": sorted(targets), "results": results}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
