import hashlib
import hmac
import json
import tempfile
from pathlib import Path

INTEGRITY_MANIFEST_NAME = "integrity_manifest.json"


def _sha256_file(path: Path) -> str:
    """Compute sha256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_integrity_key(integrity_key: str) -> str:
    """Require a signing key so FAISS pickle content can be integrity-checked."""
    key = (integrity_key or "").strip()
    if not key:
        raise RuntimeError(
            "INDEX_INTEGRITY_KEY is not set. Refusing unsafe FAISS deserialization. "
            "Set INDEX_INTEGRITY_KEY in .env and rebuild/re-upload the index."
        )
    return key


def _sign_payload(payload: dict, key: str) -> str:
    """Return HMAC signature for canonical JSON payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hmac.new(key.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()


def get_manifest_path(db_faiss_path: str) -> Path:
    """Return full path to the signed integrity manifest for a FAISS store."""
    return Path(db_faiss_path) / INTEGRITY_MANIFEST_NAME


def write_faiss_integrity_manifest(db_faiss_path: str, integrity_key: str) -> None:
    """Persist signed hashes for FAISS artifacts so tampering is detected before loading."""
    key = _require_integrity_key(integrity_key)
    db_path = Path(db_faiss_path)
    index_faiss = db_path / "index.faiss"
    index_pkl = db_path / "index.pkl"

    if not index_faiss.exists() or not index_pkl.exists():
        raise RuntimeError("Cannot create integrity manifest: FAISS index files are missing.")

    payload = {
        "index_faiss_sha256": _sha256_file(index_faiss),
        "index_pkl_sha256": _sha256_file(index_pkl),
    }
    manifest = {
        "version": 1,
        "algorithm": "sha256+hmac-sha256",
        **payload,
        "signature": _sign_payload(payload, key),
    }

    manifest_path = get_manifest_path(db_faiss_path)
    payload = json.dumps(manifest, ensure_ascii=True, indent=2)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(manifest_path.parent),
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    tmp_path.replace(manifest_path)


def verify_faiss_integrity(db_faiss_path: str, integrity_key: str) -> None:
    """Verify signed FAISS artifact hashes before dangerous deserialization."""
    key = _require_integrity_key(integrity_key)
    db_path = Path(db_faiss_path)
    index_faiss = db_path / "index.faiss"
    index_pkl = db_path / "index.pkl"
    manifest_path = get_manifest_path(db_faiss_path)

    if not index_faiss.exists() or not index_pkl.exists():
        raise FileNotFoundError("FAISS index files are missing.")
    if not manifest_path.exists():
        raise RuntimeError(
            "FAISS integrity manifest is missing. Refusing unsafe deserialization. "
            "Rebuild or re-upload documents to regenerate a signed index."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_payload = {
        "index_faiss_sha256": str(manifest.get("index_faiss_sha256", "")),
        "index_pkl_sha256": str(manifest.get("index_pkl_sha256", "")),
    }
    expected_sig = str(manifest.get("signature", ""))
    actual_sig = _sign_payload(expected_payload, key)

    if not expected_sig or not hmac.compare_digest(expected_sig, actual_sig):
        raise RuntimeError("FAISS integrity signature mismatch. Possible tampering detected.")

    actual_faiss = _sha256_file(index_faiss)
    actual_pkl = _sha256_file(index_pkl)

    if not hmac.compare_digest(actual_faiss, expected_payload["index_faiss_sha256"]):
        raise RuntimeError("FAISS index.faiss hash mismatch. Possible tampering detected.")
    if not hmac.compare_digest(actual_pkl, expected_payload["index_pkl_sha256"]):
        raise RuntimeError("FAISS index.pkl hash mismatch. Possible tampering detected.")
