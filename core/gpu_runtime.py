import os


def _is_true(value: str | None) -> bool:
    return (value or "").strip().lower() == "true"


def configure_gpu_environment() -> dict[str, str]:
    """Apply runtime GPU env defaults before model libraries are imported."""
    updates: dict[str, str] = {}

    if not _is_true(os.getenv("GPU_ENABLED")):
        return updates

    gpu_device_id = (os.getenv("GPU_DEVICE_ID") or "0").strip() or "0"

    if not os.getenv("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device_id
        updates["CUDA_VISIBLE_DEVICES"] = gpu_device_id

    if not os.getenv("CUDA_DEVICE_ORDER"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        updates["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Keep runtime defaults explicit for embedding/reranker device auto-resolution.
    if not os.getenv("EMBEDDING_DEVICE"):
        os.environ["EMBEDDING_DEVICE"] = "auto"
        updates["EMBEDDING_DEVICE"] = "auto"

    if not os.getenv("RERANKER_DEVICE"):
        os.environ["RERANKER_DEVICE"] = "auto"
        updates["RERANKER_DEVICE"] = "auto"

    return updates
