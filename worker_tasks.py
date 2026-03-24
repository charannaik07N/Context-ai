import gc
from rag_pipeline import append_document_to_vector_store, compute_metrics


def process_single_upload_task(namespace: str, filename: str, file_path: str, batch_size: int = 32) -> dict:
    """External worker task: ingest one file and return API-shaped payload."""
    try:
        ingest_stats = append_document_to_vector_store(
            file_path=file_path,
            batch_size=batch_size,
            namespace=namespace,
        )
        return {
            "message": "Document processed successfully.",
            "filename": filename,
            "namespace": namespace,
            "ingestion": ingest_stats,
        }
    finally:
        gc.collect()


def process_batch_upload_task(namespace: str, file_records: list[dict], batch_size: int = 32) -> dict:
    """External worker task: ingest many files and return API-shaped payload."""
    processed = []
    failed = []

    for record in file_records:
        try:
            ingest_stats = append_document_to_vector_store(
                file_path=record["file_path"],
                batch_size=batch_size,
                namespace=namespace,
            )
            processed.append(
                {
                    "filename": record["filename"],
                    "ingestion": ingest_stats,
                }
            )
        except Exception as e:
            failed.append(
                {
                    "filename": record.get("filename", "unknown"),
                    "error": (
                        "Failed to process the document. Ensure it is a valid supported file "
                        f"(.pdf, .docx, .txt, .html, .htm). ({str(e)})"
                    ),
                }
            )
        finally:
            gc.collect()

    if not processed:
        raise RuntimeError(
            "No files were processed successfully. "
            f"failed={failed}"
        )

    return {
        "message": "Batch upload completed.",
        "namespace": namespace,
        "processed_count": len(processed),
        "failed_count": len(failed),
        "processed": processed,
        "failed": failed,
    }


def compute_metrics_task(namespace: str) -> dict:
    """External worker task: compute metrics for a namespace."""
    result = compute_metrics(namespace=namespace)
    if result is None:
        raise RuntimeError("Failed to compute metrics.")
    return result
