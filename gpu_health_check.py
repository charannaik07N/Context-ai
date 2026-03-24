import importlib.util
import os
import platform
import socket
import subprocess
from pathlib import Path

from dotenv import load_dotenv


def _safe_bool(value: str | None) -> bool:
    return (value or "").strip().lower() == "true"


def _run(cmd: list[str], timeout: int = 8) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except Exception as exc:
        return 1, "", str(exc)


def _check_dgx_ssh(host: str, port: str, user: str) -> tuple[bool, str]:
    gpu_password = os.getenv("GPU_SSH_PASSWORD", "")
    gpu_host_key = os.getenv("GPU_SSH_HOST_KEY", "")
    code, out, err = _run(
        [
            "ssh",
            "-p",
            str(port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            f"{user}@{host}",
            "nvidia-smi -L",
        ],
        timeout=10,
    )
    if code == 0 and out:
        return True, out.splitlines()[0]

    # Fallback for password-based SSH accounts on Windows when key auth is not configured.
    if gpu_password and platform.system().lower() == "windows":
        plink_cmd = [
            "plink",
            "-batch",
            "-P",
            str(port),
        ]
        if gpu_host_key:
            plink_cmd.extend(["-hostkey", gpu_host_key])
        plink_cmd.extend(
            [
                "-pw",
                gpu_password,
                f"{user}@{host}",
                "nvidia-smi -L",
            ]
        )
        plink_code, plink_out, plink_err = _run(
            plink_cmd,
            timeout=10,
        )
        if plink_code == 0 and plink_out:
            return True, plink_out.splitlines()[0]
        return False, plink_err or plink_out or err or "DGX unreachable"

    return False, err or "DGX unreachable"


def main() -> int:
    env_path = Path(".env")
    load_dotenv(dotenv_path=env_path if env_path.exists() else None, override=True)

    gpu_enabled = _safe_bool(os.getenv("GPU_ENABLED"))
    gpu_host = os.getenv("GPU_HOST", "")
    gpu_port = os.getenv("GPU_SSH_PORT", "9876")
    gpu_user = os.getenv("GPU_SSH_USER", "")
    gpu_id = os.getenv("GPU_DEVICE_ID", "0")

    print("=== Contexta GPU Health Check ===")
    print(f"os={platform.system()} {platform.release()}")
    print(f"hostname={socket.gethostname()}")
    print(f"GPU_ENABLED={gpu_enabled}")
    print(f"GPU_DEVICE_ID={gpu_id}")

    nvidia_code, nvidia_out, nvidia_err = _run(["nvidia-smi", "-L"], timeout=6)
    local_gpu_ok = nvidia_code == 0 and bool(nvidia_out)
    print(f"local_nvidia_smi={'ok' if local_gpu_ok else 'missing_or_unavailable'}")
    if local_gpu_ok:
        print(f"local_gpu={nvidia_out.splitlines()[0]}")
    elif nvidia_err:
        print(f"local_nvidia_error={nvidia_err}")

    torch_available = bool(importlib.util.find_spec("torch"))
    print(f"torch_installed={torch_available}")

    torch_cuda_ok = False
    torch_version = "not-installed"
    if torch_available:
        try:
            import torch  # type: ignore
            torch_version = getattr(torch, "__version__", "unknown")
            torch_cuda_ok = bool(torch.cuda.is_available())
            print(f"torch_version={torch_version}")
            print(f"torch_cuda_available={torch_cuda_ok}")
            print(f"torch_cuda_devices={torch.cuda.device_count()}")
            if torch_cuda_ok:
                print(f"torch_current_device={torch.cuda.current_device()}")
        except Exception as exc:
            print(f"torch_runtime_error={exc}")

    dgx_ok = False
    if gpu_enabled and gpu_host and gpu_user:
        dgx_ok, dgx_info = _check_dgx_ssh(gpu_host, gpu_port, gpu_user)
        print(f"dgx_ssh={'ok' if dgx_ok else 'failed'}")
        print(f"dgx_info={dgx_info}")
    else:
        print("dgx_ssh=skipped")

    effective_gpu = local_gpu_ok and torch_cuda_ok
    print(f"effective_runtime_device={'cuda' if effective_gpu else 'cpu'}")

    if effective_gpu:
        print("RESULT: GPU is ready for local runtime acceleration.")
        return 0

    if dgx_ok:
        print("RESULT: Remote DGX is reachable, but local runtime is still CPU.")
        return 2

    print("RESULT: Project will run on CPU in current setup.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
