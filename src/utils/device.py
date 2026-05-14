from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_torch_device(device=None):
    if device is None:
        return None

    device_name = str(device).strip().lower()
    if device_name in {"", "auto"}:
        return None

    needs_cuda = device_name == "cuda" or device_name.startswith("cuda:") or device_name.isdigit()
    if not needs_cuda:
        return device

    try:
        import torch
    except Exception as exc:
        logger.warning(f"Torch is unavailable; falling back to CPU device: {exc}")
        return "cpu"

    if torch.cuda.is_available():
        return device

    logger.warning("CUDA device requested but this Torch install cannot see CUDA; falling back to CPU.")
    return "cpu"
