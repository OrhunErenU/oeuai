"""GPU/cihaz secimi yardimci fonksiyonlari."""

import torch


def get_device(device_str: str = "cuda:0") -> torch.device:
    """Belirtilen cihazi dondur. CUDA yoksa CPU'ya duser."""
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_gpu_info():
    """GPU bilgilerini yazdir."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")
    else:
        print("  CUDA kulanilamiyor, CPU kullanilacak")
