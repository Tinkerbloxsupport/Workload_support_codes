import sys

def check():
    print("=== Python ===")
    print(sys.version)

    print("\n=== PyTorch + CUDA ===")
    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        print("torch NOT installed")

    print("\n=== Required packages ===")
    pkgs = ["transformers", "datasets", "accelerate", "trl", "peft", "bitsandbytes"]
    missing = []
    for pkg in pkgs:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  {pkg}: {ver}")
        except ImportError:
            print(f"  {pkg}: MISSING")
            missing.append(pkg)

    if missing:
        print(f"\nInstall missing:")
        print(f"pip install {' '.join(missing)} --break-system-packages")
    else:
        print("\nAll packages present. Ready.")

check()
