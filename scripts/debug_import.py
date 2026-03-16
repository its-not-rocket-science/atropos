import sys
sys.stdout.flush()
print("1. Importing torch")
import torch
print(f"torch imported, CUDA: {torch.cuda.is_available()}")
sys.stdout.flush()
print("2. Importing lib.prune")
try:
    from lib.prune import find_layers
    print("lib.prune imported")
except Exception as e:
    print(f"Error importing lib.prune: {e}")
sys.stdout.flush()
print("3. Importing lib.prune_opt")
try:
    from lib.prune_opt import check_sparsity as check_sparsity_opt
    print("lib.prune_opt imported")
except Exception as e:
    print(f"Error importing lib.prune_opt: {e}")
sys.stdout.flush()
print("4. Importing patched_prune")
try:
    import patched_prune
    print("patched_prune imported")
except Exception as e:
    print(f"Error importing patched_prune: {e}")
    import traceback
    traceback.print_exc()
sys.stdout.flush()
print("Done")