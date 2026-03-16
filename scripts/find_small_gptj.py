#!/usr/bin/env python3
"""Find smaller GPT-J variants."""

from huggingface_hub import HfApi

api = HfApi()

# List models with 'gpt-j' in modelId
models = api.list_models(search="gpt-j", limit=50)

print("GPT-J models found:")
small_models = []
for model in models:
    print(f"  {model.modelId} - downloads: {model.downloads}")
    # Check if size in name
    lower_id = model.modelId.lower()
    if "125m" in lower_id or "350m" in lower_id or "1b" in lower_id:
        small_models.append(model.modelId)
        print("    -> Small variant candidate")

if small_models:
    print("\nSmall GPT-J candidates:")
    for model_id in small_models:
        print(f"  {model_id}")
else:
    print("\nNo small GPT-J variants found. Try searching for 'gpt-j-125m' manually.")
