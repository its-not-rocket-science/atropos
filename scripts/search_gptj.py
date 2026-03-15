#!/usr/bin/env python3
"""Search for small GPT-J models."""

from huggingface_hub import HfApi, ModelFilter

api = HfApi()
filter = ModelFilter(
    task="text-generation",
    library="transformers",
    model_name="gpt-j",
)
models = api.list_models(filter=filter, limit=50)
for model in models:
    print(f"{model.modelId} - downloads: {model.downloads}")
    # Check if size in name
    if "125m" in model.modelId.lower() or "350m" in model.modelId.lower():
        print(f"  -> Possible small variant")
        # Try to load config
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(model.modelId)
            print(f"     config type: {config.model_type}, n_layer: {config.n_layer}")
        except:
            pass