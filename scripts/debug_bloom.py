#!/usr/bin/env python3
import sys
sys.path.insert(0, 'external/wanda')
sys.path.insert(0, 'scripts')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Loading model...')
model = AutoModelForCausalLM.from_pretrained('bigscience/bloom-560m', torch_dtype=torch.float16, low_cpu_mem_usage=True)
print('Model loaded')
tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m')
print('Tokenizer loaded')
from patched_prune import get_model_architecture
arch = get_model_architecture(model)
print(f'Architecture: {arch}')
print('Testing prune_wanda_patched...')
import argparse
args = argparse.Namespace(model='bigscience/bloom-560m', seed=0, nsamples=2, sparsity_ratio=0.1, sparsity_type='unstructured', use_variant=False, save='test', save_model='test', cache_dir=None)
device = torch.device('cpu')
print('Starting pruning...')
from patched_prune import prune_wanda_patched
prune_wanda_patched(args, model, tokenizer, device)
print('Pruning done')