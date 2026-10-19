"""Run the real two-stage launcher for one or two updates per stage."""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['kda', 'gdn', 'gla'], default='kda')
    parser.add_argument('--method', choices=['bld', 'vanilla', 'both'], default='both')
    parser.add_argument('--steps', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('The smoke test requires an allocated NVIDIA GPU')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ('datasets', 'modules', 'triton', 'inductor', 'huggingface'):
        (root / 'cache' / name).mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name) for name in [
        'transformers', 'accelerate', 'datasets', 'flash-linear-attention',
        'flash-attn', 'causal-conv1d', 'triton', 'torchdata',
    ]}
    report = dict(python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(), versions=versions, updates_per_stage=args.steps,
        model_family=args.model, sequence_length=128, methods={})
    tokenizer_dir = root / 'tokenizer'
    vocab = {'<pad>': 0, '<eos>': 1, '<unk>': 2,
             'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
        pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
    tokenizer.save_pretrained(tokenizer_dir)
    data_dirs = []
    for stage in (1, 2):
        directory = root / f'data{stage}'
        directory.mkdir()
        text = 'a b c d e ' * 200 if stage == 1 else 'e d c b a ' * 200
        Dataset.from_dict({'text': [text] * 32}).to_parquet(str(directory / 'data.parquet'))
        data_dirs.append(directory)
    model_slug = {'kda': '180m', 'gdn': 'gdn-400m', 'gla': 'gla-400m'}[args.model]
    schedule = '180m' if args.model == 'kda' else '390m'
    config_root = root / 'configs'
    (config_root / 'model').mkdir(parents=True)
    model = json.loads((ROOT / 'configs/model' / f'{model_slug}.json').read_text())
    model.update(hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                 num_heads=1, num_v_heads=1, vocab_size=128,
                 pad_token_id=0, bos_token_id=None, eos_token_id=1)
    if args.model == 'gla':
        model['num_kv_heads'] = 1
    model['attn'].update(layers=[0], num_heads=2, num_kv_heads=1)
    (config_root / 'model' / f'{model_slug}.json').write_text(json.dumps(model, indent=2))
    training_dir = config_root / 'training' / schedule
    training_dir.mkdir(parents=True)
    for stage in (1, 2):
        config = json.loads((ROOT / 'configs/training' / schedule / f'stage{stage}.json').read_text())
        config.update(max_len=128, n_train_steps=args.steps, n_warmup_steps=0,
                      n_drop_steps=0, save_interval=1, log_interval=1)
        (training_dir / f'stage{stage}.json').write_text(json.dumps(config, indent=2))
    env = dict(os.environ, PYTHON=sys.executable, NPROC='1', SEED='0',
        CONFIG_DIR=str(config_root), TOKENIZER=str(tokenizer_dir),
        DCLM_STAGE1=str(data_dirs[0]), DCLM_STAGE2=str(data_dirs[1]),
        OUTPUT_DIR=str(root / 'training'), HF_DATASETS_CACHE=str(root / 'cache/datasets'),
        HF_MODULES_CACHE=str(root / 'cache/modules'), TOKENIZERS_PARALLELISM='false',
        TRITON_CACHE_DIR=os.environ.get('TRITON_CACHE_DIR', str(ROOT / 'outputs/.kernel-cache/triton')),
        TORCHINDUCTOR_CACHE_DIR=os.environ.get('TORCHINDUCTOR_CACHE_DIR', str(ROOT / 'outputs/.kernel-cache/inductor')),
        HF_HOME=str(root / 'cache/huggingface'),
        PYTHONUNBUFFERED='1',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    log = root / 'training.log'
    for name in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR'):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    with log.open('w') as handle:
        subprocess.run(['bash', str(ROOT / 'scripts/run_main.sh'), model_slug, args.method],
                       cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    text = log.read_text()
    if re.search(r'loss\s+(nan|inf)', text, flags=re.I):
        raise AssertionError('Nonfinite training loss')
    methods = ['vanilla', 'bld'] if args.method == 'both' else [args.method]
    for method in methods:
        paths = [root / 'training/bld' / f'{model_slug}-{method}-seed0-stage{stage}' for stage in (1, 2)]
        for path in paths:
            assert (path / 'model.safetensors').is_file(), path
            assert (path / f'ckpt_{args.steps}/model.safetensors').is_file(), path
            assert (path / f'ckpt_{args.steps}/dl_state_dict_rank_0.bin').is_file(), path
        first = json.loads((paths[0] / 'args.json').read_text())
        second = json.loads((paths[1] / 'args.json').read_text())
        assert first['optimizer'] == 'adamw' and second['optimizer'] == 'muon'
        assert second['init_from'] == str(paths[0] / 'model.safetensors')
        assert second['resume_path'] is None and second['freeze_layers'] is None
        assert first['freeze_layers'] == ([1, 2, 3] if method == 'bld' else None)
        # BLD's early frozen blocks must retain their initialized values exactly.
        if method == 'bld':
            from accelerate.utils import set_seed
            from types import SimpleNamespace
            from modeling import get_model
            previous_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            set_seed(0)
            raw_args = SimpleNamespace(model_name=args.model,
                model_config=str(config_root / 'model' / f'{model_slug}.json'),
                init_from='scratch', grad_ckpt=0, use_cce_loss=0, as_dict=lambda: {})
            initialized = get_model(SimpleNamespace(print=lambda *a: None, device='cpu'), raw_args)
            torch.set_default_dtype(previous_dtype)
            saved = load_file(str(paths[0] / 'model.safetensors'))
            frozen = [key for key in saved if re.match(r'model\.layers\.[123]\.', key)]
            assert frozen and all(torch.equal(saved[key], initialized.state_dict()[key]) for key in frozen)
            assert not torch.equal(saved['model.layers.0.attn.q_proj.weight'],
                                   initialized.state_dict()['model.layers.0.attn.q_proj.weight'])
            final = load_file(str(paths[1] / 'model.safetensors'))
            assert any(not torch.equal(saved[key], final[key]) for key in frozen)
        loaded = AutoModelForCausalLM.from_pretrained(paths[1], trust_remote_code=True,
            torch_dtype=torch.bfloat16).cuda().eval()
        with torch.no_grad():
            out = loaded(input_ids=torch.tensor([[3, 4, 5, 6]], device='cuda'), use_cache=False)
        assert torch.isfinite(out.logits).all()
        del loaded
        torch.cuda.empty_cache()
        report['methods'][method] = dict(stage1_updates=args.steps, stage2_updates=args.steps,
            optimizer_handoff='adamw_to_muon', export_reload='passed',
            frozen_blocks='passed' if method == 'bld' else 'not_applicable')
    report['finite_losses'] = re.findall(rf'\[(\d+)/{args.steps}\] loss ([0-9.eE+\-]+)', text)
    assert len(report['finite_losses']) == 2 * args.steps * len(methods)
    report['gradient_norms'] = [float(value) for value in re.findall(r'grad_norm (\S+)', text)]
    assert len(report['gradient_norms']) == 2 * args.steps * len(methods)
    assert all(math.isfinite(value) and value > 0 for value in report['gradient_norms'])
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
