"""Pretrain BLD or Vanilla; each stage has a fresh optimizer and scheduler."""

import json
from pathlib import Path

import torch
from accelerate.utils import set_seed
from transformers import AutoTokenizer

from arguments import Args
from modeling import get_model
from preparation import (
    freeze_model_layers, get_accelerator, get_dataloaders, load_train_config,
    prepare_optimizers, restore_frozen_parameters,
)
from trainer.trainer import LMTrainer


def main():
    args = Args().parse_args()
    load_train_config(args)
    set_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    accelerator = get_accelerator(args)
    output = Path(args.output_dir, args.proj_name, args.run_name)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=bool(args.resume_path))
        args.save(str(output / 'args.json'))
    accelerator.wait_for_everyone()
    accelerator.print(f'Tokens/update: {accelerator.num_processes * args.grad_accum_steps * args.batch_size * args.max_len:,}')
    model = get_model(accelerator, args)
    freeze = freeze_model_layers(model, args.freeze_layers)
    accelerator.print(f'Frozen blocks: {list(freeze.layer_indices)}; trainable parameters: {freeze.trainable_numel:,}')
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)
    if max(tokenizer.get_vocab().values()) >= model.config.vocab_size:
        raise ValueError('Tokenizer IDs exceed the paper model vocabulary (153,000)')
    if accelerator.is_main_process:
        (output / 'model_config.json').write_text(json.dumps(model.config.to_dict(), indent=2))
    try:
        optimizer, scheduler = prepare_optimizers(model, args)
        loader, _ = get_dataloaders(args, tokenizer, accelerator)
        trainer = LMTrainer(args, output, accelerator, model, optimizer, scheduler, loader)
        trainer.train()
    finally:
        restore_frozen_parameters(freeze)
    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(trainer.model)
    unwrapped = accelerator.unwrap_model(trainer.model)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output)
        unwrapped.config.register_for_auto_class()
        unwrapped.model.register_for_auto_class('AutoModel')
        unwrapped.register_for_auto_class('AutoModelForCausalLM')
        unwrapped.save_pretrained(output, state_dict=state, safe_serialization=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == '__main__':
    main()
