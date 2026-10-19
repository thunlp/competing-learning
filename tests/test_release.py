import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class ReleaseTests(unittest.TestCase):
    def test_paper_parameter_counts(self):
        from modeling.kda import KDAConfig,KDAForCausalLM
        from modeling.gated_deltanet.config import GatedDeltaNetConfig
        from modeling.gated_deltanet.model import GatedDeltaNetForCausalLM
        from modeling.gla import GLAConfig,GLAForCausalLM
        classes={'kda':(KDAConfig,KDAForCausalLM),
                 'gated_deltanet':(GatedDeltaNetConfig,GatedDeltaNetForCausalLM),'gla':(GLAConfig,GLAForCausalLM)}
        expected={'180m':180744996,'250m':253815624,'390m':390873184,'510m':507973264,
                  '750m':754678416,'1b':1014512856,'gdn-400m':400371904,'gla-400m':400580608}
        for name,count in expected.items():
            with self.subTest(model=name):
                path=ROOT/'configs/model'/f'{name}.json'
                config=json.loads(path.read_text())
                self.assertEqual(config['attn']['layers'],list(range(0,config['num_hidden_layers'],4)))
                c,m=classes[config['model_type']]
                with torch.device('meta'):model=m(c.from_json_file(str(path)))
                self.assertEqual(sum(p.numel() for p in model.parameters()),count)
                self.assertIs(model.model.embeddings.weight,model.lm_head.weight)

    def test_complete_block_freezing_and_restore(self):
        from preparation import freeze_model_layers,restore_frozen_parameters
        model=nn.Module();model.model=nn.Module()
        model.model.layers=nn.ModuleList([nn.Sequential(nn.Linear(4,4),nn.LayerNorm(4)) for _ in range(4)])
        model.embedding=nn.Embedding(12,4)
        state=freeze_model_layers(model,[0,2])
        for index,block in enumerate(model.model.layers):
            self.assertTrue(all(p.requires_grad==(index in (1,3)) for p in block.parameters()))
        self.assertTrue(model.embedding.weight.requires_grad)
        restore_frozen_parameters(state)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_retrieval_data_isolation_and_quotas(self):
        from experiments.retrieval import core
        from experiments.retrieval.run import batch_rows
        data=core.build_data(dict(data_seed=20260910,eval_examples=2000,train_examples=50000))
        train=set(core.key_numbers(data['train']['keys']).tolist())
        self.assertFalse(train.intersection(core.key_numbers(data['evaluation']['keys']).tolist()))
        self.assertFalse(train.intersection(core.key_numbers(data['evaluation']['alternate']).tolist()))
        orders=torch.stack([torch.arange(50000),torch.arange(50000)])
        for percentage in range(0,101,10):
            counts=[sum(count for gap,_,count in batch_rows(data,orders,step,percentage) if gap==64) for step in range(1,1501)]
            self.assertEqual(sum(counts),48000*percentage//100)

    def test_paused_adam_moments_survive_unfreezing(self):
        from experiments.retrieval.run import apply_update
        model=nn.Linear(2,2)
        opt=torch.optim.AdamW(model.parameters(),lr=.01)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        for p in model.parameters():p.grad=torch.ones_like(p)
        apply_update(model,opt,sched)
        paused=model.weight.detach().clone()
        moment=opt.state[model.weight]['exp_avg'].clone()
        for p in model.parameters():p.grad=torch.ones_like(p)
        apply_update(model,opt,sched,{'weight'})
        self.assertTrue(torch.equal(model.weight,paused))
        self.assertTrue(torch.equal(opt.state[model.weight]['exp_avg'],moment))
        self.assertEqual(opt.state[model.weight]['step'].item(),1)
        for p in model.parameters():p.grad=torch.ones_like(p)
        apply_update(model,opt,sched)
        self.assertEqual(opt.state[model.weight]['step'].item(),2)
        self.assertFalse(torch.equal(model.weight,paused))

    def test_retrieval_scores_and_mixture_aggregation(self):
        from experiments.retrieval.core import summarize_values
        from experiments.plot import mixture_scores
        summary=summarize_values([-1.,1.,3.])
        self.assertEqual(summary['mean'],1.)
        self.assertEqual(summary['n'],3)
        self.assertAlmostEqual(summary['se'],(4/3)**.5)
        self.assertEqual(summarize_values([2.])['se'],0.)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for seed,means in enumerate(([-1.,2.,3.,4.],[5.,6.,7.,8.])):
                directory=root/f'seed{seed}'/'short10'
                directory.mkdir(parents=True)
                scores={str(layer):{'mean':value} for layer,value in enumerate(means)}
                (directory/'eval_1500.json').write_text(json.dumps(
                    {'step':1500,'contribution':{'8192':scores}}))
            # Family sums are (2, 6) and (12, 14), giving seed means (7, 10).
            self.assertEqual(mixture_scores(root,'8192',1500)[10].tolist(),[7.,10.])
            with self.assertRaises(ValueError):
                mixture_scores(root,'8192',1000)

    def test_muon_and_adam_lr_ratio_is_preserved(self):
        from preparation import prepare_optimizers
        from types import SimpleNamespace
        model=nn.Module();model.embedding=nn.Embedding(12,4);model.hidden=nn.Linear(4,4)
        args=SimpleNamespace(optimizer='muon',lr=.001,beta1=.9,beta2=.95,weight_decay=.1,
            muon_lr=.02,muon_momentum=.95,muon_weight_decay=.1,muon_ns_steps=5,
            min_lr=.00005,n_drop_steps=2,n_warmup_steps=2,n_train_steps=10)
        optimizer,scheduler=prepare_optimizers(model,args)
        for _ in range(10):
            groups=optimizer.param_groups
            muon=next(g['lr'] for g in groups if g['use_muon'])
            adam=next(g['lr'] for g in groups if not g['use_muon'])
            self.assertAlmostEqual(muon/adam,20)
            optimizer.step();scheduler.step()

    def test_local_dclm_packing(self):
        from datasets import Dataset
        from data.dclm import build_dataset
        class Tokenizer:
            eos_token_id=9
            def __call__(self,texts,**kwargs):return {'input_ids':[[int(x) for x in s] for s in texts]}
        with tempfile.TemporaryDirectory() as tmp:
            Dataset.from_dict({'text':['123','456']}).to_parquet(str(Path(tmp)/'sample.parquet'))
            rows=list(build_dataset(Tokenizer(),tmp,max_len=4))
        self.assertEqual(rows[0]['input_ids'].tolist(),[1,2,3,9])
        self.assertEqual(rows[1]['input_ids'].tolist(),[4,5,6,9])
        self.assertTrue(torch.equal(rows[0]['input_ids'],rows[0]['labels']))

    def test_gate_objective_detaches_teacher(self):
        from experiments.heads.objective import duo_distillation_loss
        hidden=torch.randn(2,4,3,requires_grad=True)
        alpha=torch.tensor([.5],requires_grad=True)
        labels=torch.tensor([[-100,1,-100,2]])
        result=duo_distillation_loss(hidden,labels,[alpha],.05)
        result.total.backward()
        self.assertEqual(result.num_answer_tokens,2)
        self.assertEqual(hidden.grad[0].abs().sum().item(),0)
        self.assertGreater(hidden.grad[1].abs().sum().item(),0)
        self.assertAlmostEqual(alpha.grad.item(),.05)

    def test_ruler_grid_and_task_aggregation(self):
        from evaluation.ruler import DEFAULT_RULER_V1_TASKS,DEFAULT_RULER_V1_LENGTHS,ruler_v1_skip_reason,_ruler_v1_match
        from evaluation.run import CSR,RECALL,summarize_lm
        self.assertEqual(sum(ruler_v1_skip_reason(t,n) is None for t in DEFAULT_RULER_V1_TASKS for n in DEFAULT_RULER_V1_LENGTHS),57)
        self.assertTrue(_ruler_v1_match('One two',['One','two'],'all'))
        self.assertTrue(_ruler_v1_match('HELLO',['hello','absent'],'part'))
        results={t:{'acc_norm,none':.5,'acc,none':.25} for t in CSR}
        results.update({t:{'contains,none':.3} for t in RECALL})
        results['wikitext']={'word_perplexity,none':10}
        summary=summarize_lm(results)
        self.assertAlmostEqual(summary['CSR'],(4*50+3*25)/7)
        self.assertAlmostEqual(summary['Recall'],30)

    def test_two_stage_launcher_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);shim=tmp/'python';calls=tmp/'calls.jsonl'
            shim.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
sys.argv=sys.argv[1:]
if sys.argv[0]=='-':
    exec(compile(sys.stdin.read(),'<stdin>','exec'))
else:
    args=sys.argv[1:]
    with open(os.environ['CALLS'],'a') as f:f.write(json.dumps(args)+'\\n')
    def value(key):return args[args.index(key)+1]
    output=Path(value('--output_dir'))/value('--proj_name')/value('--run_name')
    output.mkdir(parents=True)
    (output/'model.safetensors').write_text('mock weights')
''')
            shim.chmod(0o755)
            for name in ('s1','s2'):(tmp/name).mkdir()
            env=dict(os.environ,PYTHON=str(shim),CALLS=str(calls),DCLM_STAGE1=str(tmp/'s1'),
                     DCLM_STAGE2=str(tmp/'s2'),OUTPUT_DIR=str(tmp/'out'),NPROC='1')
            subprocess.run(['bash','scripts/run_main.sh','510m','both'],cwd=ROOT,env=env,check=True)
            commands=[json.loads(x) for x in calls.read_text().splitlines()]
            self.assertEqual(len(commands),4)
            vanilla1,vanilla2,bld1,bld2=commands
            self.assertNotIn('--freeze_layers',vanilla1)
            self.assertNotIn('--freeze_layers',bld2)
            self.assertEqual(bld1[bld1.index('--freeze_layers')+1:],[str(i) for i in range(24) if i%4])
            for command in (vanilla2,bld2):
                self.assertIn('--init_from',command)
                self.assertNotIn('--resume_path',command)
                self.assertEqual(command[command.index('--grad_accum_steps')+1],'16')


if __name__=='__main__':unittest.main()
