"""CSR, Based recall, WikiText-2, and the paper's RULER task/length grid."""

import argparse
import json
from pathlib import Path
import statistics


CSR = ['arc_easy','arc_challenge','hellaswag','piqa','boolq','winogrande','lambada_standard']
RECALL = ['based_swde','based_fda','based_squad','based_triviaqa','based_drop','based_nq_2048']


def summarize_lm(results):
    def get(task,metric):
        return float(results[task][f'{metric},none'])
    # Use the same task-specific metrics as the paper's score table.
    csr_metrics = dict(arc_easy='acc_norm',arc_challenge='acc_norm',hellaswag='acc_norm',
                       piqa='acc_norm',boolq='acc',winogrande='acc',lambada_standard='acc')
    csr = {t:100*get(t,csr_metrics[t]) for t in CSR}
    recall = {t:100*get(t,'contains') for t in RECALL}
    return dict(CSR=statistics.mean(csr.values()),Recall=statistics.mean(recall.values()),
                WikiText_PPL=get('wikitext','word_perplexity'),csr_tasks=csr,recall_tasks=recall)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--benchmark',choices=['lm','ruler','all'],default='all')
    p.add_argument('--ruler-data',type=Path)
    p.add_argument('--batch-size',default='auto:4')
    p.add_argument('--ruler-batch-size',type=int,default=16)
    p.add_argument('--limit',type=int,help='Development subset; omit for paper evaluation')
    args=p.parse_args()
    if args.benchmark in ('ruler','all') and args.ruler_data is None:
        p.error('--ruler-data is required for RULER')
    args.output.mkdir(parents=True,exist_ok=False)
    summary={}
    if args.benchmark in ('lm','all'):
        from lm_eval import evaluator
        from lm_eval.tasks import TaskManager
        result=evaluator.simple_evaluate(model='hf',
            model_args=dict(pretrained=args.model,tokenizer=args.model,trust_remote_code=True,
                            max_length=65536,dtype='bfloat16'),
            tasks=[*CSR,*RECALL,'wikitext'],num_fewshot=0,batch_size=args.batch_size,
            task_manager=TaskManager(include_path=str(Path(__file__).parent/'tasks')),
            limit=args.limit,log_samples=True)
        # Keep sample-level outputs for score auditing without a dashboard.
        if result is not None:
            samples=result.pop('samples',{})
            for task,rows in samples.items():
                with (args.output/f'{task}_samples.jsonl').open('w') as f:
                    for row in rows:f.write(json.dumps(row,default=str)+'\n')
            (args.output/'lm_eval.json').write_text(json.dumps(result,indent=2,default=str))
            summary.update(summarize_lm(result['results']))
    if args.benchmark in ('ruler','all'):
        from .ruler import run_ruler_v1
        rows=run_ruler_v1(models={'model':args.model},tokenizer_path=args.model,
            tokenizer_name='qwen3',data_root=args.ruler_data,output_dir=args.output/'ruler',
            run_name='ruler',base_batch_size=args.ruler_batch_size,save_results=False,max_samples=args.limit)
        if rows:
            if len(rows)!=57:raise ValueError(f'Expected 57 RULER task/length entries, got {len(rows)}')
            summary['RULER']=100*statistics.mean(row['value'] for row in rows)
    if summary:
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
