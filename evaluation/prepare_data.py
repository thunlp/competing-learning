"""Download evaluation datasets and the resources needed to generate RULER data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

from .run import CSR, RECALL


RULER_REVISION = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'
LM_TASKS = [*CSR, *RECALL, 'wikitext']
QA_URLS = {
    'qa_1': ['https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'],
    'qa_2': [
        'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json',
        'https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/'
        '7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json',
    ],
}


def load_lm_tasks(names):
    from lm_eval.tasks import TaskManager
    manager = TaskManager(include_path=str(Path(__file__).parent / 'tasks'))
    if set(names) & {'based_squad', 'based_triviaqa'}:
        import evaluate
        evaluate.load('squad_v2')
    return manager.load(names)['tasks'], manager


def prepare_lm(names):
    print(f'Preparing tasks: {", ".join(names)}', flush=True)
    tasks, _ = load_lm_tasks(names)
    records = {}
    for name in names:
        task = tasks[name]
        docs = task.eval_docs
        if not len(docs):
            raise ValueError(f'{name} has no evaluation documents')
        task.doc_to_text(docs[0])
        records[name] = dict(dataset=task.DATASET_PATH, subset=task.DATASET_NAME,
                             documents=len(docs), fingerprint=getattr(docs, '_fingerprint', None))
        print(f'{name}: {len(docs)} evaluation documents ready', flush=True)
    return records


def read_resource(path, kind):
    value = json.loads(path.read_text())
    if kind == 'essay':
        valid = isinstance(value, dict) and isinstance(value.get('text'), str) and bool(value['text'].strip())
    elif kind == 'qa_1':
        valid = isinstance(value, dict) and bool(value.get('data'))
    else:
        valid = isinstance(value, list) and bool(value)
    if not valid:
        raise ValueError(f'Invalid {kind} data: {path}')
    return value


def download_qa(task, target):
    if target.exists():
        read_resource(target, task)
        return
    partial = target.with_suffix('.download')
    errors = []
    for url in QA_URLS[task]:
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output)
            read_resource(partial, task)
            partial.replace(target)
            return
        except Exception as error:
            errors.append(f'{url}: {error}')
            partial.unlink(missing_ok=True)
    raise RuntimeError(f'Could not download {target.name}:\n' + '\n'.join(errors))


def prepare_ruler(names, revision):
    import nltk
    from .prepare_ruler import ESSAY_TASKS, QA_FILES
    source = Path(os.environ.get('RULER_SOURCE', 'third_party/RULER')).resolve()
    if not shutil.which('git'):
        raise RuntimeError('git is required to obtain NVIDIA/RULER')
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/NVIDIA/RULER.git', str(source)],
                       check=True, env=env)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', revision], check=True, env=env)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise ValueError(f'RULER revision is {actual}; expected {revision}. '
                         'Use a matching checkout or explicitly select --ruler-revision.')
    resources = source / 'scripts/data/synthetic/json'
    nltk_dir = Path(os.environ.get('NLTK_DATA', 'datasets/nltk_data')).resolve()
    nltk_dir.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(nltk_dir))
    for resource in ('punkt', 'punkt_tab'):
        try:
            nltk.data.find(f'tokenizers/{resource}')
            continue
        except LookupError:
            pass
        if not nltk.download(resource, download_dir=str(nltk_dir), quiet=True, raise_on_error=True):
            raise RuntimeError(f'Could not download NLTK resource {resource}')
    selected = []
    if set(names) & ESSAY_TASKS:
        essays = resources / 'PaulGrahamEssays.json'
        if not essays.exists():
            subprocess.run([sys.executable, '-c',
                'import runpy, socket; socket.setdefaulttimeout(30); '
                'runpy.run_path("download_paulgraham_essay.py", run_name="__main__")'],
                cwd=resources, check=True)
        read_resource(essays, 'essay')
        selected.append(essays)
    for task, filename in QA_FILES.items():
        if task in names:
            target = resources / filename
            download_qa(task, target)
            selected.append(target)
    return dict(generator_revision=actual, resources={
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in selected})


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--benchmark', choices=['lm', 'ruler'], default='lm')
    p.add_argument('--tasks', help='Comma-separated subset; default: all tasks in the selected group')
    p.add_argument('--ruler-revision', default=RULER_REVISION, help='Full NVIDIA/RULER commit ID')
    p.add_argument('--manifest', type=Path, default=Path('datasets/evaluation-manifest.json'))
    args = p.parse_args()
    if args.benchmark == 'lm':
        allowed = LM_TASKS
    else:
        from .ruler import DEFAULT_RULER_V1_TASKS
        allowed = [task for task in DEFAULT_RULER_V1_TASKS if task != 'cwe']
    names = args.tasks.split(',') if args.tasks else allowed
    if not names or set(names) - set(allowed):
        p.error(f'Expected tasks from {allowed}')
    result = prepare_lm(names) if args.benchmark == 'lm' else prepare_ruler(names, args.ruler_revision)
    existing = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
    if args.benchmark == 'lm':
        existing.setdefault('lm', {}).update(result)
    else:
        existing['ruler'] = result
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(existing, indent=2) + '\n')
    print(f'Data preparation complete. Manifest: {args.manifest}')


if __name__ == '__main__':
    main()
