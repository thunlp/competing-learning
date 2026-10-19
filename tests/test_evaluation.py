import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class EvaluationTests(unittest.TestCase):
    def test_based_tasks_load_documents_and_score_answers(self):
        from datasets import Dataset, DatasetDict
        from evaluation.prepare_data import load_lm_tasks
        from evaluation.run import RECALL
        row = dict(text='Context. Summary of information above... Name: example',
                   context='Context.', question='Name?', key='name', value='example',
                   answers=['example'], doc_id='sample')
        def dataset(*args, **kwargs):
            return DatasetDict(validation=Dataset.from_list([row]))
        with patch('datasets.load_dataset', side_effect=dataset), patch('evaluate.load') as metric:
            tasks, _ = load_lm_tasks(RECALL)
        metric.assert_called_once_with('squad_v2')
        self.assertEqual(set(tasks), set(RECALL))
        for name, task in tasks.items():
            with self.subTest(task=name):
                doc = task.eval_docs[0]
                self.assertEqual(task.OUTPUT_TYPE, 'generate_until')
                prompt = task.doc_to_text(doc)
                self.assertTrue(prompt)
                requests = task.construct_requests(doc, prompt)
                results = [' example' if item.request_type == 'generate_until' else (-10., False)
                           for item in requests]
                self.assertEqual(task.process_results(doc, results)['contains'], 1)

    def test_missing_ruler_data_fails_before_creating_output(self):
        from evaluation import run
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(sys, 'argv', ['evaluation.run', '--model', 'unused',
                    '--output', str(root / 'output'), '--ruler-data', str(root / 'missing')]):
                with self.assertRaises(FileNotFoundError):
                    run.main()
            self.assertFalse((root / 'output').exists())

    def test_lm_data_is_prepared_before_model_evaluation(self):
        from evaluation import run
        events = []
        task = object()
        results = {name: {'acc_norm,none': .5, 'acc,none': .5} for name in run.CSR}
        results.update({name: {'contains,none': .25} for name in run.RECALL})
        results['wikitext'] = {'word_perplexity,none': 12.}
        def data(names):
            events.append('data')
            self.assertEqual(names, [*run.CSR, *run.RECALL, 'wikitext'])
            return {'prepared': task}, object()
        def evaluate(**kwargs):
            events.append('model')
            self.assertEqual(kwargs['tasks'], [task])
            return {'results': results}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            with patch.object(sys, 'argv', ['evaluation.run', '--model', 'unused',
                    '--output', str(output), '--benchmark', 'lm']), \
                    patch('evaluation.prepare_data.load_lm_tasks', side_effect=data), \
                    patch('lm_eval.evaluator.simple_evaluate', side_effect=evaluate):
                run.main()
            self.assertEqual(events, ['data', 'model'])
            self.assertEqual(json.loads((output / 'summary.json').read_text())['Recall'], 25.)

    def test_ruler_skips_unused_resources_and_replaces_short_cache(self):
        from evaluation import prepare_ruler as prepare
        record = dict(index=0, input='Input', outputs=['answer'], length=64, answer_prefix='Answer:')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'qwen3/1024/niah_single_1/test.jsonl'
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(record) + '\n')
            def generate(*args, **kwargs):
                target.write_text((json.dumps(record) + '\n') * 2)
                return SimpleNamespace(returncode=0, stdout='', stderr='')
            with patch.object(prepare, 'preflight') as preflight, \
                    patch.object(prepare.subprocess, 'run', side_effect=generate) as process:
                successes, failures, skipped = prepare.generate(
                    'qwen3', 'unused', [1024], ['cwe', 'niah_single_1'], 2, root, False)
            preflight.assert_called_once_with(['niah_single_1'])
            self.assertEqual(process.call_count, 1)
            self.assertFalse(failures)
            self.assertEqual(len(skipped), 1)
            self.assertEqual(successes[0][2], 2)

    def test_qa_download_rejects_invalid_response_and_uses_fallback(self):
        from evaluation.prepare_data import download_qa
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'hotpotqa.json'
            with patch('urllib.request.urlopen', side_effect=[
                    io.BytesIO(b'<html>Error</html>'), io.BytesIO(b'[{"question":"Q"}]')]) as request:
                download_qa('qa_2', target)
            self.assertEqual(request.call_count, 2)
            self.assertEqual(json.loads(target.read_text()), [{'question': 'Q'}])
            self.assertFalse(target.with_suffix('.download').exists())

    def test_ruler_passes_absolute_paths_and_selected_interpreter(self):
        from evaluation import prepare_ruler as prepare
        record = dict(index=0, input='Input', outputs=['answer'], length=64, answer_prefix='Answer:')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = root / 'tokenizer'
            tokenizer.mkdir()
            def generate(command, **kwargs):
                self.assertEqual(command[command.index('--tokenizer_path') + 1], str(tokenizer))
                output = Path(command[command.index('--save_dir') + 1])
                self.assertTrue(output.is_absolute())
                self.assertEqual(kwargs['env']['PATH'].split(os.pathsep)[0], str(Path(sys.executable).parent))
                target = output / 'niah_single_1/test.jsonl'
                target.parent.mkdir(parents=True)
                target.write_text(json.dumps(record) + '\n')
                return SimpleNamespace(returncode=0, stdout='', stderr='')
            with patch.object(prepare, 'preflight'), patch.object(prepare.subprocess, 'run', side_effect=generate):
                successes, failures, _ = prepare.generate('qwen3', os.path.relpath(tokenizer),
                    [1024], ['niah_single_1'], 1, Path(os.path.relpath(root / 'generated')), False)
            self.assertEqual(len(successes), 1)
            self.assertFalse(failures)


if __name__ == '__main__':
    unittest.main()
