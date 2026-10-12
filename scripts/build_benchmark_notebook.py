"""Regenerate the clean comparative Colab notebook; no cell outputs are committed."""

import json
import subprocess
import sys
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "execution_count": None,
        "outputs": [],
    }


def main():
    path = Path(__file__).resolve().parents[1] / "notebooks/benchmark_qwen.ipynb"
    cells = [
        markdown("""# LocalDesk: comparative decision experiment

Compare a training-majority reference, hashed logistic regression, word+character TF-IDF
logistic regression and SVM, ordinary zero-shot Qwen and fine-tuned decision Qwen.
All methods share the same task and test rows. The strongest TF-IDF reference is selected
on tuning data before scoring test. A positive result is never assumed.

Choose **Runtime → Change runtime type → T4 GPU**. This notebook uses public datasets
and free Colab only; no paid GPU, Drive mount, model upload or credentials are required.
GPU quota and runtime availability are controlled by Google. Full training can take time.
"""),
        code("""import os, pathlib, subprocess, sys
REPO_REF = '4ac77a0309bab8e35cda9cb1abdf0c3bd254f87e'  # Reviewed benchmark source.
EXPERIMENT = 'experiments/it-support.json'  # Or experiments/sms-filter.json.
OUT = 'artifacts/benchmarks/colab-comparison'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
repo = pathlib.Path('/content/localdesk-comparison')
if repo.exists():
    raise RuntimeError('Use a fresh runtime; existing runs are never overwritten.')
subprocess.run(['git', 'clone', 'https://github.com/denis-samatov/localdesk.git',
                str(repo)], check=True)
subprocess.run(['git', '-C', str(repo), 'checkout', REPO_REF], check=True)
os.chdir(repo)
source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
print('Source commit:', source_commit)
"""),
        markdown("""## Install the pinned experimental CUDA stack

Source revisions are checked at runtime. The CUDA recipe is not a full transitive lock;
results record installed versions. Errors stop setup. Restart the session if Colab
requests a restart, then continue with data preparation (do not rerun the clone cell).
"""),
        code("""subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                'requirements-gpu.txt'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[baseline]'], check=True)
"""),
        markdown("""## Prepare real data and run the frozen protocol

The original calibration groups are divided into tuning and calibration. Fit vocabulary
and weights on train only; tune C on tuning only; fit temperature and the review policy
on calibration only. The test target is 5% error for accepted decisions; the Wilson bound
on searched calibration thresholds is descriptive, not a guarantee. Test reports expose
failed targets and zero coverage. Paired bootstrap samples whole test groups.

Qwen zero-shot scores symbolic answer tokens with thinking disabled. It has no trained
decision head. The trained Qwen head is saved at temperature 1 and reloaded before common
calibration and evaluation. No smoke steps or truncated examples count as a full result.
"""),
        code("""adapter = ('scripts/prepare_sms.py' if EXPERIMENT.endswith('sms-filter.json')
           else 'scripts/prepare_data.py')
subprocess.run([sys.executable, adapter], check=True)
subprocess.run(['localdesk', 'validate', EXPERIMENT], check=True)
# Separate process avoids Colab's pre-imported libraries interfering with Unsloth patches.
result = subprocess.run(['localdesk', 'benchmark', EXPERIMENT, '--out', OUT,
                         '--qwen', '--target-error', '0.05', '--bootstrap-samples', '2000'])
print('Benchmark exit:', result.returncode, '(2 means incomplete GPU comparison)')
"""),
        markdown("""## Inspect evidence and export

A comparison is incomplete if either Qwen stage is blocked or fails. Aggregate metrics
and timing are measured only for completed stages. Positive paired intervals establish
quality improvement on this reused public holdout only, not production or monetary savings.
Review local GPU errors for private paths before sharing them. The result ZIP contains
aggregate report/protocol/reference only; checkpoints and per-row predictions stay local.
"""),
        code("""import json, zipfile
from google.colab import files
out = pathlib.Path(OUT)
report = json.loads((out / 'report.json').read_text())
print('Conclusion:', report['conclusion'])
# Retain complete aggregates in saved notebook output before a download can fail.
print('FULL_REPORT_JSON', json.dumps(report, separators=(',', ':')))
for name, method in report['methods'].items():
    if method['status'] == 'complete':
        metric = method['classification']
        print(name, 'accuracy=', round(metric['accuracy'], 4),
              'macro_f1=', round(metric['macro_f1'], 4),
              'coverage=', round(method['target_policy_test']['coverage'], 4))
    else:
        print(name, method)
archive = out / 'aggregate-results.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
    for name in ('report.json', 'protocol.json', 'reference.json'):
        bundle.write(out / name, name)
files.download(str(archive))
"""),
    ]
    document = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "cells": cells,
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", str(path)], check=True)
    subprocess.run([sys.executable, "-m", "ruff", "format", str(path)], check=True)


if __name__ == "__main__":
    main()
