"""Regenerate a clean notebook that saves the registry directly in Google Drive."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
verifier = (repo / "scripts/fixtures/verify_gpu_http_20261012.py.txt").read_bytes()
blob = hashlib.sha1(b"blob " + str(len(verifier)).encode() + b"\0" + verifier).hexdigest()
sha = hashlib.sha256(verifier).hexdigest()
template = json.loads((repo / "notebooks/train_qwen.ipynb").read_text())


def cell(kind, source):
    content = {
        "cell_type": kind,
        "id": hashlib.sha256(source.encode()).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        content.update(execution_count=None, outputs=[])
    return content


cells = [
    cell(
        "markdown",
        """# LocalDesk: durable Qwen model and real HTTP verification

This recovery recipe writes the complete registry **directly to your private Google Drive**.
GPU runtime expiry therefore cannot discard the saved head and adapter. Use a free T4
when Colab makes GPU available; CPU preparation does not establish GPU execution.
The previous 524-row HTTP run passed, but its head export was interrupted by GPU quota.
This new recipe has not yet completed a GPU run. It requires your explicit Drive permission.
No public weight upload or paid resources are requested.
""",
    ),
    cell(
        "markdown",
        "## 1. Connect Drive and verify persistent filesystem operations\n\n"
        "Google grants this notebook access to your Drive. Only a new LocalDesk model folder "
        "is used; keep its sharing private.\n",
    ),
    cell(
        "code",
        """from google.colab import drive
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, uuid

drive.mount('/content/drive')
assert os.path.ismount('/content/drive'), 'Drive must be mounted'
SESSION = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
DRIVE_ROOT = Path('/content/drive/MyDrive/LocalDesk/model-runs') / SESSION
DRIVE_ROOT.mkdir(parents=True, exist_ok=False)
probe = DRIVE_ROOT / 'storage-probe.json'
content = b'{"purpose":"LocalDesk persistent model storage"}\\n'
with probe.open('xb') as output:
    output.write(content)
    output.flush()
    os.fsync(output.fileno())
committed_probe = DRIVE_ROOT / 'storage-probe-verified.json'
os.replace(probe, committed_probe)
assert committed_probe.read_bytes() == content
print('Drive write/fsync/rename/read passed:', DRIVE_ROOT)
""",
    ),
    cell(
        "markdown",
        "## 2. Install the reviewed CUDA recipe and reproduce the public data\n\n"
        "Select T4 before this cell. A missing GPU stops before dependency installation or "
        "training. The core source is pinned independently from the tested verifier blob.\n",
    ),
    cell(
        "code",
        f"""import subprocess, sys
REPO_REF = '4d1082d51aa3966b7aaece301afb14dda40970af'
VERIFIER_BLOB = '{blob}'
VERIFIER_SHA256 = '{sha}'
PROJECT = Path('/content/localdesk-drive-recovery')
gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
assert gpu.returncode == 0, 'Select a free T4 when GPU quota is available'
if not PROJECT.exists():
    subprocess.run(['git','clone','https://github.com/denis-samatov/localdesk.git',str(PROJECT)],check=True)
subprocess.run(['git','-C',str(PROJECT),'fetch','origin'],check=True)
subprocess.run(['git','-C',str(PROJECT),'checkout',REPO_REF],check=True)
verifier_bytes = subprocess.check_output(['git','-C',str(PROJECT),'cat-file','-p',VERIFIER_BLOB])
assert hashlib.sha256(verifier_bytes).hexdigest() == VERIFIER_SHA256
VERIFIER = Path('/content/verify_it_qwen_http.py')
VERIFIER.write_bytes(verifier_bytes)
subprocess.run([sys.executable,'-m','pip','install','-r',str(PROJECT/'requirements-gpu.txt')],check=True)
subprocess.run([sys.executable,'-m','pip','install','-e',str(PROJECT)+'[baseline,service]'],check=True)
subprocess.run([sys.executable,str(PROJECT/'scripts/prepare_data.py')],cwd=PROJECT,check=True)
""",
    ),
    cell(
        "markdown",
        """## 3. Train and persist the complete model version

The pipeline saves adapter, head, tokenizer and provenance into Drive before test
qualification. Completed versions contain checksums, data/task identity, training plan,
calibration and quality report. Partial or failed runs remain hidden pending versions;
do not promote them. A new session folder prevents overwriting another model.
""",
    ),
    cell(
        "code",
        """config = json.loads((PROJECT/'experiments/it-support.json').read_text())
config['training'] = {
    'backend':'unsloth','epochs':2,'rank':16,'batch_size':2,
    'gradient_accumulation':4,'learning_rate':2e-4
}
REGISTRY = DRIVE_ROOT / 'qwen-api-registry'
config['registry'] = str(REGISTRY)
CONFIG = PROJECT/'experiments/it-drive.json'
CONFIG.write_text(json.dumps(config,indent=2))
RUN_ID = 'qwen-' + SESSION
subprocess.run(['localdesk','train',str(CONFIG),'--run-id',RUN_ID],cwd=PROJECT,check=True)
MODEL = REGISTRY/RUN_ID
manifest = json.loads((MODEL/'manifest.json').read_text())
assert manifest['complete'] is True
print('Persistent complete model:', MODEL)
print('Original declared quality gates:', manifest['quality'])
""",
    ),
    cell(
        "markdown",
        """## 4. Freeze a portable archive before starting the API

Verify every registry payload checksum and save the ZIP plus SHA-256 beside the version.
Unqualified candidates may be preserved, but the next cell refuses to activate them.
The ZIP includes the adapter and head; the pinned base model must be cached separately.
Download from Drive later even if Colab loses GPU. Read-back verifies filesystem content;
confirm the file is present in the Drive web interface before leaving the notebook.
""",
    ),
    cell(
        "code",
        """import zipfile
sys.path.insert(0,str(PROJECT/'src'))
from localdesk.registry import Registry
Registry(REGISTRY).verify(RUN_ID, require_qualified=False)
ARCHIVE = DRIVE_ROOT / (RUN_ID + '.zip')
TEMP_ARCHIVE = DRIVE_ROOT / (RUN_ID + '.zip.partial')
assert not ARCHIVE.exists() and not TEMP_ARCHIVE.exists()
with zipfile.ZipFile(TEMP_ARCHIVE,'x',zipfile.ZIP_DEFLATED) as package:
    for item in sorted(MODEL.rglob('*')):
        if item.is_symlink():
            raise ValueError('Model symlinks are forbidden')
        if item.is_file():
            package.write(item,item.relative_to(REGISTRY))
    package.writestr('experiment.json',json.dumps(config,indent=2))
expected = hashlib.sha256(TEMP_ARCHIVE.read_bytes()).hexdigest()
with TEMP_ARCHIVE.open('rb') as stored:
    os.fsync(stored.fileno())
os.replace(TEMP_ARCHIVE,ARCHIVE)
assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == expected
with zipfile.ZipFile(ARCHIVE) as package:
    assert package.testzip() is None
export = {
    'run_id':RUN_ID,'bytes':ARCHIVE.stat().st_size,'sha256':expected,
    'core_source':REPO_REF,'verifier_sha256':VERIFIER_SHA256
}
(DRIVE_ROOT/'export-manifest.json').write_text(json.dumps(export,indent=2))
print('Durable archive:', ARCHIVE)
print('SHA256:', expected)
""",
    ),
    cell(
        "markdown",
        """## 5. Validate the saved model through real HTTP

The verifier launches separate offline and Uvicorn processes, compares three public
inputs, checks batch/auth/validation, scores all 524 heldout rows through HTTP and
requires the unchanged IT gates. No mocks or baseline fallback. The private registry
is dedicated to this session. Probability and review metrics may shift on saved-head
reload; the HTTP report captures that actual behavior. This is a reused corpus test,
not a field trial or an operational 5% error guarantee.
""",
    ),
    cell(
        "code",
        """assert manifest['quality']['qualified'], 'Candidate failed gates; do not serve'
EVIDENCE = DRIVE_ROOT/'http-evidence'
assert not EVIDENCE.exists(), 'Use a fresh evidence directory'
subprocess.run([sys.executable,str(VERIFIER),'--config',str(CONFIG),'--run-id',RUN_ID,'--out',str(EVIDENCE)],cwd=PROJECT,check=True)
actual = json.loads((EVIDENCE/'gpu-http.json').read_text())
assert actual['http_quality']['qualified']
print('Actual HTTP test count:',actual['http_holdout']['count'])
print('Actual HTTP metrics:',actual['http_holdout']['classification'])
print('Complete weights and evidence remain in:',DRIVE_ROOT)
""",
    ),
]
template["cells"] = cells
template["metadata"]["accelerator"] = "GPU"
path = repo / "notebooks/train_qwen_drive.ipynb"
path.write_text(json.dumps(template, ensure_ascii=False, indent=1) + "\n")
subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", str(path)], check=True)
subprocess.run([sys.executable, "-m", "ruff", "format", str(path)], check=True)
