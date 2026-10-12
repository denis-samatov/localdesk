"""Run the exact reviewed GPU HTTP fixture retained with its execution fingerprint."""

import hashlib
import runpy
from pathlib import Path

fixture = Path(__file__).parent / "fixtures/verify_gpu_http_20261012.py.txt"
expected = "62eb9ec73c088804e782b53f3360c4a326136caa050d5cc134a8e8b978a0ca08"
if hashlib.sha256(fixture.read_bytes()).hexdigest() != expected:
    raise ValueError("reviewed verifier fingerprint mismatch")
runpy.run_path(str(fixture), run_name="__main__")
