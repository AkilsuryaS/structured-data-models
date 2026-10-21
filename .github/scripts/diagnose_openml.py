import faulthandler
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

if os.environ.get("OPENML_DIAGNOSTIC_CHILD") != "1":
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "-u", __file__],
            env={**os.environ, "OPENML_DIAGNOSTIC_CHILD": "1"},
            timeout=90,
            check=False,
        )
        print(f"Child exited {result.returncode} after {time.monotonic() - started:.2f}s", flush=True)
        sys.exit(result.returncode)
    except subprocess.TimeoutExpired:
        print("Child exceeded 90 seconds; stack dumps above identify the request.", flush=True)
        sys.exit(124)

faulthandler.enable()
faulthandler.dump_traceback_later(20, repeat=True)

import openml
import requests

started = time.monotonic()
original_request = requests.Session.request

def request(self, method, url, *args, **kwargs):
    print(f"{time.monotonic() - started:.3f}s HTTP BEGIN {method} {url}", flush=True)
    result = original_request(self, method, url, *args, **kwargs)
    print(f"{time.monotonic() - started:.3f}s HTTP END {result.status_code}", flush=True)
    return result

requests.Session.request = request
cache = Path(tempfile.mkdtemp(prefix="openml-ci-diagnostic-"))
openml.config.set_root_cache_directory(cache)
print(f"OpenML {openml.__version__}, cold cache {cache}", flush=True)
task = openml.tasks.get_task(
    363621,
    download_splits=True,
    download_data=True,
    download_qualities=False,
    download_features_meta_data=True,
)
print(f"Task loaded, dataset {task.dataset_id}", flush=True)
dataset = task.get_dataset(download_data=True)
dataset.get_data(task.target_name)
print(f"Completed task/data/splits in {time.monotonic() - started:.3f}s", flush=True)
faulthandler.cancel_dump_traceback_later()
