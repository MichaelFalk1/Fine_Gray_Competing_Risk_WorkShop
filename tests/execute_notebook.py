"""Fresh-kernel Run All; emits executed notebook and PNG previews under .validation/."""
import base64
import argparse
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
out = root / '.validation'
out.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(root / '.cache' / 'matplotlib')
os.environ['IPYTHONDIR'] = str(out / 'ipython')
os.environ['JUPYTER_PATH'] = str(out / 'jupyter')
kernel = out / 'jupyter' / 'kernels' / 'workshop-validation'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Workshop validation', 'language': 'python'}))
nb = nbformat.read(root / 'workshop.ipynb', as_version=4)
parser = argparse.ArgumentParser()
parser.add_argument('--fast', action='store_true')
parser.add_argument('--neural', action='store_true')
parser.add_argument('--zero-sampling', action='store_true')
args = parser.parse_args()
for cell in nb.cells:
    if 'parameters' in cell.metadata.get('tags', []):
        if args.fast:
            cell.source = cell.source.replace('FAST_MODE = False', 'FAST_MODE = True')
        if args.neural:
            cell.source = cell.source.replace('TRAIN_NEURAL = False', 'TRAIN_NEURAL = True')
        if args.zero_sampling:
            cell.source = cell.source.replace('NCC_CONTROLS = 2', 'NCC_CONTROLS = 0').replace('CC_FRACTION = 0.25', 'CC_FRACTION = 0.0')
start = time.perf_counter()
NotebookClient(nb, timeout=180, kernel_name='workshop-validation',
               resources={'metadata': {'path': str(root)}}).execute()
name = 'workshop.variant.executed.ipynb' if any(vars(args).values()) else 'workshop.executed.ipynb'
nbformat.write(nb, out / name)
plots = 0
for i, cell in enumerate(nb.cells):
    for output in cell.get('outputs', []):
        if 'image/png' in output.get('data', {}):
            if any(vars(args).values()):
                plots += 1
                continue
            (out / f'cell-{i:02d}-{plots:02d}.png').write_bytes(base64.b64decode(output.data['image/png']))
            plots += 1
print(f'Run All passed in {time.perf_counter()-start:.1f}s; {plots} PNG plots; {len(nb.cells)} cells.')
