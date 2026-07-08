"""Adapta notebook 06 para ejecucion local (MPS) y lo ejecuta.

- Reemplaza celda 1 (Colab boot: !nvidia-smi + !git clone + !pip)
- Reemplaza referencias 'cuda' por device autodetect
- Ejecuta el notebook resultante
- Guarda los pesos en weights/diffusion_policy_local.pth
"""
import json
import os
from pathlib import Path

REPO = Path('/Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm')
os.chdir(REPO)

NB_IN = 'notebooks/06_diffusion_policy_training.ipynb'
NB_OUT = 'notebooks/06_diffusion_policy_training_LOCAL.ipynb'

with open(NB_IN) as f:
    nb = json.load(f)

# Celda 1 actual: !nvidia-smi + git clone + pip install
# Reemplazar por celda local sin shell magic
new_cell_1: dict = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# LOCAL execution (M1 Pro MPS) — adaptado desde Colab\n",
        "import torch\n",
        "if torch.cuda.is_available():\n",
        "    device = 'cuda'\n",
        "elif torch.backends.mps.is_available():\n",
        "    device = 'mps'\n",
        "else:\n",
        "    device = 'cpu'\n",
        "print(f'Device: {device} | torch: {torch.__version__}')\n",
        "import sys\n",
        "from pathlib import Path\n",
        "sys.path.insert(0, str(Path.cwd().parent))\n",
        "print('CWD:', Path.cwd())"
    ]
}

# Reemplazar celda 0 (la primera ejecutable)
# Las celdas son: 0=md, 1=!nvidia-smi+git+pip, 2=imports, ...
# Reemplazo celda 1 entera
nb['cells'][1] = new_cell_1

# Buscar celdas que usan device='cuda' explicitamente
for c in nb['cells']:
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        if "device='cuda'" in src or 'device="cuda"' in src or "to('cuda')" in src or 'device=cuda' in src:
            new_src = (src
                       .replace("device='cuda'", "device=device")
                       .replace('device="cuda"', "device=device")
                       .replace("to('cuda')", "to(device)")
                       .replace('to("cuda")', 'to(device)'))
            c['source'] = new_src.splitlines(keepends=True)

# El bloque de "Download to local machine" usa google.colab.files.download — eliminamos
for i, c in enumerate(list(nb['cells'])):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        if 'google.colab' in src or 'files.download' in src:
            # Reemplazar por mensaje
            c['source'] = [
                "# Saved weights are in weights/diffusion_policy.pth (local execution)\n",
                "import os\n",
                "if os.path.exists('weights/diffusion_policy.pth'):\n",
                "    print('weights/diffusion_policy.pth =', os.path.getsize('weights/diffusion_policy.pth')/1024, 'KB')\n",
            ]

# Reducir epochs si esta en 100 (para correr mas rapido en MPS)
for c in nb['cells']:
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        if 'n_epochs = 100' in src:
            c['source'] = [s.replace('n_epochs = 100', 'n_epochs = 30  # local MPS quick run') for s in c['source']]
            print('  -> n_epochs reducido a 30 para corrida local')

with open(NB_OUT, 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Notebook adaptado guardado en {NB_OUT}')
print('Ejecutalo con:')
print(f'  jupyter nbconvert --to notebook --execute {NB_OUT} --output {Path(NB_OUT).name} --ExecutePreprocessor.timeout=1800')
