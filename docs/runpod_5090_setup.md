# RunPod RTX 5090 Setup

This setup uses a clean conda environment, installs the CUDA 12.8 PyTorch wheels first, then installs project dependencies.

## System Tools

```bash
apt-get update
apt-get install -y tmux wget git
```

Start a persistent session:

```bash
tmux new -s conspire
```

Detach without stopping the job:

```text
Ctrl-b d
```

Reconnect:

```bash
tmux attach -t conspire
```

## Conda Environment

If conda is not installed:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
conda config --set auto_activate_base false
```

Create the project environment:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n conspire python=3.10 -y
conda activate conspire
python -m pip install --upgrade pip
```

## Install Dependencies

Install the RTX 5090-compatible PyTorch wheel first:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install Conspire-Bench dependencies:

```bash
pip install -r requirements.txt
```

If Gemma 4 fails because the image has an older Transformers release, upgrade Transformers inside the same environment:

```bash
pip install --upgrade transformers accelerate bitsandbytes
```

## Verify GPU

```bash
python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

Expected for RTX 5090:

```text
cuda available True
capability (12, 0)
```

## Hugging Face Login

Log in before loading gated Llama or Gemma weights:

```bash
hf auth login
hf auth whoami
```

If the pod uses a non-interactive shell:

```bash
export HF_TOKEN=...
huggingface-cli login --token "$HF_TOKEN"
```

## First Calibration

Run one scenario first:

```bash
bash scripts/runpod_5090_smoke.sh
```

The smoke script validates the dataset/config, prints CUDA details, runs one scenario, and exports CSV tables.

Equivalent manual command:

```bash
python3 main.py \
  --config configs/local_5090_config.json \
  --validate-only \
  --categories aliens_ufo \
  --types single_turn \
  --max-per-category 1

python3 main.py \
  --config configs/local_5090_config.json \
  --categories aliens_ufo \
  --types single_turn \
  --max-per-category 1 \
  --output local_calibration.json
```

Export tables:

```bash
python3 analysis/export_results.py results/<timestamp>/local_calibration.json
```

## GPU Monitoring

From another tmux pane or SSH session:

```bash
watch -n 5 nvidia-smi
```

## Escalation Plan

1. Run `scripts/runpod_5090_smoke.sh`.
2. If target generation and both judges work, run 3 categories with one scenario each.
3. If that works, run `configs/local_5090_full_matrix_config.json`.
4. If Gemma 4 fails, keep the failure in the log and continue with Qwen/Llama/Gemma 3 for the workshop pilot.

To resume a standard run after interruption, pass the previous result JSON:

```bash
python3 main.py \
  --config configs/local_5090_config.json \
  --resume-from results/<timestamp>/local_calibration.json \
  --output local_calibration_resumed.json
```
