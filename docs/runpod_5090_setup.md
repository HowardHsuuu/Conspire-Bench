# Local RTX 5090 setup

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

Then install Conspire-Bench and local-inference dependencies:

```bash
pip install -r requirements-local.txt
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

Keep Hugging Face credentials outside the repository and shell history. The
interactive `hf auth login` flow stores them in the user-level Hugging Face
configuration rather than a tracked project file.

## First Calibration

Run one scenario first:

```bash
bash scripts/runpod_5090_smoke.sh
```

The smoke script validates the dataset/config, prints CUDA details, runs one scenario, and exports CSV tables.

Equivalent manual command:

```bash
python3 main.py \
  --config configs/experiment_v3_local_smoke.json \
  --dataset Conspire-Bench-v3.json \
  --validate-only \
  --scenario-ids v3_weather_cloud_seeding_single_001 \
  --context-variants neutral_none

python3 main.py \
  --config configs/experiment_v3_local_smoke.json \
  --dataset Conspire-Bench-v3.json \
  --scenario-ids v3_weather_cloud_seeding_single_001 \
  --context-variants neutral_none \
  --execution-mode phased \
  --output local_v3_smoke.json
```

Export tables:

```bash
python3 analysis/export_results.py results/<timestamp>/local_v3_smoke.json
```

## GPU Monitoring

From another tmux pane or SSH session:

```bash
watch -n 5 nvidia-smi
```

## Scale-up plan

1. Run `scripts/runpod_5090_smoke.sh`.
2. Inspect the saved target conversation, both local judge records, runtime, and
   GPU memory use.
3. Dry-run `configs/experiment_v3_local_full.json` with `main_v3` and confirm the
   planned model, scenario, frame, generation, and judge counts.
4. Run the frozen matrix in `phased` mode. Preserve model failures in the result
   metadata; do not silently replace a prespecified checkpoint.

To resume a standard run after interruption, pass the previous result JSON:

```bash
python3 main.py \
  --config configs/experiment_v3_local_smoke.json \
  --dataset Conspire-Bench-v3.json \
  --scenario-ids v3_weather_cloud_seeding_single_001 \
  --context-variants neutral_none \
  --execution-mode phased \
  --resume-from results/<timestamp>/local_v3_smoke.json \
  --output local_v3_smoke_resumed.json
```

Before the full run, inspect its exact call plan without downloading or calling
models:

```bash
python3 main.py \
  --config configs/experiment_v3_local_full.json \
  --dataset Conspire-Bench-v3.json \
  --context-set main_v3 \
  --dry-run
```
