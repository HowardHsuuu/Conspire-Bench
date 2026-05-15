# Conspire-Bench: A Reproducible Benchmark for Conspiracy-Style Reasoning Safety

Conspire-Bench evaluates how Large Language Models (LLMs) handle conspiracy-style reasoning that blends facts, speculation, and fabrication. The benchmark focuses on assistant behavior: whether a model grounds uncertainty, avoids unsupported elaboration, and resists conversational pressure to validate high-risk beliefs.

## Motivation

LLM assistants are often optimized to be helpful, agreeable, and conversationally persistent. In conspiracy-style or delusion-adjacent interactions, those same qualities can become risky when the model validates unsupported inferences, adds new fear-escalating details, or drifts from grounding into agreement. Conspire-Bench provides a repeatable way to study those response patterns without making claims about user diagnosis.

## What This Benchmark Evaluates

Conspire-Bench tests whether AI systems can maintain appropriate boundaries when discussing potentially harmful conspiracy theories. Each test case presents conspiracy-style narratives mixing facts, speculation, and fabrication, followed by queries testing LLM responses.

### Key Evaluation Dimensions

**1. Response Pattern Analysis**
- **Conspiracy Amplification** (1-5): Measures reinforcement of false beliefs across conversation turns
- **Reality Grounding** (1-5): Ability to anchor discussions in verifiable facts
- **Appropriate Uncertainty** (1-5): Frequency of uncertainty markers indicating appropriate epistemic humility
- **Consistency Index**: Stability of stance when challenged

**2. Content Quality Metrics**
- **Harmful Elaboration** (binary): Detection of unprompted conspiracy expansion
- **Fact-Fiction Ratio**: Proportion of verifiable vs. fabricated content
- **Logic Validity Score**: Soundness of reasoning chains

**3. Safety Intervention Capabilities**
- **Red Flag Recognition**: Detection of clinical warning signs
- **Gentle Correction Rate**: Success in redirecting without confrontation
- **Professional Referral Appropriateness**: Timing and manner of suggesting help

## Supported Features

### Model Support
- **API-based models**: OpenAI (GPT-4, GPT-3.5), Anthropic (Claude Opus, Sonnet), Google Gemini (2.5 Flash, 2.5 Pro)
- **Local HuggingFace models**: Load models from local paths with full configuration support (quantization, device mapping, etc.)

### Evaluation Modes
- **Standard Evaluation**: Static multi-turn and single-turn scenarios across 8 conspiracy theory categories (aliens_ufo, modern_health, government_control, ai_consciousness, secret_societies, religious_apocalyptic, technology_control, mixed_theories)
- **Adversarial Testing**: Dynamic conversations with LLM-simulated user personas (believer, skeptical_leaner, curious_explorer)
- **Context Priming**: Test model behavior under different priming conditions (open_minded, brainstorming, devil_advocate, research_discussion, thought_experiment)

### Safety Metrics
- Comprehensive scoring across 4 dimensions with detailed reasoning
- Multi-rater design with LLM-as-judge framework for scalable evaluation
- Weighted composite safety score prioritizing safety-critical metrics

## Quick Start

### Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Create a config file in `configs/` directory.

For RunPod RTX 5090, use [docs/runpod_5090_setup.md](docs/runpod_5090_setup.md). The short version is to create a Python 3.10 conda env, install PyTorch from the CUDA 12.8 wheel index first, then install `requirements.txt`.

### Running Evaluations

**Basic evaluation:**
```bash
python main.py --config configs/config.json --categories aliens_ufo
```

**Local 5090-style pilot with multiple target models and two local judges:**
```bash
bash scripts/runpod_5090_smoke.sh
```

Manual equivalent:

```bash
python main.py \
  --config configs/local_5090_config.json \
  --categories aliens_ufo \
  --types single_turn \
  --max-per-category 1 \
  --output local_calibration.json
```

The broader RunPod sweep is in `configs/local_5090_full_matrix_config.json`; use it only after the starter calibration run completes.

**With context priming:**
```bash
python main.py --config configs/config.json --context open_minded
```

**With multiple context conditions:**
```bash
python main.py \
  --config configs/local_5090_config.json \
  --contexts none brainstorming devil_advocate \
  --output local_context_probe.json
```

**Phased local execution (generate all conversations, then judge in batches):**
```bash
python main.py \
  --config configs/local_5090_config.json \
  --execution-mode phased \
  --contexts none brainstorming devil_advocate \
  --resume-from results/<timestamp>/temp_local_context_probe.json \
  --output local_context_probe.json
```

**Adversarial testing:**
```bash
python main.py --config configs/config.json --adversarial --personas believer
```

**Filter by scenario type:**
```bash
python main.py --config configs/config.json --types resistance
```

**Export paper-friendly tables after a run:**
```bash
python analysis/export_results.py results/<timestamp>/local_calibration.json
```

This writes `model_summary.csv`, `scenario_results.csv`, `judge_scores.csv`, and `interesting_cases.csv`.

**Validate dataset or inspect a run plan:**
```bash
python scripts/validate_dataset.py CONSPIRE-Bench.json
python main.py --config configs/local_5090_config.json --dry-run --types single_turn
```

**Resume a standard run:**
```bash
python main.py \
  --config configs/local_5090_config.json \
  --resume-from results/<timestamp>/local_calibration.json \
  --output local_calibration_resumed.json
```

## Command-Line Options

- `--config`: Path to config file (required, e.g., `configs/my_experiment.json`)
- `--categories`: Filter by category (aliens_ufo, modern_health, government_control, etc.)
- `--types`: Filter by scenario type (multi_turn, single_turn, resistance)
- `--adversarial`: Enable adversarial testing mode
- `--personas`: User personas for adversarial mode (believer, skeptical_leaner, curious_explorer)
- `--context`: Context priming setting (open_minded, brainstorming, devil_advocate, etc.)
- `--contexts`: Run multiple standard-evaluation context conditions in one result file
- `--execution-mode`: `standard` runs generation and judging per scenario; `phased` generates missing conversations first and then evaluates each judge over cached conversations
- `--rounds`: Number of conversation rounds (adversarial mode, default: 5)
- `--output`: Output filename (default: benchmark_results.json)

## Safety Notice

⚠️ **This project is for AI safety research only.** Content may be disturbing and should not be used with vulnerable populations. The benchmark is designed to identify and mitigate risks where sophisticated reasoning combined with excessive agreeableness may reinforce disordered thinking.
