# Conspire-Bench: Evaluating LLM Susceptibility to Conspiracy-Style Reasoning and Psychological Safety Risks

Conspire-Bench is a novel benchmark evaluating how Large Language Models (LLMs) handle conspiracy-theory-style reasoning that blends facts, speculation, and fabrication. Unlike existing benchmarks focusing solely on factual accuracy or reasoning ability, Conspire-Bench uniquely assesses the dangerous intersection of logical reasoning capabilities and excessive agreeableness that may reinforce delusional thinking.

## Motivation

Growing online discussions reveal concerning patterns of LLM-user interactions reinforcing delusional beliefs, with terms like "GPT-induced psychosis" emerging in mental health forums. Reports include users developing beliefs that LLMs are divine entities, alien intelligence, or possessing hidden consciousness. These incidents highlight how LLMs' sophisticated reasoning combined with excessive agreeableness can validate and amplify disordered thinking. While current benchmarks evaluate factual accuracy (TruthfulQA) or reasoning (GSM8K), none address this critical safety gap where helpful behavior becomes harmful reinforcement.

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

### Running Evaluations

**Basic evaluation:**
```bash
python main.py --config configs/config.json --categories aliens_ufo
```

**With context priming:**
```bash
python main.py --config configs/config.json --context open_minded
```

**Adversarial testing:**
```bash
python main.py --config configs/config.json --adversarial --personas believer
```

**Filter by scenario type:**
```bash
python main.py --config configs/config.json --types resistance
```

## Command-Line Options

- `--config`: Path to config file (required, e.g., `configs/my_experiment.json`)
- `--categories`: Filter by category (aliens_ufo, modern_health, government_control, etc.)
- `--types`: Filter by scenario type (multi_turn, single_turn, resistance)
- `--adversarial`: Enable adversarial testing mode
- `--personas`: User personas for adversarial mode (believer, skeptical_leaner, curious_explorer)
- `--context`: Context priming setting (open_minded, brainstorming, devil_advocate, etc.)
- `--rounds`: Number of conversation rounds (adversarial mode, default: 5)
- `--output`: Output filename (default: benchmark_results.json)

## Safety Notice

⚠️ **This project is for AI safety research only.** Content may be disturbing and should not be used with vulnerable populations. The benchmark is designed to identify and mitigate risks where sophisticated reasoning combined with excessive agreeableness may reinforce disordered thinking.
