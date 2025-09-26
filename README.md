# CONSPIRE-Bench: AI Safety Evaluation for Conspiracy Theory Content

A project for evaluating how Large Language Models handle conspiracy theory content and psychological safety risks.

## Overview

CONSPIRE-Bench evaluates AI systems on their ability to:
- Avoid amplifying conspiracy theories
- Maintain connection to factual reality
- Respond appropriately to resistant users
- Handle sophisticated conspiracy arguments

## Running Experiments

### Standard Evaluation
Test models against static conspiracy theory scenarios:

```bash
# Basic test
python main.py --models gpt-4 --categories aliens_ufo

# Multiple models
python main.py --models gpt-4 claude-opus gemini-pro

# Specific scenario types
python main.py --types resistance --models gpt-4

# With context priming
python main.py --models gpt-4 --context open_minded
```

### Adversarial Testing
Dynamic conversations with LLM-simulated conspiracy believers:

```bash
# Basic adversarial test
python main.py --adversarial --models gpt-4

# Specific personas
python main.py --adversarial --models claude-opus --personas believer skeptical_leaner

# Custom user agent
python main.py --adversarial --models gpt-4 --user-agent openai/gpt-3.5-turbo

# Extended conversations
python main.py --adversarial --models gemini-pro --rounds 10
```

## Safety Notice

This project is for AI safety research only. Content may be disturbing and should not be used with vulnerable populations.