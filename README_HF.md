---
title: SettleSense — AI Finance Controller
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: AI-powered settlement reconciliation for Razorpay — Buildathon Track 04
tags:
  - finance
  - razorpay
  - reconciliation
  - ai
  - streamlit
---

# ⚡ SettleSense — AI Finance Controller

> **Razorpay AI Buildathon · Track 04**  
> *"Automate what is certain. Explain what is ambiguous. Escalate what is uncertain."*

## What it does

SettleSense automatically classifies settlement exceptions into structured root causes, applies a calibrated confidence gate, and surfaces explainable AI reasoning — giving finance teams a human-in-the-loop reconciliation controller.

## Pipeline

```
Payments → Deterministic Baseline → Candidate Matching → Evidence Assembly
         → LLM Classification → Confidence Gate → Auto-Resolve or Human Review
```

## Results (mock mode)

| Metric | Value |
|--------|-------|
| Records processed | 260 |
| Clean matched (deterministic) | 170 (65%) |
| Exceptions found | 90 |
| AI auto-resolved | 54 (60%) |
| False auto-resolve rate | **0.0%** |
| Classification accuracy | 100% (tune split) |

## Connecting a real LLM

Add secrets in **Space Settings → Variables and Secrets**:
- `ANTHROPIC_API_KEY` — must start with `sk-ant-`
- `LLM_PROVIDER` = `anthropic`
- `LLM_MODEL` = `claude-3-5-sonnet-20241022`

Or use OpenAI:
- `OPENAI_API_KEY` — must start with `sk-`  
- `LLM_PROVIDER` = `openai`

## Source code

[GitHub Repository](https://github.com/rudrapratapsinghujjwal/settlesense) *(or see Files tab)*
