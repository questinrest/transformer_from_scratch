# Decoder-Only Transformer (From First Principles)

This repository contains a **from-scratch implementation of a GPT-style decoder-only Transformer**, built to study **architectural design choices and training stability** in autoregressive language models.

The emphasis of this project is on **controlled experimentation and interpretability**, rather than benchmark optimization.

---

## Project Goals

- Implement a decoder-only Transformer **from first principles** using PyTorch  
- Make training dynamics and architectural effects explicit  
- Design **controlled experiments** to isolate key design choices  
- Analyze results using gradients, attention behavior, and qualitative generation  

---

## Model Overview

The model follows a standard GPT-style decoder architecture:

- Token embedding + positional encoding  
- Stacked decoder blocks  
  - Masked multi-head self-attention  
  - Feed-forward network  
  - Residual connections  
  - Layer normalization (not yet configurable)  
- Linear projection to vocabulary logits  

### Key Characteristics

- Decoder-only (autoregressive)  
- Causal masking for next-token prediction  
- Custom character-level tokenization  
- Trained on the Tiny Shakespeare dataset  

---

## Experiments

All experiments are **controlled**: only one factor is modified at a time while keeping all other variables fixed.

---

### 1. Pre-LN vs Post-LN Normalization

**Question:**  
Where should LayerNorm be placed for stable Transformer training?

**Metrics Logged:**
- Training loss  
- Per-layer gradient norms  
- Activation statistics  
- Attention weights  
- Qualitative text generation  

**Observations:**  
*To be filled after completing experiments.*

---

### 2. Positional Encoding: Sinusoidal vs RoPE

**Question:**  
How does positional information affect attention and long-range dependency modeling?

**Metrics Logged:**
- Training loss  
- Attention heatmaps and entropy  
- Long-range token dependency behavior  
- Generated text samples  

**Observations:**  
*To be filled after completing experiments.*

---

### 3. Gradient Clipping: ON vs OFF

**Question:**  
What role does gradient clipping play in training stability?

**Metrics Logged:**
- Gradient norm distributions  
- Frequency of clipping  
- Loss stability  
- Divergence events (if any)  

**Observations:**  
*To be filled after completing experiments.*

---

## Experiment Tracking

All experiments are tracked using **Weights & Biases (W&B)** to enable:

- Consistent run comparison  
- Metric visualization  
- Attention and gradient analysis  
- Reproducibility  

---

## Tokenization Choice

A **custom character-level tokenizer** is used throughout this project to:

- Keep preprocessing transparent  
- Reduce confounding variables  
- Suit the Tiny Shakespeare dataset  
- Simplify attention interpretation  

Tokenizer comparisons are intentionally excluded to prioritize architectural analysis under limited compute.

---

## Repository Structure

will be updated, once complete.

---

## Key Takeaways

*To be added after all experiments are completed.*

---

## Findings Summary

| Experiment | Key Result | Evidence |
|----------|------------|----------|
| Pre-LN vs Post-LN | — | — |
| Sinusoidal vs RoPE | — | — |
| Gradient Clipping | — | — |

*Table to be populated after experiments.*

---

## Future Work

*Planned extensions and follow-up experiments will be documented here.*

---

## Limitations

- Experiments are conducted on a small corpus (Tiny Shakespeare)  
- Results focus on training dynamics and interpretability  
- No large-scale or production-level training is attempted  

These constraints are intentional to maintain experimental clarity.

---

## Notes

This project is designed as an **engineering study**, emphasizing controlled experimentation, interpretability, and reproducibility over raw performance.

---

## License

MIT
