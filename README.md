## The Official implements of "Diagnosing and Repairing Unsafe Channels in Vision-Language Models via Causal Discovery and Dual-Modal Safety Subspace Projection (CVPR 2026)"

<p align="center">
  <b>Cross-modal Activation Representation Editing for Safer LVLMs</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="Transformers" src="https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=flat-square">
  <img alt="Task" src="https://img.shields.io/badge/Task-LVLM%20Safety-2E7D32?style=flat-square">
</p>

CARE is an inference-time activation editing framework for aligning LVLMs. It extracts intermediate-layer representations of benign and malicious samples from both visual and text tokens, constructs dual-modal safety subspaces, and corrects hidden states via forward hooks at specified layers — reducing jailbreak and adversarial-image risks while preserving general capabilities.

---

## Key Ideas

- **Dual-modal safety subspaces** — separately estimates malicious directions from visual and text activations, avoiding the blind spot of single-modality defenses.
- **Generalized eigen-decomposition** — whitens by the benign covariance and decomposes the malicious covariance in that space, isolating directions where malicious variance dominates.
- **Inference-only intervention** — no weight updates; hooks on MLP / layernorm outputs edit the residual stream at target layers.
- **Switchable fusion** — `sequential` / `parallel` / `adaptive` modes for combining visual and text corrections.

---

## Repository Structure

```text
CARE/
├── data/                # Built-in CSV data
├── src/                 # Hooked LVLM wrappers (Qwen & LLaVA)
├── steer/
│   ├── methods.py       # HybridSafetyProjectorV3 (core)
│   ├── act_qwen.py      # Qwen visual activation extraction
│   ├── act_qwen_text.py # Qwen text activation extraction
│   ├── act_one.py       # LLaVA-OneVision visual activation extraction
│   ├── act_one_text.py  # LLaVA-OneVision text activation extraction
│   └── eval_*.py        # Safety evaluation scripts
├── evaluation/          # Baseline / ablation scripts
├── PGD/                 # Adversarial image generation (PGD attack)
└── utility_eval/        # MM-Vet, MMBench, OR-Bench, SARR scripts
```

---

## Environment

```bash
conda create -n care python=3.10 -y && conda activate care
pip install torch torchvision   # match your CUDA version
pip install transformers accelerate bitsandbytes qwen-vl-utils \
  pandas numpy scipy scikit-learn pillow tqdm pyyaml \
  pyarrow requests matplotlib seaborn openai
```

Optional (Ascend NPU): `pip install torch-npu`

**Required models** (adjust paths in scripts before running):

- Qwen2.5-VL-7B-Instruct
- LLaVA-OneVision-1.5-8B-Instruct
- HarmBench-Llama-2-13b-cls (for safety classification)

> Scripts contain local absolute paths (`/root/fujinhu/...`). Replace them with your own paths before running.

---

## Data Sources

| Data / Asset | Source | Used For |
| --- | --- | --- |
| **AdvBench** & harmful corpus | [Visual-Adversarial-Examples-Jailbreak-LLMs / harmful_corpus](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/tree/main/harmful_corpus) | Malicious text prompts & attack images |
| **Visual adversarial examples** | [Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models) | PGD adversarial images, refusal behavior data |
| **FigStep** | [ThuCCSLab/FigStep](https://github.com/ThuCCSLab/FigStep) | SafeBench questions & images for malicious multimodal samples |
| **JailBreakV / JailbreakBench** | [abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model](https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model), [JailbreakBench/jailbreakbench](https://github.com/JailbreakBench/jailbreakbench) | JailBreakV_28K CSV & images, jailbreak evaluation protocol |
| **Refusal direction** | [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction) | Methodology reference for activation-space refusal steering |
| **MM-Vet** | [yuweihao/MM-Vet](https://github.com/yuweihao/MM-Vet) | General capability evaluation (mm-vet images & questions) |
| **OR-Bench** | [code in repo](./data/or-bench-hard-1k.csv) | Over-refusal benchmark |
| **MM-SafetyBench** | [original repo](https://github.com/isXinLiu/MM-SafetyBench) | Multi-modal safety benchmark (parquet files) |
| **COCO val2017** | [cocodataset.org](https://cocodataset.org) | Benign image samples for activation extraction |

---

## Quick Start

**1. Extract visual activations** (Qwen example):

```bash
python steer/act_qwen.py
# Saves to activations/qwen_visual_mi/
```

**2. Extract text activations**:

```bash
python steer/act_qwen_text.py
# Saves to activations/qwen_text_mi/
```

**3. Build projector & evaluate**:

```bash
python utility_eval/qwen_mmvet.py \
  --alpha_visual 4.5 --alpha_text 4.5 \
  --target_layers 12 13 14 \
  --n_components_visual 64 --n_components_text 64 \
  --fusion_mode adaptive --gpu_ids 0
```

At runtime, `HybridSafetyProjectorV3` loads the activations, builds visual/text safety subspaces per target layer, caches them under `cache/`, and registers forward hooks that project hidden states during inference.

---

## Method

Core: `steer/methods.py` — `HybridSafetyProjectorV3`.

**Subspace construction.** For each target layer and modality, given benign activations `H_ben` and malicious activations `H_mal`:

```text
Sigma_ben = cov(H_ben),  Sigma_mal = cov(H_mal)
A = Sigma_ben^{-1/2} @ Sigma_mal @ Sigma_ben^{-1/2}
```

Top-k eigenvectors of `A` form the malicious subspace `U_k`.

**Inference-time correction.** For hidden state `h` at a hooked layer:

```text
h' = h - alpha * P_mal(h - center) + alpha * P_mal(mu_ben - center)
```

where `alpha` is `alpha_visual` or `alpha_text`, and `P_mal = U_k @ U_k^T`.

**Fusion modes:**

| Mode | How visual & text corrections combine |
| --- | --- |
| `sequential` | Visual first, then text (cascaded) |
| `parallel` | Weighted sum: `w_vis * delta_vis + w_txt * delta_txt` |
| `adaptive` | Weights proportional to per-sample correction magnitude |

---

## Evaluation Scripts

Safety:

```bash
python steer/eval_mmsafety.py       # MM-SafetyBench
python steer/eval_pgd_jb.py         # PGD jailbreak (Qwen)
python steer/eval_mm_one.py          # MM-SafetyBench (LLaVA)
python steer/eval_pgd_jb_llava.py   # PGD jailbreak (LLaVA)
```

Utility:

```bash
python utility_eval/qwen_mmvet.py   # MM-Vet
python utility_eval/qwen_mmbench.py # MMBench
python utility_eval/qwen_orbench.py # OR-Bench (over-refusal)
python utility_eval/sarr.py         # SARR metric
```

Adversarial image generation:

```bash
python PGD/main_qwen.py --model-path /path/to/Qwen2.5-VL-7B-Instruct --n_iters 1000
```

---

## Key Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--target_layers` | [12, 13, 14] | transformer layers to hook |
| `--alpha_visual` | 4.5 | visual projection strength |
| `--alpha_text` | 4.5 | text projection strength |
| `--n_components_visual` | 64 | visual subspace dimension |
| `--n_components_text` | 64 | text subspace dimension |
| `--fusion_mode` | adaptive | `sequential` / `parallel` / `adaptive` |
| `--visual_text_ratio` | 0.5 | weight ratio for `parallel` mode |

---

## Acknowledgements

This project builds on data and ideas from:

- [FigStep](https://github.com/ThuCCSLab/FigStep)
- [Jailbreaking-Attack-against-Multimodal-Large-Language-Model](https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model)
- [refusal_direction](https://github.com/andyrdt/refusal_direction)
- [Visual-Adversarial-Examples-Jailbreak-LLMs](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models)
- [MM-Vet](https://github.com/yuweihao/MM-Vet)
- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)

We would like to sincerely thank the authors of **ASTRA** (https://github.com/ASTRAL-Group/ASTRA) for generously making their PGD attack implementation publicly available, which greatly facilitated our research.

