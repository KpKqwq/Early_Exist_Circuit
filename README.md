# NEAT: Neuron-Based Early Exit for Large Reasoning Models

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2602.02010-b31b1b.svg)](https://arxiv.org/abs/2602.02010)
[![Code](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/KpKqwq/Early_Exist_Circuit)

Official repository for our paper **NEAT: Neuron-Based Early Exit for Large Reasoning Models**.

## News

- **[2026-04]** Our paper has been accepted 🎉
- **[2026-02]** We released the paper on arXiv: [NEAT: Neuron-Based Early Exit for Large Reasoning Models](https://arxiv.org/abs/2602.02010)

## Overview

Large Reasoning Models (LRMs) often suffer from **overthinking**, where redundant reasoning steps are generated after the model has already reached a correct solution.

We propose **NEAT** (**N**euron-based **E**arly re**A**soning exi**T**), a **training-free** early exit framework that monitors **neuron-level activation dynamics** to detect exit signals during reasoning. NEAT can dynamically:

- trigger early exit when the model has likely reached a reliable solution,
- suppress unnecessary reflection,
- reduce redundant reasoning tokens,
- preserve solution quality without introducing additional test-time rollout computation.

Across four reasoning benchmarks and six models with different scales and architectures, **NEAT reduces the average number of generated tokens by 22%–28% while maintaining accuracy**.

## Repository Structure

```text
Early_Exist_Circuit/
├── README.md
├── identify_neurons.py
├── neuro_early_generate.py
└── utils/
    ├── __init__.py
    ├── neuro_identify_utils.py
    └── neuron_utils.py
```

## What is included in this repository

This repository currently includes:

- **Neuron identification code** for finding exit-associated neurons
- **Neuron-based early generation / early exit code**
- Utility functions used by the above pipelines

## Requirements

Please prepare your environment with Python and the dependencies required by the scripts.

A typical setup may look like:

```bash
conda create -n neat python=3.10 -y
conda activate neat
pip install -r requirements.txt
```

> Note: You may need to additionally install packages required by your local setup, such as `transformers`, `torch`, `vllm`, `nnsight`, `einops`, `matplotlib`, and related dependencies.

## Usage

### 1. Identify exit-associated neurons

Use `identify_neurons.py` to identify neurons relevant to early reasoning exit:

```bash
python identify_neurons.py \
  --model_path /path/to/model \
  --output_data
```

### 2. Run neuron-based early exit generation

Use `neuro_early_generate.py` to perform generation with neuron-based early exit:

```bash
python neuro_early_generate.py \
  --model_path /path/to/model \
  --data_path /path/to/data.json \
  --output_file /path/to/output.jsonl
```

## Method

NEAT identifies **exit-associated neurons** and tracks their activation patterns during reasoning. Based on these neuron-level signals, it decides whether to:

1. continue reasoning,
2. stop generation early,
3. suppress unnecessary reflection.

This enables efficient reasoning without requiring an additional probing model, extra rollout computation, or externally labeled datasets.

## Current Status

- [x] Paper released
- [x] Code released
- [ ] Add dependency list / `requirements.txt`
- [ ] Add benchmark preparation instructions
- [ ] Add pretrained neuron sets / checkpoints
- [ ] Add full reproduction scripts
- [ ] Add examples and expected outputs

## Citation

If you find this repository useful, please cite:

```bibtex
@article{liu2026neat,
  title={NEAT: Neuron-Based Early Exit for Large Reasoning Models},
  author={Liu, Kang and Liu, Yongkang and Yang, Xiaocui and Wang, Peidong and Zhang, Wen and Feng, Shi and Zhang, Yifei and Wang, Daling},
  journal={arXiv preprint arXiv:2602.02010},
  year={2026}
}
```

## Acknowledgement

If this project is helpful to your research, please consider giving this repository a star.

## Contact

For questions or collaborations, please open an issue or contact the authors.
