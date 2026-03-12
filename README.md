<div align="center">
<h1>🚀 On-Policy Supervised Fine-Tuning for Efficient Reasoning</h1>
</div>

<p align="center">
<a href="https://arxiv.org/abs/2602.13407" target="_blank"><img src="https://img.shields.io/badge/arXiv-2602.13407-DA644E?logo=arxiv" alt="arXiv"></a>
<a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
<a href="https://github.com/EIT-NLP/On-Policy-SFT" target="_blank"><img src="https://img.shields.io/badge/GitHub-Code-white?logo=github" alt="GitHub"></a>
<img src="https://img.shields.io/github/last-commit/EIT-NLP/On-Policy-SFT?logo=github&color=orange" alt="Last Commit">
</p>

<table>
  <tr>
    <td style="width: 45%;"><img src="figures/paper_fig1.png" alt="paper_figure_1" style="width: 100%;"></td>
    <td style="width: 45%;"><img src="figures/paper_fig2.png" alt="paper_figure_2" style="width: 100%;"></td>
  </tr>
</table>

## 🔥 News <a id="news"></a>

- **[2026.03.12]** GitHub code release: [EIT-NLP/On-Policy-SFT](https://github.com/EIT-NLP/On-Policy-SFT).
- **[2026.02.13]** arXiv preprint release: [On-Policy Supervised Fine-Tuning for Efficient Reasoning](https://arxiv.org/abs/2602.13407).

> [!NOTE]
> This repository focuses on on-policy SFT training, a simple yet effective method that can reduce CoT length by up to **80**% while maintaining original accuracy, surpassing more complex RL-based methods.

## 🛠️ Setup

```sh
git clone https://github.com/EIT-NLP/On-Policy-SFT
cd On-Policy-SFT/opsft

conda create -n opsft python=3.10
conda activate opsft

pip install -r requirements.txt
pip install flash_attn==2.7.4.post1 --no-build-isolation  # choose a suitable version for your own machine
pip install -e . --no-dependencies
```

> [!NOTE]
> Installation behavior can vary across different machines and CUDA environments. If needed, please adapt package versions accordingly.

## 🚀 Run

Before running the script below, please ensure you are in the project root directory (`On-Policy-SFT/opsft`).

```sh
conda activate opsft

# for on-policy SFT training
bash examples/On_Policy_SFT.sh
```

## 🗂️ Dataset

The datasets are located in `opsft/data`.
- Training sets: `deepscaler` (DSR), `OpenThoughts3-1.2M/math_question` (OpenThoughts math-only), and `gsm8k`.
- Benchmarks: files in `opsft/data/benchmarks`.

## 🎯 On-Policy SFT Built upon VERL

We mainly modify the training objective, trainer workflow, validation protocol, and verifier integration to support efficient on-policy SFT.

### 🧠 Training Core

- `recipe/On_Policy_SFT/on_policy_sft_trainer`: On-policy SFT training logic.
- `recipe/On_Policy_SFT/dp_actor`: Switched from policy-gradient loss to cross-entropy loss.

### ✅ Evaluation Support

- `verl/trainer/ppo/metric_utils.py`: Added pass@k calculation during validation.
- `verl/trainer/ppo/ray_trainer.py`: In `_validate`, selects 16 question-response pairs for each validation dataset.

### 🧪 Verifier

- `verl/utils/reward_score/__init__.py`

## 🙏 Source Acknowledgement

This project is implemented on top of the [VERL](https://github.com/volcengine/verl) codebase at commit `38d9a88170786a45cb189a08290c4651e6d6f671`.

For verifier, we use [HuggingFace Math-Verify](https://github.com/huggingface/Math-Verify).

## 📌 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@misc{zhao2026onpolicysupervisedfinetuningefficient,
      title={On-Policy Supervised Fine-Tuning for Efficient Reasoning},
      author={Anhao Zhao and Ziyang Chen and Junlong Tong and Yingqi Fan and Fanghua Ye and Shuhao Li and Yunpu Ma and Wenjie Li and Xiaoyu Shen},
      year={2026},
      eprint={2602.13407},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.13407},
}
```


