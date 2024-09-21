<p align="center">
<img src="assets/logo.jpg" width="30%"> <br>
</p>

<div align="center">
<h1>Famba-V: Fast Vision Mamba with Cross-Layer Token Fusion</h1>
</div>
<div align="center">
<a href="https://opensource.org/licenses/Apache-2.0">
  <img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-4E94CE.svg">
</a>
</div>

## Introduction
  
> **[Famba-V: Fast Vision Mamba with Cross-Layer Token Fusion](https://arxiv.org/abs/2409.09808)** [[arXiv]](https://arxiv.org/abs/2409.09808)   
> *Hui Shen, Zhongwei Wan, Xin Wang, Mi Zhang*   
> *The Ohio State University*
> *ECCV 2024 Workshop on Computational Aspects of Deep Learning (Best Paper Award Candidate)*

## Abstract
Mamba and Vision Mamba (Vim) models have shown their potential as an alternative to methods based on Transformer architecture. This work introduces Fast Mamba for Vision (Famba-V), a cross-layer token fusion technique to enhance the training efficiency of Vim models. The key idea of Famba-V is to identify and fuse similar tokens across different Vim layers based on a suit of cross-layer strategies instead of simply applying token fusion uniformly across all the layers that existing works propose. We evaluate the performance of Famba-V on CIFAR-100. Our results show that Famba-V is able to enhance the training efficiency of Vim models by reducing both training time and peak memory usage during training. Moreover, the proposed cross-layer strategies allow Famba-V to deliver superior accuracy-efficiency trade-offs. These results all together demonstrate Famba-V as a promising efficiency enhancement technique for Vim models.


## Quick Start

- Python 3.10.13

  - `conda create -n your_env_name python=3.10.13`

- torch 2.1.1 + cu118
  - `pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`

- Requirements: vim_requirements.txt
  - `pip install -r fambav/vim_requirements.txt`

- Install ``causal_conv1d`` and ``mamba``
  - `pip install -e causal_conv1d>=1.1.0`
  - `pip install -e mamba-1p1p1`
  
  

## Train Your Famba-V with Upper-layer Fusion Strategy
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 128 --drop-path 0.0 --weight-decay 0.1 --num_workers 25 --data-set CIFAR --data-path ./datasets/cifar-100-python --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --no_amp --fusion-strategy upper --fusion-layer 4 --fusion-token 8
```
## :heart: Acknowledgement 
This project is based on Vision Mamba ([paper](https://arxiv.org/abs/2401.09417), [code](https://github.com/hustvl/Vim?tab=readme-ov-file)), Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Causal-Conv1d ([code](https://github.com/Dao-AILab/causal-conv1d)), DeiT ([paper](https://arxiv.org/abs/2012.12877), [code](https://github.com/facebookresearch/deit)). Thanks for their wonderful works.

## 🥳 Citation
If you find Famba-V is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@inproceedings{fambav2024eccvw,
    title={Famba-V: Fast Vision Mamba with Sparse Fusion-based Visual Representation},
    author={Shen, Hui and Wan, Zhongwei and Wang, Xin and Zhang, Mi},
    booktitle={European Conference on Computer Vision (ECCV) Workshop on Computational Aspects of Deep Learning},
    year={2024}
}
```
