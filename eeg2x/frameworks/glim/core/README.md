## GLIM:  Learning Interpretable Representations Leads to Semantically Faithful EEG-to-Text Generation<br><sub>Official PyTorch Implementation</sub>

[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b?style=flat&logo=arxiv
)](https://arxiv.org/abs/2505.17099)
[![wandb](https://img.shields.io/badge/wandb-full%20generated%20texts-FFBE00?style=flat&logo=weightsandbiases
)](https://wandb.ai/mind-reading/glim-iclr/reports/GLIM-generation-samples--VmlldzoxMjc0Njg1NQ?accessToken=5uqxxv6ug80naqfqlni2xvxa8y8l7u6ouc1cgjt0naxk1g8g0h9lgyf8r0e97xyk)
[![figshare](https://img.shields.io/badge/figshare-download%20checkpoint-62c2b8?style=flat&logo=figshare)](https://doi.org/10.6084/m9.figshare.29115161.v1)

This repository contains:

* ⚡ A modular [implementation](model/glim.py) of GLIM, organized with PyTorch Lightning.
* ✂️ Complete data [preprocessing](data/__STEP1_text_extract_revise.ipynb) notebooks.
* 🚆 Simple [training](train.py) and [test](test.py) scripts.
* 🧭 Semantic [classification](predict_corpus.ipynb) notebooks that align with the results shown in the paper.
* 🗒️ All text samples generated with [GLIM](results/wandb_export_gen_samples_glim.csv), its [noise-input test](results/wandb_export_gen_samples_noise_input_test.csv) and [prompt-free test](results/wandb_export_gen_samples_prompt_free_test.csv).

<!-- More resources available:

* 📍 Model checkpoint of GLIM at [figshare](https://figshare.com/articles/preprint/glim-zuco-epoch_199-step_49600_ckpt/29115161).
* 📋 Full generated samples in an interactive [wandb report](https://wandb.ai/mind-reading/glim-iclr/reports/GLIM-generation-samples--VmlldzoxMjc0Njg1NQ?accessToken=5uqxxv6ug80naqfqlni2xvxa8y8l7u6ouc1cgjt0naxk1g8g0h9lgyf8r0e97xyk). -->
### TL;DR
[Are EEG-to-text models working?](https://github.com/NeuSpeech/EEG-To-Text) 🤔 ⟶ **YES** ! 👇👇👇

### Abstract

<img src="figs/intro.png" alt="GLIM overview" width="560"/>  

Pretrained generative models have opened new frontiers in brain decoding by enabling the synthesis of realistic texts and images from non-invasive brain recordings. However, the reliability of such outputs remains questionable—whether they truly reflect semantic activation in the brain, or are merely hallucinated by the powerful generative models. 
In this paper, we focus on EEG-to-text decoding and address its hallucination issue through the lens of posterior collapse. Acknowledging the underlying mismatch in information capacity between EEG and text, we reframe the decoding task as semantic summarization of core meanings rather than previously verbatim reconstruction of stimulus texts. 
To this end, we propose the Generative Language Inspection Model (GLIM), which emphasizes learning informative and interpretable EEG representations to improve semantic grounding under heterogeneous and small-scale data conditions. 
Experiments on the public ZuCo dataset demonstrate that GLIM consistently generates fluent, EEG-grounded sentences without teacher forcing. Moreover, it supports more robust evaluation beyond text similarity, through EEG-text retrieval and zero-shot semantic classification across sentiment categories, relation types, and corpus topics. Together, our architecture and evaluation protocols lay the foundation for reliable and scalable benchmarking in generative brain decoding.

### Model architecture
<img src="figs/method.png" alt="Model architecture" width="560"/>  


### Representative generation examples
<img src="figs/text_samples.png" alt="Representative examples" width="560"/>  

You can find full generated samples in this [interactive report](https://wandb.ai/mind-reading/glim-iclr/reports/GLIM-generation-samples--VmlldzoxMjc0Njg1NQ?accessToken=5uqxxv6ug80naqfqlni2xvxa8y8l7u6ouc1cgjt0naxk1g8g0h9lgyf8r0e97xyk) or in the [`results/`](results/) directory.

## Setup
- Run `conda env create -f environment.yml` to create the environment.
- Download the ZuCo dataset, including versions [1.0](https://osf.io/q3zws/) and [2.0](https://osf.io/2urht/). 

  <details>
  <summary> 💡 You can just download part of the files and organize them as the following structure.</summary>


  ```
  data/
  ├── raw_data/
  │   ├── 🌐 ZuCo1/                   ## see https://osf.io/q3zws/files/osfstorage
  │   │   ├── ☑️ task_materials/      ## download texts and lables
  │   │   ├── task1- SR/
  │   │   │   └── ✅ Matlab files/    ## download sentence-level EEG segments
  │   │   ├── task2 - NR/
  │   │   │   └── ✅ Matlab files/
  │   │   └── task3 - TSR/
  │   │       └── ✅ Matlab files/
  │   └── 🌐 ZuCo2/                   ## see https://osf.io/2urht/files/osfstorage
  │       ├── ☑️ task_materials/
  │       ├── task1 - NR/
  │       │   └── ✅ Matlab files/
  │       └── task2 - TSR/
  │           └── ✅ Matlab files/

  ```
  </details>

## Data preprocessing
You can either  
  - run all four preprocessing notebooks step by step; or just 
  - start from [STEP3](data/__STEP3_eeg_preproc.ipynb) with this [label table](data/tmp/zuco_label_8variants.df) to skip generating text variants.

## Reproduce our results
- Download the [model checkpoint](https://doi.org/10.6084/m9.figshare.29115161.v1) and put it in [`checkpoints/`](checkpoints/).
- Run [`test.py`](test.py) to generate sentences and compute overall metrics, with one single GPU.
- Run each `predict_xxx.ipynb` to reproduce the classification results (with both CLIP-like and LLM-assisted approaches).

## Train from scratch
- Run [`train.py`](train.py) with default parameters (except for those assosiated with your devices and directories).

## BibTeX

  ```bibtex
  @article{liu2025glim,
    title={Learning Interpretable Representations Leads to Semantically Faithful EEG-to-Text Generation},
    author={Xiaozhao Liu and Dinggang Shen and Xihui Liu},
    year={2025},
    journal={arXiv preprint arXiv:2505.17099},
  }
  ```

## License
⚖️ GLIM © 2025 by the repository owner is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

