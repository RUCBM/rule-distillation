# Rule Distillation
Code&Data for the paper **[Distilling Rule-based Knowledge into Large Language Models](https://arxiv.org/pdf/2311.08883)** [COLING 2025]

## Installation
Our training and evaluation code is mainly based on the open-sourced training platform [alpaca-lora](https://github.com/tloen/alpaca-lora). Users can follow the instructions in [alpaca-lora](https://github.com/tloen/alpaca-lora) to prepare the experimental environment.

## Usage

### Data
We provide all the training, validation and testing data for all 3 tasks in each k-shot setting under the folder ```data/```. In each file's name, ''_full'' represents that each task sample in this file contains both the input-output pair and the rule description, while ''_no'' represents that each task sample in this file does not contain the textual rule but only has an input-output pair.

### Code
Please first download the source code of [alpaca-lora](https://github.com/tloen/alpaca-lora) into local. We provide the core source code for instruction tuning (w/ and w/o textual rule), rule distillation and inference in ```finetune.py```, ```distill.py```, ```inference.py```, respectively. Users can put these 3 files into the downloaded [alpaca-lora](https://github.com/tloen/alpaca-lora) folder for further usage. Also, we provide examples of command lines (along with explanations for some key arguments) for running our experiments in ```examples.sh```.

## Acknowledgments
We sincerely thank the contributors of [alpaca-lora](https://github.com/tloen/alpaca-lora) for open-sourcing the platform.

## Citation
If you find our code and data useful, please kindly cite as
```
@misc{yang2024distilling,
      title={Distilling Rule-based Knowledge into Large Language Models}, 
      author={Wenkai Yang and Yankai Lin and Jie Zhou and Ji-Rong Wen},
      year={2024},
      eprint={2311.08883},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2311.08883}, 
}
```

or

```
@article{yang2023enabling,
  title={Enabling Large Language Models to Learn from Rules},
  author={Yang, Wenkai and Lin, Yankai and Zhou, Jie and Wen, Jirong},
  journal={arXiv preprint arXiv:2311.08883},
  year={2023}
}
```