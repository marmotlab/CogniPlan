# CogniPlan: Predict Layouts Before Plan

### [Paper](https://arxiv.org/pdf/2508.03027) | [Project Page](https://yizhuo-wang.com/cogniplan/)

> CogniPlan: Uncertainty-Guided Path Planning with Conditional Generative Layout Prediction


## News / ToDo

- [ ] Release ROS simulation code.
- [x] [9 Sep 2025] Release code and model for navigation ([navigation branch](https://github.com/marmotlab/CogniPlan/tree/navigation)).
- [x] [6 Sep 2025] Release code and model for exploration ([main branch](https://github.com/marmotlab/CogniPlan/tree/main)).
- [x] [4 Aug 2025] CogniPlan is accepted to CoRL 2025!

## Setup

### Environment

We use conda/mamba to manage the environment.
The required packages are listed below.
We have tested multiple versions without major issues, so you may adjust them as needed.

```bash
conda create -n cogniplan python=3.12 scikit-learn scikit-image imageio pandas tensorboard matplotlib  # additional scikit-learn required
conda activate cogniplan
pip install torch torchvision opencv-python ray wandb
```

### Checkpoints and Datasets

Clone this repository and checkout the navigation branch.

```bash
git clone https://github.com/marmotlab/CogniPlan
cd CogniPlan
git checkout navigation
```

Download the checkpoints and datasets using the scripts below.
It will unpack them to the corresponding directories automatically.

```bash
bash dataset/download.sh
```

You can also manually download and unpack the files from our release page.


### Training

Set parameters in `planner/parameter.py` as needed, and run:

```bash
python -m planner.driver
```

### Evaluation

To evaluate our pre-trained model, run:

```bash
python -m planner.test_driver
```

> Note: If you want to debug in PyCharm, edit the run configuration kind from script to **module**, 
> and set `planner.xxx` or `mapinpaint.xxx` as the module name.
> Set the working directory to the **root** of this repository.


## Citation

```bibtex
@inproceedings{wang2025cogniplan,
  author={Wang, Yizhuo and He, Haodong and Liang, Jingsong and Cao, Yuhong and Chakraborty, Ritabrata and Sartoretti, Guillaume},
  title={CogniPlan: Uncertainty-Guided Path Planning with Conditional Generative Layout Prediction},
  booktitle={Conference on Robot Learning},
  year={2025},
  organization={PMLR}
}
```

### Authors
[Yizhuo Wang](https://www.yizhuo-wang.com/),
[Haodong He](https://hehaodong2004.github.io/),
[Jingsong Liang](https://jingsongliang.com/),
[Yuhong Cao](https://www.yuhongcao.online/),
[Ritabrata Chakraborty](https://in.linkedin.com/in/ritabrata-chakraborty-a63268251/),
[Guillaume Sartoretti](https://cde.nus.edu.sg/me/staff/sartoretti-guillaume-a/)
