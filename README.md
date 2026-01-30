# CogniPlan: Predict Layouts Before Plan

### [Paper](https://arxiv.org/pdf/2508.03027) | [Project Page](https://yizhuo-wang.com/cogniplan/)

> CogniPlan: Uncertainty-Guided Path Planning with Conditional Generative Layout Prediction


This branch hosts the ROS simulation for CogniPlan's exploration tasks. It implements a LiDAR-based 2D exploration planner that leverages learned conditional layout imagination.
The code is built upon [ARiADNE-ROS-Planner](https://github.com/marmotlab/ARiADNE-ROS-Planner).
We tested the code with ROS noetic on Ubuntu 20.04 ([installation](https://wiki.ros.org/noetic/Installation/Ubuntu)).

## Setup

### Prerequisites

We use [OctoMap](https://octomap.github.io/) for occupancy mapping:

```bash
sudo apt-get install ros-noetic-octomap
```

You will also need to install `torch (cpu), torchvision, scikit-learn, matplotlib`.
An isolated environment is recommended.
You can refer to the following commands:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib
```

Then, clone this repo, checkout noetic branch, and compile.

```bash
git clone https://github.com/marmotlab/CogniPlan
cd CogniPlan
git checkout noetic
catkin_make
```

### Checkpoints

Download the model checkpoints using the scripts below. 
It will unpack them to the corresponding directories automatically.

```bash
bash src/scripts/model/download.sh
```

You can also manually download and unpack the files from our release page.
The file structure should look like this:

```bash
├── ...
└── src
    ├── CMakeLists.txt
    ├── package.xml
    ├── launch
    │   ├── rl_planner.launch
    │   ├── ...
    ├── rviz
    │   └── rviz.rviz
    └── scripts
        ├── model
        │   ├── checkpoint.pth
        │   ├── config.yaml
        │   ├── download.sh
        │   └── generator.pt
        ├── agent.py
        ├── ...

```

### Run

We validate the planner in CMU exploration environment: https://www.cmu-exploration.com/. 
Please follow the instructions on their page to set up the simulation environment, where they provide different large scale environments, SLAM, local planner, etc.
Our package will output the waypoints to their local planner for execution.

To run the simulation, go to their development directory in a terminal and run:

```bash
source devel/setup.bash && roslaunch vehicle_simulator system_indoor.launch
```

In another terminal, go to our package directory and run:

```bash
source devel/setup.bash && roslaunch rl_planner rl_planner.launch
```

We also provide launch files in CMU forest and one [HPHS](https://github.com/bit-lsj/HPHS) environment.

**Known issue:** The test maps are a bit OOD compared to the data our inpainting module was trained on. 
For improved performance, retraining the module with more domain-relevant data may be necessary. 
We might release more generalizable checkpoints in the future to help mitigate this domain shift.

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
