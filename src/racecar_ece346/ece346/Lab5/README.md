# Lab 5: Behavioral Cloning

**[Due 11:59PM Friday, April 10]**

In this lab, we will explore **behavioral cloning** (BC), a form of imitation learning where we train a neural network to mimic human driving. You will manually drive the truck using a joystick, record state-action pairs, train a policy network offline, and deploy it to drive the truck autonomously.

There are **4 tasks** in this lab, and you will need to submit (push) your code and demo to a TA.

**Note**: Make sure you have **pulled the code from upstream** into your repository and rebuilt the Docker image:
```bash
git pull upstream SP2026
./start.sh build
```

Then start the container and build the ROS 2 workspace:
```bash
./start.sh
# Inside the container:
cd /ros2_ws && colcon build --symlink-install && source install/setup.bash
```

## Background

Behavioral cloning is the simplest form of imitation learning. Given a dataset of expert demonstrations $\mathcal{D} = \{(s_i, a_i)\}_{i=1}^{N}$, we train a policy $\pi_\theta(s) \approx a$ by minimizing:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} w_i \| \pi_\theta(s_i) - a_i \|^2$$

where $w_i$ are optional per-sample weights. The policy is a multi-layer perceptron (MLP) that maps the truck's state to Ackermann control commands (speed and steering angle).

## Software Structure

This lab introduces two new ROS 2 nodes:

- **`data_recorder_node`** — Subscribes to `/SLAM/Pose` (odometry) and `/teleop` (joystick commands). Records synchronized state-action pairs and saves them to `.pkl` files.
- **`bc_eval_node`** — Loads a trained PyTorch model and publishes `AckermannDriveStamped` commands to `/drive`. The truck's `control_gate` forwards these commands when R2 is held on the joystick.

### State Vector (6D)
| Index | Name     | Description              |
|-------|----------|--------------------------|
| 0     | $X$      | Position X (m)           |
| 1     | $Y$      | Position Y (m)           |
| 2     | $\psi$   | Heading angle (rad)      |
| 3     | $v_x$    | Forward velocity (m/s)   |
| 4     | $v_y$    | Lateral velocity (m/s)   |
| 5     | $\omega$ | Rotation rate (rad/s)    |

### Action Vector (2D)
| Index | Name             | Description             |
|-------|------------------|-------------------------|
| 0     | speed            | Linear velocity (m/s)   |
| 1     | steering_angle   | Steering angle (rad)    |

## Before You Start

This lab runs on the **real truck**, not in simulation. Before launching any nodes, make sure:

1. The truck is powered on — battery connected, power switch flipped on, Jetson booted.
2. Your host computer is connected to the **same WiFi network** as the truck.
3. **SLAM is running** on the truck (Terminal 1 on Jetson):
   ```bash
   ssh nvidia@192.168.1.2XX
   cd ~/ece346_truck && ./start_slam.sh
   ```
4. **CycloneDDS + F1Tenth stack is running** on the truck (Terminal 2 on Jetson):
   ```bash
   ssh nvidia@192.168.1.2XX
   cd ~/ece346_truck
   source setup_cyclone.sh <HOST_IP> <DOMAIN_ID>
   ros2 launch f1tenth_stack bringup_launch.py
   ```
5. **Test manual control**: Hold L2 on the PS4 controller and verify the truck responds to joystick input before proceeding.

## Task 1: Data Collection

In this task, you will drive the truck manually and record training data.

### Launching the Recorder

On your host computer, inside the Docker container:
```bash
cd /ros2_ws && colcon build --symlink-install && source install/setup.bash
source setup_cyclone.sh <TRUCK_ID> <ROS_DOMAIN>
ros2 launch racecar_ece346 lab5_record_launch.py
```

### Recording Data

In a separate terminal, arm the recorder:
```bash
ros2 service call /learning/start_record std_srvs/srv/Empty
```

Now drive the truck with the joystick (**hold L2**). Data is recorded automatically while L2 is held and pauses when L2 is released. You can do multiple driving segments without calling the service again.

When you are done collecting data, save to disk:
```bash
ros2 service call /learning/save_data std_srvs/srv/Empty
```

Data is saved to `Lab5/data/bc_data_<timestamp>.pkl`. Each file contains NumPy arrays of states, actions, and timestamps.

**Tips for good data collection:**
- Aim for **2000+ samples** across multiple laps
- **Emphasize turns** — the model needs enough examples of turning behavior
- Drive consistently — the model will learn your average behavior
- Keep AprilTags visible to prevent SLAM from crashing

## Task 2: Implement the Neural Network

Before training, you must implement your policy network in [`scripts/neural_network.py`](scripts/neural_network.py). The `BCNetwork` class has three methods marked with `TODO` that you need to complete:

1. **`__init__`** — Define your network architecture using `nn.Sequential`, `nn.Linear`, and activation functions (e.g., `nn.ReLU`). Also define your optimizer (e.g., `torch.optim.Adam`) and loss function (e.g., `nn.MSELoss`).

2. **`forward`** — Pass the input tensor through your network and return the output.

3. **`train_step`** — Implement a single training step: convert inputs to tensors, forward pass, compute loss, backpropagate, and update weights. If per-sample weights `w` are provided, use weighted MSE: $\text{loss} = \text{mean}(w \cdot (\hat{y} - y)^2)$.

The `predict`, `train_epoch`, `save_model`, and `load_model` methods are provided — do not modify them.

**Hints:**
- Start simple (e.g., 2-3 hidden layers) and increase complexity if needed
- Think about what activation function to use (or not use) on the output layer
- Consider how many parameters your network has relative to your dataset size

**Resources:**
- [PyTorch: Learning the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) — Official tutorial covering tensors, datasets, and building neural networks
- [PyTorch: Build the Neural Network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) — Defining models with `nn.Module` and `nn.Sequential`
- [PyTorch: Optimizing Model Parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — Training loops, loss functions, and optimizers
- [Stanford CS231n: Neural Networks](https://cs231n.github.io/neural-networks-1/) — Architectures, activation functions, and best practices

## Task 3: Offline Training

Open the Jupyter notebook [`notebooks/train_bc.ipynb`](notebooks/train_bc.ipynb) inside the Docker container or on your host machine.

#### Running the Notebook inside Docker

1. Make sure your Docker container is running (via `./start.sh`).
2. Attach VS Code to the running container using **Dev Containers: Attach to Running Container**.
3. In the new VS Code window, navigate to `/ros2_ws/src/racecar_ece346/ece346/Lab5/notebooks/`.
4. Open `train_bc.ipynb` and select the **Python 3.8** kernel.

The notebook will:
1. Load all `.pkl` files from `Lab5/data/`
2. Visualize the recorded trajectories and data distributions
3. Normalize features and compute per-sample weights (turns are weighted higher)
4. Train your network with an 85/15 train/validation split
5. Plot loss curves and predicted vs. true actions
6. Save `bc_model.pt` and `normalization.pkl` to `Lab5/models/`

### Important: Sample Weighting

Your training data will likely contain far more straight-driving samples than turning samples. Without correction, the model will optimize for the common case (going straight) and **understeer at corners**. The notebook computes per-sample weights based on steering magnitude — samples with higher `|steering_angle|` get higher weight in the loss function, forcing the model to pay more attention to turns.

The key parameter is `turn_weight` in the train/val split cell:
```python
turn_weight = 1.0  # how much more to weight turns vs straight
```
Higher values = more emphasis on turns. If your truck understeers, try increasing this.

### Parameters to Tune in the Notebook

| Parameter | Cell | What it does |
|-----------|------|-------------|
| `feature_indices` | Prepare Features | Which state features to use as input (default: all 6) |
| `turn_weight` | Train/Val Split | How much more to weight turning samples (default: 1.0) |
| `hidden_sizes` | Train Model | Network layer sizes — change this in your `neural_network.py` |
| `learning_rate` | Train Model | Optimizer step size (default: 1e-3) |
| `n_epochs` | Train Model | Number of training passes (default: 75) |
| `batch_size` | Train Model | Samples per gradient update (default: 64) |

**Tips:**
- If validation loss starts rising while training loss drops, you are **overfitting** — reduce `n_epochs` or collect more data.
- If both losses plateau high, your network may be **too small** — add more layers or neurons.
- Always **restart the kernel** and run all cells after changing `neural_network.py`.

## Task 4: Deployment and Evaluation

### Configure the Eval Node

Update the model paths in [`config/lab5_eval.yaml`](config/lab5_eval.yaml):
```yaml
bc_eval_node:
  ros__parameters:
    model_path: "/ros2_ws/src/racecar_ece346/ece346/Lab5/models/bc_model.pt"
    norm_path: "/ros2_ws/src/racecar_ece346/ece346/Lab5/models/normalization.pkl"
    max_speed: 0.4
    max_steering: 0.34
```

### Launch the Eval Node

```bash
cd /ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch racecar_ece346 lab5_eval_launch.py
```

In a separate terminal:
```bash
ros2 service call /learning/start_eval std_srvs/srv/Empty
```

**Hold R2** on the joystick to enable autonomous driving. Release R2 at any time to stop the truck immediately.

To stop the model:
```bash
ros2 service call /learning/stop_eval std_srvs/srv/Empty
```

### Safety

The `control_gate` on the truck ensures safety:
- **R2 held** — autonomous commands are forwarded to the motors
- **R2 released** — truck stops immediately (zero velocity)
- **L2 held** — manual override (teleop mode)
- **Both or neither** — truck stops

### Iterating

If the truck understeers or behaves poorly:
1. Record more data, especially around problem areas
2. Retrain in the notebook (restart kernel, run all cells)
3. Relaunch the eval node — no rebuild needed since model files are loaded from the source directory

## Submission

1. Demo to a TA: truck driving autonomously using your trained behavioral cloning model
2. Show your training notebook with loss curves and prediction plots
3. Push your completed code to your repository
