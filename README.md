# RL_Final_Project
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

# RL_Final_Project

Final project on application of double dqn learning to the cart-pole problem, for the Reinforcement Learning course.

## Setup

To clone this repository, you need to have `git` installed. After that you can open a terminal window and run

```
git clone https://github.com/lorenzoelia/RL_Final_Project
```

Move to the project folder

```
cd RL_Final_Project
```

Now you need to install the dependencies. Make sure you have `pip` installed and then run

```
pip install -r requirements.txt
```

To run the program, do the following

```
cd src
```

and finally

```
python main.py
```

## Directories

- [`src`](https://github.com/lorenzoelia/RL_Final_Project/tree/master/src) is the basis
  package containing all Python code. Inside there are:
    - Package [`baseline`](https://github.com/lorenzoelia/RL_Final_Project/tree/master/src/baseline),
      which contains the baseline approach of random walk used for comparing Q-Learning and Double DQN effectiveness
    - Package [`dqn_learning_v0`](https://github.com/lorenzoelia/RL_Final_Project/tree/master/src/dqn_learning_v0), which contains the classes necessary to train the DQN with Double Q-Learning. Folder [`models_v0`](https://github.com/lorenzoelia/RL_Final_Project/tree/master/src/dqn_learning_v0/models_v0) contains the trained DQN agents' weight
    - Package [`pole_test`](https://github.com/lorenzoelia/RL_Final_Project/tree/master/src/pole_test), to test the execution of the CartPole environment from Gym
    - Package [`q_learning`](https://github.com/lorenzoelia/RL_Final_Project/tree/master/src/q_learning), which contains the files for plain q-learning and testing