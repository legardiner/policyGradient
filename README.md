# Policy Gradient

Tensorflow implementation of policy gradient reinforcment learning with baseline for two games from [OpenAI Gym](https://gym.openai.com/): cartpole and pong. 

# Getting Started

## Train Models

To run the games with the default hyperparameters, use the following commmands and specify a `run_num` to create a new log directory:

```
python cartpole.py --run_num 1
```

```
python pong.py --run_num 1
```

The default hyperparameters were selected through experimentation, but can be adjusted by adding arguments to a game launch.

## Visualize Training

To visualize the average total episode reward at each epoch, launch tensorboard with the following command:

```
tensorboard --logdir=logs/pong/[run_num]
```
