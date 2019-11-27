#!/usr/bin/env python3

"""
This script will train a CNN model using imitation learning from a PurePursuit Expert.
"""

import time
import random
import argparse
import math
import json
from functools import reduce
import operator

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from utils.env import *
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from utils.teacher import PurePursuitExpert

from imitation.pytorch.model import Model

MAP_NAMES = ["loop_empty", "small_loop", "loop_obstacles", "loop_pedestrians"]
TRAINING_DATA_PATH = "../data/{}/{}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _train(args):
    env = launch_env1()
    env1 = ResizeWrapper(env)
    env2 = NormalizeWrapper(env) 
    env3 = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    def transformObs(obs):
        obs = env1.observation(obs)
        obs = env2.observation(obs)
        obs = env3.observation(obs)
        return obs

    actions = None
    rawObs = None
    for map in MAP_NAMES:
        if map == "loop_obstacles":
            episodes = 3
        else:
            episodes = 2

        print(map)
        for episode in range(episodes):
            actionFile = "actions_{}.npy".format(episode)
            action = np.load(TRAINING_DATA_PATH.format(map, actionFile))
            print(action.shape)

            observationFile = "obs_{}.npy".format(episode)
            observation = np.load(TRAINING_DATA_PATH.format(map, observationFile))

            if actions is None:
                actions = action
                rawObs = observation
            else:
                actions = np.concatenate((actions, action), axis=0)
                rawObs = np.concatenate((rawObs, observation), axis=0)
            print(actions.shape)
        print(actions.shape)
        print("---")

    observations = np.zeros((rawObs.shape[0], 3, 160, 120))
    for i, obs in enumerate(rawObs):
        observations[i] = transformObs(obs)

    
    '''
    # Create an imperfect demonstrator
    expert = PurePursuitExpert(env=env)

    observations = []
    actions = []

    # let's collect our samples
    for episode in range(0, 2):
    #for episode in range(0, args.episodes):
        print("Starting episode", episode)
        #for steps in range(0, args.steps):
        for steps in range(0, 4):
            # use our 'expert' to predict the next action.
            action = expert.predict(None)
            observation, reward, done, info = env.step(action)
            observations.append(observation)
            actions.append(action)
        env.reset()

    actions = np.array(actions)
    observations = np.array(observations)
    print(observations.shape)
    '''

    env.close()
    #raise Exception("Done with testing")

    model = Model(action_dim=2, max_action=1.)
    model.train().to(device)

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.0004,
        weight_decay=1e-3
    )

    loss_list = []
    avg_loss = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))
        obs_batch = torch.from_numpy(observations[batch_indices]).float().to(device)
        act_batch = torch.from_numpy(actions[batch_indices]).long().to(device)

        model_actions = model(obs_batch)

        loss = (model_actions - act_batch).norm(2).mean()
        loss.backward()
        optimizer.step()

        #loss = loss.data[0]
        loss = loss.item()
        avg_loss = avg_loss * 0.995 + loss * 0.005

        print('epoch %d, loss=%.3f' % (epoch, loss))
        loss_list.append(loss)

        # Periodically save the trained model
        if epoch % 50 == 0:
            print("Saving...")
            torch.save(model.state_dict(), 'imitation/pytorch/models/imitate.pt')
            save_loss(loss_list, 'imitation/pytorch/loss.npy')

    print("Saving...")
    torch.save(model.state_dict(), 'imitation/pytorch/models/imitate.pt')

def save_loss(loss_list, loss_path):
    # Saves the training loss to a file
    loss = np.array(loss_list)
    np.save(loss_path, loss)
    print('loss saved to %s' % loss_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int, help="Sets Gym, TF, and Numpy seeds")
    parser.add_argument("--episodes", default=3, type=int, help="Number of epsiodes for experts")
    parser.add_argument("--steps", default=50, type=int, help="Number of steps per episode")
    parser.add_argument("--batch-size", default=32, type=int, help="Training batch size")
    parser.add_argument("--epochs", default=100000, type=int, help="Number of training epochs")
    parser.add_argument("--model-directory", default="models/", type=str, help="Where to save models")

    args = parser.parse_args()

    _train(args)