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

from imitation.pytorch.model import *

MAP_NAMES = ["loop_empty", "small_loop", "loop_obstacles", "loop_pedestrians"]
TRAINING_DATA_PATH = "./dagger/{}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _train(args):
    actions = None
    observations = None
    for map in MAP_NAMES:
        print(map)
        actionFile = "actions_{}.npy".format(map)
        action = np.load(TRAINING_DATA_PATH.format(actionFile))
        print(action.shape)

        observationFile = "obs_{}.npy".format(map)
        observation = np.load(TRAINING_DATA_PATH.format(observationFile))

        if actions is None:
            actions = action
            observations = observation
        else:
            actions = np.concatenate((actions, action), axis=0)
            observations = np.concatenate((observations, observation), axis=0)
        print("---")

    model = Model(action_dim=2, max_action=1.)
    try:
        state_dict = torch.load('./models/imitate.pt')
        model.load_state_dict(state_dict)
    except:
        print('failed to load model')
        exit()
    model.train().to(device)

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.0004,
        weight_decay=1e-3
    )

    length = min(len(actions), len(observations))
    actions = actions[:length, :]
    observations = observations[:length, :, :, :]
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
            torch.save(model.state_dict(), 'imitation/pytorch/models/dagger_imitate.pt')
            save_loss(loss_list, 'imitation/pytorch/dagger_loss.npy')

    print("Saving...")
    torch.save(model.state_dict(), 'imitation/pytorch/models/dagger_imitate.pt')

def _windowtrain(args):
    actions = None
    rawObs = None
    for map in MAP_NAMES:
        print(map)
        actionFile = "windowactions_{}.npy".format(map)
        action = np.load(TRAINING_DATA_PATH.format(actionFile))
        print(action.shape)

        observationFile = "windowobs_{}.npy".format(map)
        observation = np.load(TRAINING_DATA_PATH.format(observationFile))

        if actions is None:
            actions = action
            rawObs = observation
        else:
            actions = np.concatenate((actions, action), axis=0)
            rawObs = np.concatenate((rawObs, observation), axis=0)
        print("---")

    model = WindowModel(action_dim=2, max_action=1.)
    try:
        state_dict = torch.load('./models/windowimitate.pt')
        model.load_state_dict(state_dict)
    except:
        print('failed to load model')
        exit()
    model.train().to(device)

    observations = np.zeros((rawObs.shape[0], 3, 160, 120))
    windowobservations = np.zeros((rawObs.shape[0]-3, 12, 160, 120))
    for i, obs in enumerate(rawObs):
        if i > 2:
            windowobservations[i-3,:3,:,:] = observations[i-3, :, :, :]
            windowobservations[i-3,3:6,:,:] = observations[i-2, :, :, :]
            windowobservations[i-3,6:9,:,:] = observations[i-1, :, :, :]
            windowobservations[i-3,9:12,:,:] = observations[i, :, :, :]

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.0004,
        weight_decay=1e-3
    )

    length = min(len(actions), len(windowobservations))
    actions = actions[:length, :]
    windowobservations = windowobservations[:length, :, :, :]
    loss_list = []
    avg_loss = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        batch_indices = np.random.randint(0, windowobservations.shape[0], (args.batch_size))
        obs_batch = torch.from_numpy(windowobservations[batch_indices]).float().to(device)
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
            torch.save(model.state_dict(), 'imitation/pytorch/models/dagger_windowimitate.pt')
            save_loss(loss_list, 'imitation/pytorch/dagger_windowloss.npy')

    print("Saving...")
    torch.save(model.state_dict(), 'imitation/pytorch/models/dagger_windowimitate.pt')

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
    parser.add_argument("--epochs", default=50000, type=int, help="Number of training epochs")
    parser.add_argument("--model-directory", default="models/", type=str, help="Where to save models")

    args = parser.parse_args()

    #_train(args)
    _windowtrain(args)