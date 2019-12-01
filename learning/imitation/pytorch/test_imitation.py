#!/usr/bin/env python3

"""
Control the simulator or Duckiebot using a model trained with imitation
learning, and visualize the result.
"""

import time
import sys
import argparse
import math

import torch

import numpy as np
import gym

from utils.env import *
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from utils.teacher import PurePursuitExpert

from imitation.pytorch.model import *
from pyglet.window import key

NUM_TESTS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initenv1():
    env = launch_env1()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)

    return env

def initenv2():
    env = launch_env2()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)

    return env


def _enjoy(env, dagger=False):
    model = Model(action_dim=2, max_action=1.)

    try:
        if dagger:
            state_dict = torch.load('./models/dagger_imitate.pt')
        else:
            state_dict = torch.load('./models/imitate.pt')
        model.load_state_dict(state_dict)
    except:
        print('failed to load model')
        exit()

    model.eval().to(device)

    obs = env.reset()

    successes = 0
    count = 0
    written = False

    while count < NUM_TESTS:
        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)

        action = model(obs)
        action = action.squeeze().data.cpu().numpy()
        
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            count += 1

            if info['Simulator']['done_code'] == 'lap-completed':
                print('*** SUCCESS ***')
                successes += 1
            else:
                print('*** FAILED ***')

            obs = env.reset()
            env.render()

        if count != 0 and count % 50 == 0 and written is False:
            if dagger:
                f = open("../single_test_{}_dagger.txt".format(env.map_name), "a")
            else: 
                f = open("../single_test_{}.txt".format(env.map_name), "a")
            f.write("{} {}".format(env.map_name, count))
            f.write("\n{}/{}\n\n".format(successes, NUM_TESTS))
            f.close()
            written = True
        else:
            if count % 50 != 0:
                written = False


def _enjoyWindow(env, dagger=False):
    model = WindowModel(action_dim=2, max_action=1.)

    try:
        if dagger:
            state_dict = torch.load('./models/dagger_windowimitate.pt')
        else:
            state_dict = torch.load('./models/windowimitate.pt')
        model.load_state_dict(state_dict)
    except:
        print('failed to load model')
        exit()

    model.eval().to(device)

    obs = env.reset()

    obsWindow = np.zeros((12,160,120))

    successes = 0
    count = 0
    written = False

    while count < NUM_TESTS:
        obsWindow[:9,:,:] = obsWindow[3:,:,:]
        obsWindow[9:12,:,:] = obs
        obs = torch.from_numpy(obsWindow).float().to(device).unsqueeze(0)

        action = model(obs)
        action = action.squeeze().data.cpu().numpy()
        
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            count += 1
                
            if info['Simulator']['done_code'] == 'lap-completed':
                print('*** SUCCESS ***')
                successes += 1
            else:
                print('*** FAILED ***')

            obs = env.reset()
            env.render()

        if count != 0 and count % 50 == 0 and written is False:
            if dagger:
                f = open("../window_test_{}_dagger.txt".format(env.map_name), "a")
            else: 
                f = open("../window_test_{}.txt".format(env.map_name), "a")
            f.write("{} {}".format(env.map_name, count))
            f.write("\n{}/{}\n\n".format(successes, NUM_TESTS))
            f.close()
            written = True
        else:
            if count % 50 != 0:
                written = False

if __name__ == '__main__':
    env = initenv1()
    #_enjoy(env)
    #_enjoy(env, dagger=True)
    _enjoyWindow(env)
    #_enjoyWindow(env, dagger=True)
    env.close()
    #env = initenv2()
    #_enjoy(env)
    #_enjoy(env, dagger=True)
    #_enjoyWindow(env)
    #_enjoyWindow(env, dagger=True)
    #env.close()