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

from utils.env import launch_env1
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from utils.teacher import PurePursuitExpert

from imitation.pytorch.model import *
from pyglet.window import key

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _enjoy():
    model = Model(action_dim=2, max_action=1.)

    try:
        state_dict = torch.load('./models/imitate.pt')
        model.load_state_dict(state_dict)
    except:
        print('failed to load model')
        exit()

    model.eval().to(device)

    env = launch_env1()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)

    obs = env.reset()

    while True:
        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)

        action = model(obs)
        action = action.squeeze().data.cpu().numpy()
        
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            if reward < 0:
                print('*** FAILED ***')
                time.sleep(0.7)
                
            obs = env.reset()
            env.render()

def _enjoyWindow():
    model = WindowModel(action_dim=2, max_action=1.)

    try:
        state_dict = torch.load('./models/windowimitate.pt')
        model.load_state_dict(state_dict)
    except:
        print('failed to load model')
        exit()

    model.eval().to(device)

    env = launch_env1()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)

    obs = env.reset()

    obsWindow = np.zeros((12,160,120))

    while True:
        obsWindow[:9,:,:] = obsWindow[3:,:,:]
        obsWindow[9:12,:,:] = obs
        obs = torch.from_numpy(obsWindow).float().to(device).unsqueeze(0)

        action = model(obs)
        action = action.squeeze().data.cpu().numpy()
        
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            if reward < 0:
                print('*** FAILED ***')
                time.sleep(0.7)
                
            obs = env.reset()
            env.render()

def _dagger():
    model = Model(action_dim=2, max_action=1.)

    try:
        state_dict = torch.load('./models/imitate.pt')
        model.load_state_dict(state_dict)
    except:
        print('failed to load model')
        exit()

    model.eval().to(device)

    env = launch_env1()
    # Register a keyboard handler

    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)

    obs = env.reset()
    env.render()
    key_handler = key.KeyStateHandler()
    env.unwrapped.window.push_handlers(key_handler)

    print(env.map_name)
    raise Exception("asdfsadf")

    obsHistory = []
    actionHistory = []

    while True:
        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)

        action = model(obs)
        action = action.squeeze().data.cpu().numpy()

        obs, reward, done, info = env.step(action)

        print(key_handler)
        daggerAction = np.array([0.0, 0.0])
        if key_handler[key.UP]:
            print("as===as=df=sad=f=asdf=sad=fs=adf")
            daggerAction = np.array([1.00, 0.0])
            #action = np.array([0.44, 0.0])
        if key_handler[key.DOWN]:
            print("as===as=df=sad=f=asdf=sad=fs=adf")
            daggerAction = np.array([-1.00, 0])
            #action = np.array([-0.44, 0])
        if key_handler[key.LEFT]:
            print("as===as=df=sad=f=asdf=sad=fs=adf")
            daggerAction = np.array([0.35, +1])
        if key_handler[key.RIGHT]:
            print("as===as=df=sad=f=asdf=sad=fs=adf")
            daggerAction = np.array([0.35, -1])
        if key_handler[key.SPACE]:
            obsHistoryArray = np.array(obsHistory)
            actionHistoryArray = np.array(actionHistory)
            np.save('./dagger/obs_{}.npy'.format(len(count)), obsHistoryArray)
            np.save('./dagger/actions_{}.npy'.format(len(count)), actionHistoryArray)

        print(daggerAction)
        obsHistory.append(obs) 
        actionHistory.append(daggerAction) 
        
        env.render()

        if done:
            if reward < 0:
                print('*** FAILED ***')
                time.sleep(0.7)
                
            obs = env.reset()
            env.render()

if __name__ == '__main__':
    #_enjoyWindow()
    #_enjoy()
    _dagger()