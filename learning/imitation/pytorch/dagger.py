#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv, MultiMapEnv
from gym_duckietown.wrappers import UndistortWrapper
from tempfile import TemporaryFile
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper

from utils.env import launch_env1
import torch
from imitation.pytorch.model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initRegModel():
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

    env.reset()
    env.render()

    return env, model

def initWindowModel():
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

    env.reset()
    env.render()

    return env, model


def dagger(dt, actionHistory, obsHistory, count, env, model):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    step_count = env.unwrapped.step_count

    daggerAction = np.array([0.0, 0.0])
    if key_handler[key.UP]:
        daggerAction = np.array([1.00, 0.0])
        #action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        daggerAction = np.array([-1.00, 0])
        #action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        daggerAction = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        daggerAction = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        obsHistoryArray = np.array(obsHistory)
        actionHistoryArray = np.array(actionHistory)
        np.save('./dagger/obs_{}.npy'.format(env.map_name), obsHistoryArray)
        np.save('./dagger/actions_{}.npy'.format(env.map_name), actionHistoryArray)

    print(daggerAction)
    actionHistory.append(daggerAction)

    if step_count == 0:
        obs = np.zeros((3,160,120))
    else:
        obs = obsHistory[step_count-1]

    obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)

    action = model(obs)
    action = action.squeeze().data.cpu().numpy()
    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))
    obsHistory.append(obs)

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        '''
        if info['Simulator']['done_code'] == 'lap-completed' and args.record:
            obsHistoryArray = np.array(obsHistory)
            print(obsHistoryArray.shape)
            np.save('./data/{}/obs_{}.npy'.format(args.map_name, len(count)), obsHistoryArray)
            actionHistoryArray = np.array(actionHistory)
            print(actionHistoryArray.shape)
            np.save('./data/{}/actions_{}.npy'.format(args.map_name, len(count)), actionHistoryArray)
            count.append(0)
        '''
        #raise Exception("Stopping the program")
        env.reset()
        #obsHistory.clear()
        #actionHistory.clear()
        env.render()

    env.render()

def windowdagger(dt, actionHistory, obsHistory, count, env, model):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    step_count = env.unwrapped.step_count

    daggerAction = np.array([0.0, 0.0])
    if key_handler[key.UP]:
        daggerAction = np.array([1.00, 0.0])
        #action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        daggerAction = np.array([-1.00, 0])
        #action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        daggerAction = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        daggerAction = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        obsHistoryArray = np.array(obsHistory)
        actionHistoryArray = np.array(actionHistory)
        np.save('./dagger/windowobs_{}.npy'.format(env.map_name), obsHistoryArray)
        np.save('./dagger/windowactions_{}.npy'.format(env.map_name), actionHistoryArray)

    print(daggerAction)
    actionHistory.append(daggerAction)

    obsWindow = np.zeros((12,160,120))

    if len(obsHistory) > 3:
        i = len(obsHistory)-1
        obsWindow[:3,:,:] = obsHistory[i-3]
        obsWindow[3:6,:,:] = obsHistory[i-2]
        obsWindow[6:9,:,:] = obsHistory[i-1]
        obsWindow[9:12,:,:] = obsHistory[i]

    obs = torch.from_numpy(obsWindow).float().to(device).unsqueeze(0)

    action = model(obs)
    action = action.squeeze().data.cpu().numpy()
    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))
    obsHistory.append(obs)

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        '''
        if info['Simulator']['done_code'] == 'lap-completed' and args.record:
            obsHistoryArray = np.array(obsHistory)
            print(obsHistoryArray.shape)
            np.save('./data/{}/obs_{}.npy'.format(args.map_name, len(count)), obsHistoryArray)
            actionHistoryArray = np.array(actionHistory)
            print(actionHistoryArray.shape)
            np.save('./data/{}/actions_{}.npy'.format(args.map_name, len(count)), actionHistoryArray)
            count.append(0)
        '''
        #raise Exception("Stopping the program")
        env.reset()
        #obsHistory.clear()
        #actionHistory.clear()
        env.render()

    env.render()


if __name__ == '__main__':
    #env, model = initRegModel()
    env, model = initWindowModel()

    # Register a keyboard handler
    key_handler = key.KeyStateHandler()
    env.unwrapped.window.push_handlers(key_handler)

    obsHistory = []
    actionHistory = []
    count = []

    #pyglet.clock.schedule_interval(dagger, 1.0 / env.unwrapped.frame_rate, actionHistory, obsHistory, count, env, model)
    pyglet.clock.schedule_interval(windowdagger, 1.0 / env.unwrapped.frame_rate, actionHistory, obsHistory, count, env, model)

    # Enter main event loop
    pyglet.app.run()

    env.close()
