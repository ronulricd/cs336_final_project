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

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--action', type=str, help='action filename')
parser.add_argument('--record', action='store_true', help='record observations and actions of expert')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

'''
@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
'''

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

def update(dt, actionHistory, obsHistory, count, actions, args):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    step_count = env.unwrapped.step_count

    if actions is None: 
        action = np.array([0.0, 0.0])
        if key_handler[key.UP]:
            action = np.array([1.00, 0.0])
            #action = np.array([0.44, 0.0])
        if key_handler[key.DOWN]:
            action = np.array([-1.00, 0])
            #action = np.array([-0.44, 0])
        if key_handler[key.LEFT]:
            action = np.array([0.35, +1])
        if key_handler[key.RIGHT]:
            action = np.array([0.35, -1])
        if key_handler[key.SPACE]:
            action = np.array([0, 0])
    else: 
        action = actions[step_count]

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    actionHistory.append(action)
    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))
    obsHistory.append(obs)
    print(obs.shape)

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        if info['Simulator']['done_code'] == 'lap-completed' and args.record:
            obsHistoryArray = np.array(obsHistory)
            print(obsHistoryArray.shape)
            np.save('./data/{}/obs_{}.npy'.format(args.map_name, len(count)), obsHistoryArray)
            actionHistoryArray = np.array(actionHistory)
            print(actionHistoryArray.shape)
            np.save('./data/{}/actions_{}.npy'.format(args.map_name, len(count)), actionHistoryArray)
            count.append(0)
        #raise Exception("Stopping the program")
        env.reset()
        obsHistory.clear()
        actionHistory.clear()
        env.render()

    env.render()

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

obsHistory = []
actionHistory = []
count = []

actions = None
if args.action is not None:
    actions = np.load(args.action)
    print(actions)
    raise Exception("test")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate, actionHistory, obsHistory, count, actions, args)

# Enter main event loop
pyglet.app.run()

env.close()
