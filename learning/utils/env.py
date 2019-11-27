import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv

def launch_env(id=None):
    env = None
    if id is None:
        # Launch the environment
        from gym_duckietown.simulator import Simulator
        env = Simulator(
            seed=123, # random seed
            map_name="loop_empty",
            max_steps=500001, # we don't want the gym to reset itself
            domain_rand=0,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4, # start close to straight
            full_transparency=True,
            distortion=True,
        )
    else:
        env = gym.make(id)

    return env

def launch_env1(id=None):
    env = None
    if id is None:
        # Launch the environment
        env = DuckietownEnv(
            seed = 1,
            map_name="loop_empty",
            domain_rand = 1,
            distortion = False,
            max_steps=500001, # we don't want the gym to reset itself
            camera_width=640,
            camera_height=480,
        )
    else:
        env = gym.make(id)

    return env