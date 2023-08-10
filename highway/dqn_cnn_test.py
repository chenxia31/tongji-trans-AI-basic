import json
import os

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

import highway_env
from my_env import MyEnv

import imageio

dump_path = 'dump'

if __name__ == '__main__':
    # Create the environment
    env = MyEnv(render_mode='rgb_array')
    env.configure({'rand_obstacle': False})

    # obs, info = env.reset()
    # env.render()
    # obs, info = env.reset()
    # env.render()

    # Run the trained model and record video
    model = DQN.load("myEnv_dqn/320000", env=env)
    env = RecordVideo(env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)

    for videos in range(1):
        # init
        frames = []
        ep_log = []

        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            # render
            env.render()
            # add to gif
            frames.append(obs[-1].T)
            # add to log
            veh = env.controlled_vehicles[0]
            ep_log.append(
                {
                    'step': env.step_id,
                    'action': veh.action,
                    'position': veh.position.tolist(),
                    'speed': veh.speed, 
                    'target_lane': int(veh.target_lane_index[-1]*4),
                    'target_speed': int(veh.target_speed)
                }
            )
        
        name = os.path.join(dump_path, f'{env.episode_id}')
        with open(name + '.json', 'w') as f:
            json.dump(ep_log, f)

        imageio.mimsave(
            name + '.gif', frames, 
            format='GIF', duration=0.05, loop=0)

    env.close()
    
    