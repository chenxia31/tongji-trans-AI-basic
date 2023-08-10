from stable_baselines3 import DQN

import highway_env
from my_env import MyEnv

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Create the environment
    env = MyEnv(render_mode='rgb_array')
    env.configure({'rand_obstacle': True})

    obs, info = env.reset()
    env.render()
    plt.imshow(obs[-1].T, cmap='gray')
    plt.show()

    # Create the model
    model = DQN('CnnPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log="myEnvls_dqn/")
    # model = DQN.load("myEnv_dqn/100000", env=env)
    print(model.policy)

    # Train the model
    timesteps = 10000
    for i in range(10, 10000):
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name='DQN')
        model.save(f"./myEnv_dqn/{timesteps*(i+1)}")
    env.close()