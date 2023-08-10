from abc import abstractmethod
from typing import Optional, Dict,Text

from gymnasium import Env
import numpy as np

from highway_env import utils
from highway_env.envs.common.action import Action
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation, observation_factory
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics

from highway_env.vehicle.objects import Landmark, Obstacle

from gym.envs.registration import register

class GoalEnv(Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class MyEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    # For parking env with GrayscaleObservation, the env need
    # this PARKING_OBS to calculate the reward and the info.
    # Bug fixed by Mcfly(https://github.com/McflyWZX)
    PARKING_OBS = {"observation": {
            "type": "KinematicsGoal",
            "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False
        }}

    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        super().__init__(config, render_mode)
        self.observation_type_parking = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (256, 64),
                "stack_size": 4,
                "weights": [0.30, 0.59, 0.11],  # weights for RGB conversion
                "scaling": 1,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [5, 10, 15, 20, 25, 30, 35, 40]
            },

            'reward_speed_range': [20, 30],
            "normalize_reward": True,
            "reward_weights": [1, 0.5, 0, 0, 0, 0],
            "success_goal_reward": 0.15,

            "collision_reward": -100,
            "high_speed_reward": 1,
            "proximity_reward": 10,

            "steering_range": np.deg2rad(45),

            "simulation_frequency": 20,
            "policy_frequency": 10,
            "duration": 20,
            "screen_width": 1000,
            "screen_height": 100,
            "centering_position": [0.5, 0.5],
            "scaling": 3,
            "controlled_vehicles": 1,
            "vehicles_count": 10,

            "rand_obstacle": True
        })
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        super().define_spaces()
        self.observation_type_parking = observation_factory(self, self.PARKING_OBS["observation"])

    def _info(self, obs, action) -> dict:
        info = super(MyEnv, self)._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        info.update({"is_success": success})
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                lanes=3, length=250,
                speed_limit=40
            ),
            np_random=self.np_random, 
            record_history=self.config["show_trajectories"]
        )
        
    def _create_vehicles(self) -> None:
         # create ego vehicle
        self.controlled_vehicles = []
        vehicle = self.action_type.vehicle_class(
            road=self.road, 
            position=[0, 0], 
            heading=0, 
            speed=0)
        vehicle.color = VehicleGraphics.EGO_COLOR
        self.controlled_vehicles.append(vehicle)
        self.road.vehicles.append(vehicle)

        # create obstacle
        lanewidth = 4
        x_interval = 40
        
        obstacle_size = [4, 2]
        obstacle_num = 4
        
        obstacle_xs = [0, 1, 1, 2]

        obstacle_ys = [0, 2, 1, 2]
        if self.config["rand_obstacle"]:
            choice_set = [0, 1, 2]
            rand_choice = np.random.choice(choice_set, size=3)
            obstacle_ys = [
                rand_choice[0],
                choice_set[rand_choice[1] - 1],
                choice_set[rand_choice[1] - 2],
                rand_choice[2]
            ]
        
        
        xpos = np.cumsum(np.ones(4) * x_interval)
        ypos = np.arange(0, 3*lanewidth, lanewidth)
        
        self.obstacles = []
        for idx in range(obstacle_num):
            obstacle = Obstacle(
                self.road, 
                [xpos[obstacle_xs[idx]], ypos[obstacle_ys[idx]]]
            )
            obstacle.LENGTH, obstacle.WIDTH = obstacle_size
            obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)

            self.obstacles.append(obstacle)
            self.road.objects.append(obstacle)
        
        # create landmark
        goal_pos = [xpos[-1], ypos[2]]
        if self.config["rand_obstacle"]:
            goal_pos[1] = ypos[np.random.choice([0, 1, 2])]
        self.goal = Landmark(self.road, goal_pos, heading=0, speed=20)
        self.road.objects.append(self.goal)

    def _rewards(self, action: Action) -> Dict[Text, float]:

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        vehicle = self.controlled_vehicles[0]
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])

        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {}) for agent_obs in obs)

        return {
            "collision_reward": float(vehicle.crashed),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(vehicle.on_road),
            "proximity_reward": reward
        }

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)
    
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["proximity_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached or time is over."""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        notOnroad = any(not vehicle.on_road for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        return bool(crashed or success or notOnroad)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time is over."""
        return self.time >= self.config["duration"]

register(
    id='my-env-v0',
    entry_point='highway_env.envs: MyEnv',
)