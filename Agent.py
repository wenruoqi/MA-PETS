from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from dotmap import DotMap
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1].absolute()))
from smarts.core.observations import Observation

#from smarts.env.utils.observation_conversion import ObservationOptions
import numpy as np
import math
from typing import Dict, List, NamedTuple, Optional, Tuple
import os
os.environ['LD_LIBRARY_PATH']='$LD_LIBRARY_PATH:/home/wrq/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH:/usr/lib/nvidia'


class Agent:
    """An general class for RL agents.
    """

    def __init__(self, params):
        """Initializes an agent.    

        Arguments:
            params: (DotMap) train_action DotMap of agent parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will
                    be added to the action of this agent.
                .noise_stddev: (float) The standard deviation to be used for the
                    action noise if params.noisy_actions is True.
        """
        assert params.get("noisy_actions", False) is False
        self.env = params.env

        if isinstance(self.env, DotMap):
            raise ValueError("Environment must be provided to the agent at initialization.")

    def sample(self, horizon, policy, record_fname=None):
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for action.
            record_fname: (str/None) The name of the file to which a recording of the rollout
                will be saved. If None, the rollout will not be recorded.

        Returns: (dict) train_action dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
        N_AGENTS = 22
        AGENT_IDS = ["Agent_%i" % i for i in range(N_AGENTS)]
        
    
        O, _ = self.env.reset()
        train_action, done = {}, False
        ep_reward = 0
        policy.reset()
        ac_ub = 13.89
        ac_lb = 0
        action_init = {}
        for agent_id in AGENT_IDS:
            action_init.update({agent_id : ((ac_ub + ac_lb)/2, 0)})
            
        while(len(O) < N_AGENTS):
            O, reward, done, info, _ = self.env.step(action_init)
        
        train_obs = [] 
        train_action = []
        rewards = []
        reward_sums = []
        Un_save = []
        filename = f'episode_1_result_4.txt'
        
        for t in range(horizon): 
            group_obs = {}
            actions = {}
            actions_train = {}
            reward = {}
            reward_sum = 0
            
            for agent_id, agent_obs in O.items():
               
                if "neighborhood_vehicle_states" in agent_obs:
                    print(agent_obs["neighborhood_vehicle_states"])


                single_obs = Assignment(agent_obs) 
                group_obs.update({agent_id: single_obs}) 
                policy_act = policy.act(single_obs, t, agent_id) 
                
                action = (policy_act[0], 0) 
                
                actions_train.update({agent_id: policy_act})
                actions.update({agent_id: action})
                
                single_reward = single_obs[0] 
                
                reward.update({agent_id: single_reward})
               
                reward_sum += single_obs[0]

            ep_reward += reward_sum 
            if t==199:
                R=10
            else:
                R=0     
            un_save = ep_reward/(N_AGENTS*t) * 0.1*t +R 
            Un_save.append(un_save)
            with open(filename, 'a') as file:
                file.write(str(Un_save))    
            
            train_obs.append(group_obs)
            train_action.append(actions_train)
            rewards.append(reward)
            reward_sums.append(reward_sum)
            
            O, _, done, info, _ = self.env.step(actions)
            
            if len(O) < N_AGENTS or t == horizon - 1: 
                last_obs = group_obs
                for agent_id in AGENT_IDS: 
                    
                    if agent_id not in O:
                        last_obs[agent_id][3] = 1
                        print(agent_id, "collapsed") 
                train_obs.append(last_obs)
                break
        
        if t==199:
            R=10
        else:
            R=0     
        un = ep_reward/(N_AGENTS*t) * 0.1*t +R 
        return {
            "obs": np.array(train_obs),
            "ac": np.array(train_action),
            "reward_sum": reward_sums,
            "rewards": np.array(rewards),
        }

class TrainState(NamedTuple): 
    ego_speed: float
    ego_x: float
    ego_y: float
    ego_travelled: float
    """Center coordinate of the vehicle bounding box's bottom plane. `shape=(3,)`. `dtype=np.float64`."""
    social_speed_former: float
    social_speed_behind: float
    social_dist_former: float
    """Calculated with"""
    social_dist_behind: float
    """To Judge whether ego is going to dead"""
    ego_dead: int

def distance(a, b): 
    sum = 0
    for i in range(2):
        sum += pow((a[i] - b[i]), 2)
    return math.sqrt(sum)

def Assignment(obs: Observation): 
    state = TrainState
    state.ego_speed = obs.ego_vehicle_state.speed
    state.ego_x = obs.ego_vehicle_state.position[0]
    state.ego_y = obs.ego_vehicle_state.position[1]
    state.ego_travelled = obs.distance_travelled
   
    if "neighborhood_vehicle_states" not in obs:
        state.social_speed_behind = 0
        state.social_dist_behind = 75
        state.social_speed_former = 15
        state.social_dist_former = 75
    else:
        state.social_speed_former = obs.neighborhood_vehicle_states[0].speed
        state.social_dist_former = distance(obs.neighborhood_vehicle_states[0].position, obs.ego_vehicle_state.position)
        if len(obs.neighborhood_vehicle_states) > 1: 
            state.social_speed_behind = obs.neighborhood_vehicle_states[1].speed
            state.social_dist_behind = distance(obs.neighborhood_vehicle_states[1].position, obs.ego_vehicle_state.position)
        else:
            state.social_speed_behind = 0
            state.social_dist_behind = 75    
    
    state.ego_dead = 0
   
    output = np.array([state.ego_speed, state.ego_x, state.ego_y, state.ego_dead, state.social_speed_former, state.social_speed_behind, state.social_dist_former, state.social_dist_behind], dtype=float)
    return output
