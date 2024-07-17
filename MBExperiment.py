from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os 
from time import localtime, strftime
from dotmap import DotMap
from tqdm import trange  
from Agent import Agent  
from DotmapUtils import get_required_argument  
import numpy as np
import math

class MBExperiment:

    N_AGENTS = 22
    AGENT_IDS = ["Agent_%i" % i for i in range(N_AGENTS)]

    def __init__(self, params):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment
                    .task_hor (int): Task horizon
                    .stochastic (bool): (optional) If True, agent adds noise to its actions.
                        Must provide noise_std (see below). Defaults to False.
                    .noise_std (float): for stochastic agents, noise of the form N(0, noise_std^2I)
                        will be added.

                .exp_cfg:
                    .ntrain_iters (int): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.

                .log_cfg:
                    .logdir (str): Parent of directory path where experiment data will be saved.
                        Experiment will be saved in logdir/<date+time of experiment start>
                    .nrecord (int): (optional) Number of rollouts to record for every iteration.
                        Defaults to 0.
                    .neval (int): (optional) Number of rollouts for performance evaluation.
                        Defaults to 1.
        """

        assert params.sim_cfg.get("stochastic", False) == False

        self.env = get_required_argument(params.sim_cfg, "env", "Must provide environment.")
        self.task_hor = get_required_argument(params.sim_cfg, "task_hor", "Must provide task horizon.")
        self.agent = Agent(DotMap(env=self.env, noisy_actions=False))

        self.ntrain_iters = get_required_argument(
            params.exp_cfg, "ntrain_iters", "Must provide number of training iterations."
        )

        self.nrollouts_per_iter = params.exp_cfg.get("nrollouts_per_iter", 1)
        self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")

        self.logdir = os.path.join(
            get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory."),
            strftime("%Y-%m-%d--%H:%M:%S", localtime())
        )
        self.nrecord = params.log_cfg.get("nrecord", 0)
        self.neval = params.log_cfg.get("neval", 20)

    def run_experiment(self):
        samples = [] 
        single_sample = self.agent.sample(self.task_hor, self.policy)
        samples.append(single_sample)
            
        len_rollout = len(samples[-1]["obs"])
        print("len_rollout",len_rollout)    
        
        
        for agent_id in MBExperiment.AGENT_IDS:
                print(agent_id)

                new_train_in = []
                new_train_targs = []

                for sample in samples:
                    print("Communiting...")
                    commute_id = commute(sample['obs'][-1], agent_id) 
                    print("commute_id", commute_id)
                    for commute_agent in commute_id:
                        train_in, train_targs = transfer(samples, commute_agent)

                        new_train_in.append(train_in)
                        new_train_targs.append(train_targs)
                self.policy.train(new_train_in, new_train_targs)    

        for i in trange(self.ntrain_iters): 
            print("##########################################")
            print("Starting training iteration %d." % (i + 1))
            
            samples = []
            samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy
                    )
                )
        
            len_rollout= len(samples[-1]["obs"])
         
            print("Rollout length", len_rollout)
            
            samples = samples[:self.nrollouts_per_iter]

            if i < self.ntrain_iters - 1:
                for agent_id in MBExperiment.AGENT_IDS:
                    print(agent_id)
                    new_train_in = []
                    new_train_targs = []

                    for sample in samples:
                        commute_id = commute(sample['obs'][-1], agent_id) 
                        print("commute_id", commute_id)
                        for commute_agent in commute_id:
                            train_in, train_targs = transfer(samples, commute_agent)

                            new_train_in.append(train_in)
                            new_train_targs.append(train_targs)
    
                    self.policy.train(new_train_in, new_train_targs)

def transfer(samples, agent_id):
    for sample in samples:
        obs_sample = []
        acs_sample = []
        for i in range(len(sample['obs'])):
            ego_obs = sample["obs"][i][agent_id]
            obs_sample.append(ego_obs)
            if i == len(sample['obs']) - 1:
                break
            else:
                ego_ac = sample["ac"][i][agent_id]
                acs_sample.append(ego_ac)

        new_train_in = np.concatenate([obs_sample[:-1], acs_sample], axis=-1)
        new_train_targs = np.array(obs_sample[1:]) - np.array(obs_sample[:-1])

    return new_train_in, new_train_targs

def commute(group_obs, agent_id):
    radius =100
    commute_id = [agent_id]
    ego_pos = [group_obs[agent_id][1], group_obs[agent_id][2]] 
    for other_id in  MBExperiment.AGENT_IDS:
        if other_id == agent_id:
            continue
        other_pos = [group_obs[other_id][1], group_obs[other_id][2]]
        if distance(ego_pos, other_pos) < radius:
            commute_id.append(other_id)
        
    return commute_id

def distance(a, b): 
    sum = 0
    for i in range(2):
        sum += pow((a[i] - b[i]), 2)
    return math.sqrt(sum)
