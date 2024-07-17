# MA-PETS
> Multi-Agent Probabilistic Ensembles With Trajectory Sampling for Connected Autonomous Vehicles
> published in IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY

## Abstract

Connected Autonomous Vehicles (CAVs) have attracted significant attention in recent years and Reinforcement Learning (RL) has shown remarkable performance in improving the autonomy of vehicles. In that regard, Model-Based RL (MBRL) manifests itself in sample-efficient learning, but the asymptotic performance of MBRL might lag behind the state-of-the-art Model-Free RL algorithms. Furthermore, most studies for CAVs are limited to the decision-making of a single vehicle only, thus underscoring the performance due to the absence of communications. In this study, we try to address the decision-making problem of multiple CAVs with limited communications and propose a decentralized Multi-Agent Probabilistic Ensembles (PEs) with Trajectory Sampling (TS) algorithm namely MA-PETS. In particular, to better capture the uncertainty of the unknown environment, MA-PETS leverages PE neural networks to learn from communicated samples among neighboring CAVs. Afterward, MA-PETS capably develops TS-based model-predictive control for decision-making. On this basis, we derive the multi-agent group regret bound affected by the number of agents within the communication range and mathematically validate that incorporating effective information exchange among agents into the multi-agent learning scheme contributes to reducing the group regret bound in the worst case. Finally, we empirically demonstrate the superiority of MA-PETS in terms of the sample efficiency comparable to MFRL.


<p align="center">
<img src="./pictures/figure2.jpg" width="50%">
</p>

#### Paper link: 
[Multi-Agent Probabilistic Ensembles With Trajectory Sampling for Connected Autonomous Vehicles](https://arxiv.org/html/2312.13910v2)


## Requirements
1. The provided environments require SMARTS 2022. 
2. Pytorch 1.0.0
3. Other dependencies can be installed with the pip dependency file `requirements.txt` and conda dependency file `environments.yml`.


## Quick Start
```bash
activate smarts
cd /home/wrq/SMARTS
scl run --envision /home/wrq/SMARTS/MAPETS_FOR_SMARTS/run.py -env smartscavs_v1
```

## Acknowledgement

This repository is based on modifications and extensions of [PETS](https://github.com/kchua/handful-of-trials). We express our gratitude for the original contributions.
