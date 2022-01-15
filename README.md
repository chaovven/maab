# A Cooperative-Competitive Multi-Agent Framework for Auto-bidding in Online Advertising


This is the PyTorch implementation of MAAB. The paper can be found [here](https://arxiv.org/abs/2106.06224). 

The code includes the experiments for the following two environments:

* two-agent bidding game
* offline dataset simulation

and the implementations of the following algorithms:

* CM-IL
* CO-IL
* MAAB (iql_vrl)
* MAAB-fix (iql_vfix)
* DQN-S (iql_ali_single)


## Running experiments


### Two-Agent Bidding Game

This is a simplified bidding environment with only two agents bidding either in a competitive or cooperative manner. 

For a competitive manner (CM-IL), run
```shell
python src/main.py --config=iql --env-config=auction with batch_size=32 env_args.coop=0 # CM-IL
```

For a cooperative manner (CO-IL), run
```shell
python src/main.py --config=iql --env-config=auction with batch_size=32 env_args.coop=100 # CO-IL
```

MAAB is also provided in the two-agent bidding game:
```shell
python src/main.py --config=iql_vrl --env-config=auction with batch_size=32 env_args.coop=4 # MAAB (with bar agents)
```


### Offline Dataset Simulation

For MAAB in offline simulation, run
```shell
python src/main.py --config=iql_ali_vfix --env-config=auction_ali with batch_size=32 v_threshold=0.5 env_args.coop=4 
```

Note that the dataset for the offline simulation is not provided due to data security concern.


## Results
The running results are stored in the `results/tb_logs` folder, in the tensorboard format. You can view the logs as well as the results by running
```shell
tensorboard --logdir results/tb_logs 
```

## Citation
```
@inproceedings{wen2022maab,
  title={A Cooperative-Competitive Multi-Agent Framework for Auto-bidding in Online Advertising},
  author={Wen, Chao and Xu, Miao and Zhang, Zhilin and Zheng, Zhenzhe and Wang, Yuhui and Liu, Xiangyu and Rong, Yu and Xie, Dong and Tan, Xiaoyang and Yu, Chuan and others},
  booktitle={WSDM 2022},
  year={2022}
}
```
