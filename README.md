# FebAA
Features Based Adaptive Augmentation for Graph Contrastive Learning


## BGRL+FebAA
BGRL+FebAA can be executed using BGRL official code available at the [github](https://github.com/nerdslab/bgrl)  repository, files uploaded in `BGRL+FebAA` folder, needed to be placed in relevant folders given in [BGRL code](https://github.com/nerdslab/bgrl)  . 
The main script file is `BGRL_FebAA.py` used for training on the transductive task datasets and configuration files can be found in `./config` folder. 

### Regenerate Training Results: 

To run BGRL on a dataset from the transductive setting, use `BGRL_FebAA.py` and one of the configuration files that can be found in `config/`.
For example, to train on the wiki-cs dataset, use the following command:

`python BGRL_FebAA.py --flagfile=config/*-wiki-cs_FeBAA.cfg`

Above same command can be used to regenerate the results as seeds values are given in `.cfg` files, while `*` will be replaced with `inf` or `rand`.


### Verify Test Results: 

The `runs` folder contains log files, `Get_Results.py` can be executed to get the results from log files. 
Note that our reported results are based on an average of 20 runs.
Test accuracies under linear evaluation are reported on TensorBoard. To start the tensorboard server run the following command:

`tensorboard --logdir runs`

||  WikiCS | Amazon Computers   | Amazon Photos  | CoAuthorCS   | CoAuthorPhy  |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|Inf| **80.59&plusmn;0.58** |  **91.07&plusmn;0.20** | 93.74&plusmn;0.19  | 93.55&plusmn;0.14 |  95.90&plusmn;0.08  |
|Rand| 80.57&plusmn;0.50 |  90.94&plusmn;0.23 | **93.80&plusmn;0.23**  | **93.58&plusmn;0.13**  | **95.90&plusmn;0.09**  |





## GRACE+FebAA

Grace+FebAA is implemented using [PyGCL](https://github.com/PyGCL/PyGCL). To execute the codes, one need to install [PyGCL](https://github.com/PyGCL/PyGCL) and place the augmentors folder files from this repository in [PyGCL](https://github.com/PyGCL/PyGCL) augmentors folder. Then execute the below command to get results. Make sure to enter relevent seeds given in the last table, dataset in the code (if you are trying to recreate our results). 

`python GRACE+FebAA.py`

We intentionally did not create any configration ( `.cfg` or `.yaml` ) file for input to keep it same as [PyGCL](https://github.com/PyGCL/PyGCL). 

||  Cora | CiteSeer   | Actor  |
| ------------ | ------------ | ------------ | ------------ |
|Inf| 87.30&plusmn;1.12 |  **76.26&plusmn;1.46** | **30.58&plusmn;1.06**  | 
|Rand| **87.48&plusmn;0.50** |  75.36&plusmn;1.29 | 30.35&plusmn;1.09 | 


### Hyper-parameters to Recreate Results

Below table contains the hyper-paramter values to recreate the results, where 1 & 2 indicates the graph view 1 and graph view 2. 

|  Dataset | Edge Drop Prob.  1 & 2  | Feature Ratio 1 & 2  | Feature Drop Prob. 1 & 2  | manual_seed  |random.seed |Least or Most |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ |
|Cora| 0.4 & 0.2  | 100% & 80% | 0.4 & 0.375 | 125656 | 896146|Least |
|CiteSeer| 0.4 & 0.2 | 100% & 70% | 0.4 & 0.43 | 553358  | 559648 | Most |
|Actor| 0.3 & 0.3 | 100% & 30% | 0.3 & 1 | 8833511 | 7396411|Most |


