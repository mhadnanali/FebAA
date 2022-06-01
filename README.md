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
|Inf| 80.59&plusmn;0.58 |  91.07&plusmn;0.20 | 93.74&plusmn;0.19  | 93.55&plusmn;0.14 |  95.89&plusmn;0.06  |
|Rand| &plusmn; |  &plusmn; | &plusmn;  | &plusmn;  | &plusmn;  | &plusmn;  |





## GRACE+FebAA
