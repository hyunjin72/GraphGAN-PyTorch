## GraphGAN_pytorch

This repository is a **PyTorch** implementation of [GraphGAN](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16611)[(arXiv)](https://arxiv.org/abs/1711.08267).

> GraphGAN: Graph Representation Learning With Generative Adversarial Nets\
Hongwei Wang, Jia Wang, Jialin Wang, Miao Zhao, Weinan Zhang, Fuzheng Zhang, Xing Xie, Minyi Guo\
32nd AAAI Conference on Artificial Intelligence, 2018

![](/framework.jpg)

### Files in the folder

- `data/`: training and test data
- `pre_train/`: pre-trained node embeddings
  > Note: the dimension of pre-trained node embeddings should equal n_emb in src/GraphGAN/config.py
- `results/`: evaluation results and the learned embeddings of the generator and the discriminator
- `src/`: source codes

### Requirements

The code has been tested running under Python 3.8.12, with the following packages installed (along with their dependencies):

- pytorch == 1.9.0
- cuda == 10.2
- tqdm == 4.62.3 (for displaying the progress bar)
- numpy == 1.21.2
- sklearn == 1.0.2

### Input format

The input data should be an undirected graph in which node IDs start from *0* to *N-1* (*N* is the number of nodes in the graph). Each line contains two node IDs indicating an edge in the graph.

##### txt file sample

```0	1```

```3	2```

```...```

### Usage

Create directories to store your result and cache file.

```
mkdir cache
mkdir results
```

Before training, you should modify the `PATH` in `config.py`.

```
# src/GraphGAN/config.py
PATH = "/home/[YOUR NAME]/GraphGAN_pytorch/src/"
```

Then, train and evaluate GraphGAN for link prediction:

```
python graph_gan.py
```

