# Heuristic Learning Graph Neural Network (HL-GNN)

This repository contains the official implementation of HL-GNN, as presented in the paper ["Heuristic Learning with Graph Neural Networks: A Unified Framework for Link Prediction,"](https://arxiv.org/pdf/2406.07979) accepted at KDD 2024.

## Overview

HL-GNN is a novel method for link prediction that unifies local and global heuristics into a matrix formulation and implements it efficiently using graph neural networks. HL-GNN is simpler than GCN and can effectively reach up to 20 layers. It demonstrates effectiveness in link prediction tasks and scales well to large OGB datasets. Notably, HL-GNN requires only a few parameters for training (excluding the predictor) and is significantly faster than existing methods.

<div align=center>
    <img src="/HL-GNN.png" alt="HL-GNN" width="60%" height="60%">
</div>

For more details, please refer to the [paper](https://arxiv.org/pdf/2406.07979).

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/LARS-research/HL-GNN.git
cd HL-GNN
pip install -r requirements.txt
```

## Usage

### Planetoid Datasets

#### Cora

```bash
cd Planetoid
python planetoid.py --dataset cora --mlp_num_layers 3 --hidden_channels 8192 --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR
```

#### Citeseer

```bash
cd Planetoid
python planetoid.py --dataset citeseer --mlp_num_layers 2 --hidden_channels 8192 --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR
```

#### Pubmed

```bash
cd Planetoid
python planetoid.py --dataset pubmed --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 300 --K 20 --alpha 0.2 --init KI
```

### Amazon Datasets

#### Photo

```bash
cd Planetoid
python amazon.py --dataset photo --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 200 --K 20 --alpha 0.2 --init RWR
```

#### Computers

```bash
cd Planetoid
python amazon.py --dataset computers --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 200 --K 20 --alpha 0.2 --init RWR
```

### OGB Datasets

#### ogbl-collab

```bash
cd OGB
python main.py --data_name ogbl-collab --predictor DOT --use_valedges_as_input True --year 2010 --epochs 800 --eval_last_best True --dropout 0.3 --use_node_feat True
```

#### ogbl-ddi

```bash
cd OGB
python main.py --data_name ogbl-ddi --emb_hidden_channels 512 --gnn_hidden_channels 512 --mlp_hidden_channels 512 --num_neg 3 --dropout 0.3 --loss_func WeightedHingeAUC
```

#### ogbl-ppa

```bash
cd OGB
python main.py --data_name ogbl-ppa --emb_hidden_channels 256 --mlp_hidden_channels 512 --gnn_hidden_channels 512 --grad_clip_norm 2.0 --epochs 500 --eval_steps 1 --num_neg 3 --dropout 0.5 --use_node_feat True --alpha 0.5 --loss_func WeightedHingeAUC
```

#### ogbl-citation2

```bash
cd OGB
python main.py --data_name ogbl-citation2 --emb_hidden_channels 64 --mlp_hidden_channels 256 --gnn_hidden_channels 256 --grad_clip_norm 1.0 --epochs 100 --eval_steps 1 --num_neg 3 --dropout 0.3 --eval_metric mrr --neg_sampler local --use_node_feat True --alpha 0.6
```

## Results

The performance of HL-GNN on various datasets is summarized in the table below. The best and second-best performances are highlighted in **bold** and *italic*, respectively.

|         |   Cora    | Citeseer  |  Pubmed   |   Photo   | Computers |  collab   |    ddi    |    ppa    | citation2 |
| :-----: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| Method  | Hits@100  | Hits@100  | Hits@100  |    AUC    |    AUC    |  Hits@50  |  Hits@20  | Hits@100  |    MRR    |
|  SEAL   |   81.71   |   83.89   |  *75.54*  |   98.85   |  *98.70*  |   64.74   |   30.56   |   48.80   |  *87.67*  |
| NBFNet  |   71.65   |   74.07   |   58.73   |   98.29   |   98.03   |    OOM    |   4.00    |    OOM    |    OOM    |
| Neo-GNN |   80.42   |   84.67   |   73.93   |   98.74   |   98.27   |   62.13   |   63.57   |   49.13   |   87.26   |
|  BUDDY  |  *88.00*  |  *92.93*  |   74.10   |  *99.05*  |   98.69   |  *65.94*  |  *78.51*  |  *49.85*  |   87.56   |
| HL-GNN  | **94.22** | **94.31** | **88.15** | **99.11** | **98.82** | **68.11** | **80.27** | **56.77** | **89.43** |

## Acknowledgement

We sincerely thank the [PLNLP repository](https://github.com/zhitao-wang/PLNLP) for providing an excellent pipeline that greatly facilitated our work on the OGB datasets.

## Citation

If you find HL-GNN useful in your research, please cite our paper:

```bibtex
@inproceedings{zhang2024heuristic,
  title={Heuristic Learning with Graph Neural Networks: A Unified Framework for Link Prediction},
  author={Zhang, Juzheng and Wei, Lanning and Xu, Zhen and Yao, Quanming},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```

Feel free to reach out if you have any questions!

