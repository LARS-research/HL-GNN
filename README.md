# Heuristic Learning with Graph Neural Networks: A Unified Framework for Link Prediction

This repository contains the implementation for the paper titled "Heuristic Learning with Graph Neural Networks: A Unified Framework for Link Prediction," which is currently under review for KDD 2024.

## Requirements

Ensure you have the required dependencies installed by running:

```
pip install -r requirements.txt
```

## Usages

### **Cora**

```
cd Planetoid
python planetoid.py --dataset cora --mlp_num_layers 3 --hidden_channels 8192 --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR
```

### Citeseer

```
cd Planetoid
python planetoid.py --dataset citeseer --mlp_num_layers 2 --hidden_channels 8192 --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR
```

### **Pubmed**

```
cd Planetoid
python planetoid.py --dataset pubmed --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 300 --K 20 --alpha 0.2 --init KI
```

### **Photo**

```
cd Planetoid
python amazon.py --dataset photo --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 200 --K 20 --alpha 0.2 --init RWR
```

### **Computers**

```
cd Planetoid
python amazon.py --dataset computers --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 200 --K 20 --alpha 0.2 --init RWR
```

### **ogbl-collab**

```
cd OGB
python main.py --data_name ogbl-collab --predictor DOT --use_valedges_as_input True --year 2010 --epochs 800 --eval_last_best True --dropout 0.3 --use_node_feat True
```

### **ogbl-ddi**

```
cd OGB
python main.py --data_name ogbl-ddi --emb_hidden_channels 512 --gnn_hidden_channels 512 --mlp_hidden_channels 512 --num_neg 3 --dropout 0.3 --loss_func WeightedHingeAUC
```



## Results

The results are presented in the table below. The format is average score ± standard deviation. OOM means out of GPU memory. Best and second-best performances are highlighted in **bold** and *italic*, respectively.

|  Metric  |      Cora      |    Citeseer    |     Pubmed     |     Photo      |   Computers    |     collab     |      ddi       |
| :------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
|    CN    |   33.92±0.46   |   29.79±0.90   |   23.13±0.15   |   96.73±0.00   |   96.15±0.00   |   56.44±0.00   |   17.73±0.00   |
|    RA    |   41.07±0.48   |   33.56±0.17   |   27.03±0.35   |   97.20±0.00   |   96.82±0.00   |   64.00±0.00   |   27.60±0.00   |
|    KI    |   42.34±0.39   |   35.62±0.33   |   30.91±0.69   |   97.45±0.00   |   97.05±0.00   |   59.79±0.00   |   21.23±0.00   |
|   RWR    |   42.57±0.56   |   36.78±0.58   |   29.77±0.45   |   97.51±0.00   |   96.98±0.00   |   60.06±0.00   |   22.01±0.00   |
|    MF    |   64.67±1.43   |   65.19±1.47   |   46.94±1.27   |   97.92±0.37   |   97.56±0.66   |   38.86±0.29   |   13.68±4.75   |
| Node2vec |   68.43±2.65   |   69.34±3.04   |   51.88±1.55   |   98.37±0.33   |   98.21±0.39   |   48.88±0.54   |   23.26±2.09   |
| DeepWalk |   70.34±2.96   |   72.05±2.56   |   54.91±1.25   |   98.83±0.23   |   98.45±0.45   |   50.37±0.34   |   26.42±6.10   |
|   GCN    |   66.79±1.65   |   67.08±2.94   |   53.02±1.39   |   98.61±0.15   |   98.55±0.27   |   47.14±1.45   |   37.07±5.07   |
|   GAT    |   60.78±3.17   |   62.94±2.45   |   46.29±1.73   |   98.42±0.19   |   98.47±0.32   |   55.78±1.39   |   54.12±5.43   |
|   SEAL   |   81.71±1.30   |   83.89±2.15   |   75.54±1.32   |   98.85±0.04   |  *98.70±0.18*  |   64.74±0.43   |   30.56±3.86   |
|  NBFNet  |   71.65±2.27   |   74.07±1.75   |   58.73±1.99   |   98.29±0.35   |   98.03±0.54   |      OOM       |   4.00±0.58    |
| Neo-GNN  |   80.42±1.31   |   84.67±2.16   |   73.93±1.19   |   98.74±0.55   |   98.27±0.79   |   62.13±0.58   |   63.57±3.52   |
|  BUDDY   |  *88.00±0.44*  |  *92.93±0.27*  |   74.10±0.78   |  *99.05±0.21*  |   98.69±0.34   |  *65.94±0.58*  |  *78.51±1.36*  |
|  HL-GNN  | **94.22±1.64** | **94.31±1.51** | **88.15±0.38** | **99.11±0.07** | **98.82±0.21** | **68.11±0.54** | **80.27±3.98** |
