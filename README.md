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
