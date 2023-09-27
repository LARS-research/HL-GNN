

|       | ogbl-ddi (Hits@20)      | ogbl-collab (Hits@50)     | ogbl-citation2 (MRR)     |
| ---------- | :-----------:  | :-----------: | :-----------: |
| Validation | 82.42 ± 2.53  | 100.00 ± 0.00 | 84.90 ± 0.31 |
|  Test | 90.88 ± 3.13  | 70.59 ± 0.29 | 84.92 ± 0.29 |

### ogbl-ddi:  

    python main.py --data_name=ogbl-ddi --emb_hidden_channels=512 --gnn_hidden_channels=512 --mlp_hidden_channels=512 --num_neg=3 --dropout=0.3 loss_func=WeightedHingeAUC

### ogbl-collab: 

Validation set is allowed to be used for training in this dataset. Meanwhile, following the trick of HOP-REC, we only use training edges after year 2010 with validation edges, and train the model on this subgraph. 
The performance of "**PLNLP (val as input)**"  on the leader board can be reproduced with following command:

    python main.py --data_name=ogbl-collab --predictor=DOT --use_valedges_as_input=True --year=2010 --epochs=800 --eval_last_best=True --dropout=0.3 --use_node_feat=True --alpha=0.5

Furthermore, we sample high-order pairs with random walk and employ them as a kind of data augmentation. This augmentation method improves the performance significantly. To reproduce the performance of "**PLNLP (random walk aug.)**" on the leader board, you can use the following command:

    python main.py --data_name=ogbl-collab  --predictor=DOT --use_valedges_as_input=True --year=2010 --train_on_subgraph=True --epochs=800 --eval_last_best=True --dropout=0.3 --gnn_num_layers=1 --grad_clip_norm=1 --use_lr_decay=True --random_walk_augment=True --walk_length=10 --loss_func=WeightedHingeAUC


### ogbl-citation2:  

    python main.py --data_name=ogbl-citation2 --use_node_feat=True --emb_hidden_channels=50 --mlp_hidden_channels=200 --gnn_hidden_channels=200 --grad_clip_norm=1 --eval_steps=1 --num_neg=3 --eval_metric=mrr --epochs=100 --neg_sampler=local 



