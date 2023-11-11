
### ogbl-ddi:  

    python main.py --data_name=ogbl-ddi --emb_hidden_channels=512 --gnn_hidden_channels=512 --mlp_hidden_channels=512 --num_neg=3 --dropout=0.3 --loss_func=WeightedHingeAUC

### ogbl-collab: 

    python main.py --data_name=ogbl-collab --predictor=DOT --use_valedges_as_input=True --year=2010 --epochs=800 --eval_last_best=True --dropout=0.3 --use_node_feat=True --alpha=0.5

    python main.py --data_name=ogbl-collab  --predictor=DOT --use_valedges_as_input=True --year=2010 --train_on_subgraph=True --epochs=800 --eval_last_best=True --dropout=0.3 --gnn_num_layers=1 --grad_clip_norm=1 --use_lr_decay=True --random_walk_augment=True --walk_length=10 --loss_func=WeightedHingeAUC


### ogbl-citation2:  

    python main.py --data_name=ogbl-citation2 --use_node_feat=True --emb_hidden_channels=50 --mlp_hidden_channels=200 --gnn_hidden_channels=200 --grad_clip_norm=1 --eval_steps=1 --num_neg=3 --eval_metric=mrr --epochs=100 --neg_sampler=local 



