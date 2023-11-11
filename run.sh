python main.py --data_name=ogbl-collab --predictor=DOT --use_valedges_as_input=True --year=2010 --epochs=800 --eval_last_best=True --dropout=0.3 --use_node_feat=True --alpha=0.5
python main.py --data_name=ogbl-ddi --emb_hidden_channels=512 --gnn_hidden_channels=512 --mlp_hidden_channels=512 --num_neg=3 --dropout=0.3 --loss_func=WeightedHingeAUC
