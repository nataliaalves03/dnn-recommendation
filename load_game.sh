#!/user/bin/env/ bash
nohup  python -u  new_data_rw.py \
 --data=Games \
 --job=10 \
 --item_max_length=50 \
 --user_max_length=50 \
 --k_hop=3 \
 --force_graph=True \
 >./results/ga_data&

