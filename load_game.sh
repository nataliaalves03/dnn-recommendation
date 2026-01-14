#!/user/bin/env/ bash
nohup  python -u  new_data.py \
 --data=Games \
 --job=10 \
 --item_max_length=50 \
 --user_max_length=50 \
 --rw_length=10 \
 --rw_width=10 \
 --version=2 \
 --max_rows=10000 \
 --force_graph=True \
 >./results/ga_data&

