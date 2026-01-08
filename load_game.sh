#!/user/bin/env/ bash
nohup  python -u  new_data.py \
 --data=Games \
 --job=10 \
 --item_max_length=50 \
 --user_max_length=50 \
 --rw_length=5 \
 --rw_width=10 \
 --noise_threshold=0.0 \
 --version=2 \
 --force_graph=True \
 >./results/ga_data&

