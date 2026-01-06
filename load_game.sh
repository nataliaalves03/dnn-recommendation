#!/user/bin/env/ bash
nohup  python -u  new_data.py \
 --data=Games \
 --job=10 \
 --item_max_length=50 \
 --user_max_length=50 \
 --rw_length=3 \
 --rw_width=20 \
 --noise_threshold=0.0 \
 --version=3 \
 --force_graph=True \
 >./results/ga_data&

