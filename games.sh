#!/user/bin/env/ bash
nohup  python -u  new_main_CA.py \
 --data=Games \
 --gpu=0 \
 --epoch=20 \
 --hidden_size=50 \
 --batchSize=50 \
 --user_long=orgat\
 --user_short=att \
 --item_long=orgat \
 --item_short=att \
 --user_update=rnn \
 --item_update=rnn \
 --lr=0.001 \
 --l2=0.0001 \
 --layer_num=2 \
 --item_max_length=50 \
 --user_max_length=50 \
 --attn_drop=0.3 \
 --feat_drop=0.3 \
 --rw_length=10 \
 --rw_width=10 \
 --version=4 \
 --record \
 >./results/ga_result&
