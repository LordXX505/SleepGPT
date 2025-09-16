#python make_pseudo_labels.py \
#  --root /data/shhs_new/shhs_new \
#  --out_dir /data/pseudo_round0 \
#  --fs 100 \
#  --eog_idx 4 \
#  --amp_phasic 150 --amp_tonic 25 --max_width_ms 400 \
#  --min_consecutive 2 --separation_k 2


python validator.py \
  --root /data/shhs_new/shhs_new \
  --result_dir /data/pseudo_round0 \
  --eeg_idx 0 1 \
  --eog_idx 4 \
  --emg_idx 3 \
  --fs 100 \
  --per_label_n 50000 \
  --per_label_show 16 \
  --nperseg 128