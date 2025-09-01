#python rem_validation.py \
#  --root /data/shhs_new/shhs_new \
#  --result_dir  /home/user/Sleep/result/concat1536 \
#  --fs 100 \
#  --per_label 5000 \
#  --save_dir /home/user/Sleep/result/concat1536/validation_all \
#  --eog_ch 3

python verify_phasic_like.py \
  --result_dir /home/user/Sleep/result/out_epoch_concat              \
  --data_root  /data/shhs_new/shhs_new               \
  --fs 100 --epoch_sec 30 --patch_sec 2              \
  --eeg_idx 0,1 --eog_idx 3 --emg_idx 4      \
  --max_per_cluster 500000                              \
  --out_dir /home/user/Sleep/result/out_epoch_concat/psd_eval           \
  --db_scale