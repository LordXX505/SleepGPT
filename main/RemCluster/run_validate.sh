#python rem_validation.py \
#  --root /data/shhs_new/shhs_new \
#  --result_dir  /home/user/Sleep/result/concat1536 \
#  --fs 100 \
#  --per_label 5000 \
#  --save_dir /home/user/Sleep/result/concat1536/validation_all \
#  --eog_ch 3

python psd_cluster_compare.py \
  --result_dir /path/to/your/run_dir                 \
  --data_root  /data/shhs_new/shhs_new               \
  --fs 100 --epoch_sec 30 --patch_sec 2              \
  --eeg_idx 0,1,2,3 --eog_idx 4,5 --emg_idx 6       \
  --max_per_cluster 5000                              \
  --out_dir /path/to/your/run_dir/psd_eval           \
  --db_scale