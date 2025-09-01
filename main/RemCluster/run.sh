#python AnalysisManager.py \
#  --features_dir /home/user/Sleep/result/no_ckpt \
#  --save_dir /home/user/Sleep/result \
#  --mode concat1536 --k 2 --pca_dim 128 --normalize --auto_plot --umap_sample 100000 \
#  --visual_only umap



python AnalysisManager.py \
  --features_dir /home/user/Sleep/result/no_ckpt \
  --save_dir   /home/user/Sleep/result/out_epoch_concat  \
  --level patch \
  --patch_feat concat1536 \
  --channel_agg mean \
  --k 2 \
  --pca_dim 64 \
  --plot --diagnose