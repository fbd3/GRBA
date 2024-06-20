# download small.bin
# python scripts/download.py --url https://drive.google.com/open?id=1JCHm39rf7HAJSp-1755wa32ToHCn2Twz --path data --fname small.bin
# For regions where Google is not accessible, use
# python scripts/download.py --url https://cloud.tsinghua.edu.cn/f/b37eed70207c468ba367/?dl=1 --path data --fname small.bin

bash scripts/pretrain2.sh \
    0 \
    --moco --nce-k \
    16384 

echo "pretraining DONE"

bash scripts/generate.sh \
    0 saved/Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999/ckpt_epoch_100.pth collab
echo "Have Generated Downstream embedding"

bash scripts/graph_classification/ours.sh saved/Pretrain_moco_True_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999 32 collab
echo "Classification DONE"