python ./test.py --datapath "../datasets" \
                 --benchmark coco \
                 --fold 0 \
                 --bsz 1 \
                 --nworker 8 \
                 --backbone swin \
                 --feature_extractor_path "../backbones/swin_base_patch4_window12_384.pth" \
                 --logpath "./logs" \
                 --load "./best_model.pt" \
                 --nshot 5 \
                 --vispath "./vis_5" \
                 --visualize
