python train.py --dataroot ./datasets/WestCaff --name westcaff_first_label --model cycle_gan --preprocess scale_width_and_crop --load_size -1 --load_width 320 --load_height 180 --crop_size 180 --input_nc 6 --output_nc 3 --lambda_identity 0 --no_flip --dataset_mode unaligned_label --display_env main --display_ncols 5 "$@"

