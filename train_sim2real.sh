python train.py --dataroot ./datasets/WestCaff --name westcaff_first_label --model cycle_gan --preprocess scale_width_and_crop --load_size -1 --load_width 640 --load_height 360 --crop_size 360 --gpu_id 3 --input_nc 6 --output_nc 3 --lambda_identity 0 --no_flip --dataset_mode unaligned_label  --verbose --max_dataset_size 5 --display_port 8099 --display_freq 1 --display_env main --display_ncols 5

