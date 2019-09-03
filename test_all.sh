python test.py --dataroot ./datasets/WestCaff --name westcaff_4x_down --model cycle_gan --load_size 160 --preprocess scale_width   &
python test.py --dataroot ./datasets/WestCaff --name westcaff_half_size --model cycle_gan --load_size 320 --preprocess scale_width &
python test.py --dataroot ./datasets/WestCaff --name westcaff360 --model cycle_gan --load_size 640 --preprocess scale_width        &
python test.py --dataroot ./datasets/WestCaff --name westcaff_deepD --model cycle_gan --load_size 640 --preprocess scale_width     &



for job in $(jobs -p); do
	wait $job;
done


