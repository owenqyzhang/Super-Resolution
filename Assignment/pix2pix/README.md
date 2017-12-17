Image to image translation

- pix2pix_2.py : Script to run image translation without the skip connections to convert from the a colored image to a sketch
- pix2pix_3.py : Script to run image translation without the skip connections to convert from the a sketch to a colored image
- pix2pix_4.py : Script to run image translation with skip connections to convert from the a colored image to a sketch

Usage:

- Training
python3 pix2pix_[FILE_NO].py --mode train --input_dir [TRAINING_IMAGES_PATH] --output_dir [PATH_TO_STORE_CHECKPOINT] --max_epochs 200 --which_direction [DECIDE_DIRECTION_OF_IMAGE_TO_IMAGE_TRANSITION]

- Testing
python3 pix2pix_[FILE_NO].py --mode test --output_dir [PATH_TO_STORE_RESULTS] --input_dir [PATH_OF_TEST_IMAGES] --checkpoint [PATH_TO_STORE_CHECKPOINT] 
