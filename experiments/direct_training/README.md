# About

This folder contains experiments on training different models on the spatialsense and vidvrd datasets. 
There are two versions -- 
1. where models are trained on these two datasets from scratch
2. where models use a pretrained backbone, and then finetuned on these two datasets. 



## Example usage. 

to run vtranse.py, simply run `python vtranse.py`

To run the script vtranse.py on CUDA device 3, run the below    `CUDA_VISIBLE_DEVICES=3  python vtranse.py`


if you want to run it in the background using nohup, do `CUDA_VISIBLE_DEVICES=3 nohup python vtranse.py > vtranse.out &`