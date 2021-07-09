# recommended paddle.__version__ == 2.0.0
CUDA_VISIBLE_DEVICES=1 python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '1'  tools/train.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml
