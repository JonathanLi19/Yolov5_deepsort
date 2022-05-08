import os
import sys
import track

# SPORTSMOT_ROOT_DIR = '/home/lqh/data3/sportsmot_final/sportsmot/test'
SPORTSMOT_ROOT_DIR = '/home/zxy/data3/sportsmot_collection/sportsmot-2022-4-24/sportsmot'
OUTPUT_DIR = "/home/zxy/data3/sportsmot_collection/results/YOLOv5-DeepSORT/full"
SPLITS = ["test", "train", "val"]

import time
import logging


def get_logger(log_file_path,
               console_log_level=logging.DEBUG,
               file_log_level=logging.INFO):
    log_dir = os.path.dirname(log_file_path)
    print(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        fmt=
        '========%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s)========\n%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger(log_file_path)

    logger.setLevel(file_log_level)
    console = logging.StreamHandler()
    # print_all_info
    console.setLevel(console_log_level)
    console.setFormatter(formatter)

    file = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file.setLevel(file_log_level)
    file.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file)

    return logger


TIME_STR = time.strftime("%Y-%m-%d-%H-%M-%S")
logger = get_logger(os.path.join(OUTPUT_DIR, f"log/{TIME_STR}.log"))


def eval_dir(d):
    # for video_name in os.listdir(d):
    videos = list(sorted(os.listdir(d)))
    for video_name in videos:
        source_dir = os.path.join(d, video_name)
        source_dir = os.path.join(source_dir, 'img1')
        # os.system(f"cd {source_dir}/img1")
        # os.system(f"python track.py --source {source_dir} --yolo_model best.pt --save-txt --save-vid")
        save_dir = os.path.join(OUTPUT_DIR, video_name)

        os.makedirs(save_dir, exist_ok=True)
        # if not os.path.isdir(save_dir):
        #     os.makedirs(save_dir)
        # else:
        #     logger.error(f"Dir {save_dir} exists. Skipped.")
        #     continue

        # os.system(
        #     f"python track.py --source {source_dir} --custom_save_dir {save_dir} --yolo_model best.pt --save-txt --save-vid --save_config"
        # )

        # ==[Note💡]==: Instead, mimic command line to set system arguments variables
        _l = [
            "--source", f"{source_dir}", "--custom_save_dir", f"{save_dir}",
            "--yolo_model", "best.pt", "--save-txt", "--save-vid",
            "--save_config"
        ]
        sys.argv.extend(_l)
        # ==[Note💡]==: invoke the main process.
        track.main()


if __name__ == '__main__':
    for split in SPLITS:
        # for split in ["test"]:
        split_dir = os.path.join(SPORTSMOT_ROOT_DIR, split)
        eval_dir(split_dir)
