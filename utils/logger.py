# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
import os
import time
from sys import stderr
import logging

def setup_logger(save_dir, filename="log.txt", mode="a", timestamp=False) -> logging.Logger:
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        mode(str): log file write mode, `append` or `override`. default is `a`.
        timestamp(bool): whether add timestamp to filename.
    Return:
        logger instance.
    """
    save_file = os.path.join(save_dir, filename)
    if timestamp:
        basename, extname = os.path.splitext(save_file)
        save_file = basename + time.strftime("-%Y-%m-%d-%H:%M:%S", time.localtime()) + extname
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(save_file, mode, encoding='utf8')
    file_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Create console handler and set level to info
    console_handler = logging.StreamHandler(stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)
    return logger


def setup_tensorboard_logger(save_dir, distributed_rank=0, name="tb", timestamp=True):
    """setup tensorboard logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        name(str): save folder
        timestamp(bool): whether add timestamp to `name`.
    Return:
        tensorboard logger instance for rank0, None for others.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("ImportError: tensorboard")
        return None
    if distributed_rank == 0:
        # save_file = os.path.join(save_dir, name) if name else save_dir
        if timestamp:
            save_file = os.path.join(save_dir, name + time.strftime("-%Y-%m-%d-%H_%M_%S", time.localtime()))
        else:
            save_file = os.path.join(save_dir, name)
        writer = SummaryWriter(log_dir=save_file)
        return writer
    else:
        return None
