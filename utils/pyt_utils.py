# encoding: utf-8
import os
import sys
import time
import argparse
from collections import OrderedDict, defaultdict

import torch
import torch.utils.model_zoo as model_zoo
import torch.distributed as dist

from .logger import get_logger

logger = get_logger()


def reduce_tensor(tensor, dst=0, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.reduce(tensor, dst, op)
    if dist.get_rank() == dst:
        tensor.div_(world_size)

    return tensor


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    if norm:
        tensor.div_(world_size)

    return tensor


def extant_file(x):
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

