import argparse
import os, sys

sys.path.append("..")

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import time

import os.path as osp
from models.TransDoDNet import MOTS_DeformTR as MOTS_model
from MOTSDataset import MOTSTrainDataSet, my_collate
import random
import timeit
from loss_functions import loss
from utils.ParaFlop import print_model_parm_nums
import utils.utils as utils
from math import ceil
from engine import Engine
from apex import amp
import logging

start = timeit.default_timer()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    parser = argparse.ArgumentParser(description="MOTS_TransDoDNet")

    parser.add_argument("--data_dir", type=str, default='../data_list/')
    parser.add_argument("--train_list", type=str, default='MOTS/MOTS_train.txt')
    parser.add_argument("--val_list", type=str, default='')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/')
    parser.add_argument("--reload_path", type=str, default='snapshots/checkpoint.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--nnUNet_preprocessed", type=str, default=os.environ['nnUNet_preprocessed'])

    parser.add_argument("--input_size", type=str, default='64,192,192')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=True)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_linear_proj_mult", type=float, default=0.1)
    parser.add_argument("--lr_tr_mult", type=float, default=1.0)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=6)

    parser.add_argument("--weight_std", type=str2bool, default=False)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--not_restore_last", action="store_true")
    parser.add_argument("--save_num_images", type=int, default=2)

    # data aug.
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=123)

    # others
    parser.add_argument("--gpu", type=str, default='None')
    parser.add_argument("--recurrence", type=int, default=1)
    parser.add_argument("--ft", type=str2bool, default=False)
    parser.add_argument("--ohem", type=str2bool, default='False')
    parser.add_argument("--ohem_thres", type=float, default=0.6)
    parser.add_argument("--ohem_keep", type=int, default=200000)

    # Res
    parser.add_argument('--res_depth', default=50, type=int)
    parser.add_argument("--dyn_head_dep_wid", type=str, default='3,8')

    # * Transformer
    parser.add_argument("--using_transformer", type=str2bool, default=True)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--dec_layers', default=3, type=int)
    parser.add_argument('--dim_feedforward', default=768, type=int)
    parser.add_argument('--hidden_dim', default=192, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=7, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--num_feature_levels', default=3, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--normalize_before', default=False, type=str2bool)
    parser.add_argument('--deepnorm', default=True, type=str2bool)
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument("--add_memory", type=int, default=2, choices=(0,1,2)) # feature fusion: 0: cnn; 1:tr; 2:cnn+tr

    parser.add_argument('--optimizer', default='adamw', type=str, choices=('sgd', 'adamw'))

    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, args):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr, num_stemps, power = args.learning_rate, args.num_epochs, args.power
    if i_iter < 10:
        lr = 1e-2 * lr + i_iter * (lr - 1e-2 * lr) / 10.
    else:
        lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * args.lr_tr_mult
    return lr


def adjust_alpha(i_iter, num_stemps):
    alpha_begin = 1
    alpha_end = 0.01
    decay = (alpha_begin - alpha_end) / num_stemps
    alpha = alpha_begin - decay * i_iter
    return alpha


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003


def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()


def make_one_hot(input, num_classes):
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    result = result.scatter_(1, input, 1)

    return result


def compute_dice_score(preds, labels):
    preds = torch.sigmoid(preds)

    pred_pa = preds[:, 0, :, :, :]
    label_pa = labels[:, 0, :, :, :]
    dice_pa = dice_score(pred_pa, label_pa)

    return dice_pa


def predict_sliding(args, net, image, tile_size, classes, task_id, name):  # tile_size:32x256x256
    image_size = image.shape
    overlap = 1 / 3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(
        np.float32)  # 1x4x155x240x240
    count_predictions = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(
        np.float32)
    full_probs = torch.from_numpy(full_probs)
    count_predictions = torch.from_numpy(count_predictions)
    tile_counter = 0

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)  
                y1 = max(int(y2 - tile_size[1]), 0)  

                img = image[:, :, d1:d2, y1:y2, x1:x2]

                tile_counter += 1
                prediction = net(img.cuda(), task_id).cpu()
                if isinstance(prediction, list):
                    shape = np.array(prediction[0].shape)
                    shape[0] = prediction[0].shape[0] * len(prediction)
                    shape = tuple(shape)
                    preds = torch.zeros(shape).cuda()
                    bs_singlegpu = prediction[0].shape[0]
                    for i in range(len(prediction)):
                        preds[i * bs_singlegpu: (i + 1) * bs_singlegpu] = prediction[i]
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += preds

                else:
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    return full_probs


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def main():
    parser = get_arguments()
    print(parser)
    os.environ["OMP_NUM_THREADS"] = "1"

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if not os.path.exists(args.snapshot_dir):  os.makedirs(args.snapshot_dir)
        logger = get_logger(os.path.join(args.snapshot_dir, 'log'))
        logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
        os.environ['nnUNet_preprocessed'] = args.nnUNet_preprocessed
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)

        dep, wid = map(int, args.dyn_head_dep_wid.split(','))
        dyn_head_dep_wid = (dep, wid)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True

        model = MOTS_model(args, norm_cfg='IN', activation_cfg='relu', num_classes=args.num_classes,
                           weight_std=False, deep_supervision=False, res_depth=args.res_depth, dyn_head_dep_wid=dyn_head_dep_wid)
        print(model)
        print_model_parm_nums(model)
        logger.info(print_model_parm_nums(model))

        model.train()

        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        param_dicts = [
            {
                "params":
                    [p for n, p in model.named_parameters()
                     if not match_name_keywords(n, ['transformer']) and p.requires_grad],
                "lr": args.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           match_name_keywords(n, ['transformer']) and p.requires_grad],
                "lr": args.learning_rate * args.lr_tr_mult,
            }
        ]
        logger.info([f"Param_dicts info. {len(param_dicts[i]['params'])}_(lr:{param_dicts[i]['lr']})" for i in range(len(param_dicts))])
        if args.optimizer == 'adamw':
            logger.info("Using Adamw optimizer!")
            optimizer = torch.optim.AdamW(param_dicts, args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            logger.info("Using SGD optimizer!")
            optimizer = torch.optim.SGD(param_dicts, args.learning_rate, weight_decay=3e-5, momentum=0.99, nesterov=True)
        else:
            logger.info("@No optimizer defined!")

        if args.FP16:
            logger.info("Note: Using FP16 during training************")
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        to_restore = {"epoch": 0}
        if args.FP16:
            utils.restart_from_checkpoint(
                os.path.join(args.snapshot_dir, "checkpoint.pth"),
                run_variables=to_restore,
                state_dict=model,
                optimizer=optimizer,
            )
        else:
            utils.restart_from_checkpoint(
                os.path.join(args.snapshot_dir, "checkpoint.pth"),
                run_variables=to_restore,
                state_dict=model,
                optimizer=optimizer,
                amp=amp,
            )

        loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).to(device)
        loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, ignore_index=255).to(device)

        trainloader, train_sampler = engine.get_train_loader(MOTSTrainDataSet(args.data_dir, args.train_list, max_iters=None,
                        crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror), collate_fn=my_collate)

        all_tr_loss = []
        start_epoch = to_restore["epoch"]
        for epoch in range(start_epoch, args.num_epochs):
            start_time = time.time()
            if engine.distributed:
                train_sampler.set_epoch(epoch)

            epoch_loss = []
            adjust_learning_rate(optimizer, epoch, args)

            prefetcher = data_prefetcher(trainloader)
            batch = prefetcher.next()
            while batch is not None:
                images = batch['image']
                labels = batch['label']
                task_ids = batch['task_id']

                optimizer.zero_grad()
                preds = model(images, task_ids)
                del images
                if args.using_transformer:
                    N_b, N_q, N_c, N_d, N_h, N_w = preds.shape
                    preds_convert = torch.zeros(size=(N_b, N_c, N_d, N_h, N_w)).cuda()

                    for i_b in range(N_b):
                        preds_convert[i_b] = preds[i_b, int(task_ids[i_b])]

                else:
                    preds_convert = preds

                term_seg_Dice = loss_seg_DICE.forward(preds_convert, labels)
                term_seg_BCE = loss_seg_CE.forward(preds_convert, labels)
                term_all = term_seg_Dice + term_seg_BCE

                reduce_all = engine.all_reduce_tensor(term_all)

                if args.FP16:
                    with amp.scale_loss(term_all, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    term_all.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

                epoch_loss.append(float(reduce_all))
                del labels

                batch = prefetcher.next()

            epoch_loss = np.mean(epoch_loss)

            all_tr_loss.append(epoch_loss)

            end_time = time.time()

            if (args.local_rank == 0):
                logger.info('Epoch_sum {}: lr1 = {:.4}, lr2 = {:.4}, loss_Sum = {:.4}, run_time = {:.0f}s'.format(epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], epoch_loss.item(), (end_time - start_time)))
                
            if args.local_rank == 0 and epoch%10==0:
                if args.FP16:
                    save_dict = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'amp': amp.state_dict()
                    }
                else:  
                    save_dict = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1
                    }
                torch.save(save_dict, osp.join(args.snapshot_dir, 'checkpoint.pth'))

        end = timeit.default_timer()
        logger.info(f"{end - start} seconds")


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch['image'] = torch.from_numpy(self.next_batch['image']).cuda(non_blocking=True)
            self.next_batch['label'] = torch.from_numpy(self.next_batch['label']).cuda(non_blocking=True)
            self.next_batch['task_id'] = torch.from_numpy(self.next_batch['task_id']).cuda(non_blocking=True)
            self.next_batch['image'] = self.next_batch['image'].float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            batch['image'].record_stream(torch.cuda.current_stream())
            batch['label'].record_stream(torch.cuda.current_stream())
            batch['task_id'].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch


if __name__ == '__main__':
    main()