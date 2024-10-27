# coding=utf-8
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import logging
import argparse
import os
import random
import numpy as np
import time
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from datetime import timedelta
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

import torch
import torch.distributed as dist
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'
# dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)
# # dist.init_process_group('gloo', world_size=1)

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.msdbn import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.origidata_utils import get_loader
from utils.dist_util import get_world_size

logger = logging.getLogger(__name__)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def reduce_mean(tensor, nprocs):
    if nprocs > 1 and dist.is_initialized():
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt
    return tensor 

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    checkpoint = {
            'model': model_to_save.state_dict(),
        }
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
def setup(args):
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=14, smoothing_value=args.smoothing_value)
    pretrained_model = torch.load('autodl-tmp/ba/Shark_denseNet_checkpoint.bin')['model']
    model.load_state_dict(pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()
    
    val_loss_history =[] # list to store validation losses 
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(args.device)
    #dist.barrier()
    #val_accuracy = reduce_mean(accuracy, args.nprocs)
    val_accuracy = accuracy
    all_preds = all_preds.tolist()
    all_label = all_label.tolist()
    f1score=f1_score(all_label, all_preds, average='macro')

    
    cm = confusion_matrix(all_label, all_preds, labels=list(range(14)))
    precision = precision_score(all_label, all_preds, average='macro')
    recall = recall_score(all_label, all_preds, average='macro')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % val_accuracy)
    logger.info("Valid F1 score so far: %.4f" % f1score)
    logger.info("Precision: %.4f" % precision)
    logger.info("Recall: %.4f" % recall)
    logger.info("Confusion Matrix:\n%s" % cm)
    if args.local_rank in [-1, 0]:
        disp.plot(cmap='viridis')
        plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
        writer.add_scalar("test/accuracy", scalar_value=val_accuracy, global_step=global_step)
        writer.add_scalar("test/precision", scalar_value=precision, global_step=global_step)
        writer.add_scalar("test/recall", scalar_value=recall, global_step=global_step)
        plt.figure(figsize=(10, 5))
    # Ploting the validation loss graph
    plt.plot(val_loss_history, color='orange', label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Time')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "validation_loss_graph.png"))
    plt.close()
    return val_accuracy, f1score, precision, recall
def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True, default="ASLO",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["ASLO", "SHARK","WILDFISH"], default="SHARK",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='ASLO')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1000, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--learning_rate", default=4e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=60000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default= -1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    args = parser.parse_args()
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = 1#torch.cuda.device_count()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))
    set_seed(args)
    args, model = setup(args)
    train_loader, test_loader = get_loader(args)
    global_step, best_acc, best_f1score = 0, 0, 0
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    train_losses = AverageMeter()
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    global_step = 0
    train_loader, test_loader = get_loader(args)
    valid(args, model, writer, test_loader, global_step)
    def save_model(args, model):
        model_to_save = model.module if hasattr(model, 'module') else model
        model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
        logger.info(f"Saving model to {model_checkpoint}")
        checkpoint = {
            'model': model_to_save.state_dict(),
        }
        torch.save(checkpoint, model_checkpoint)
        logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
    save_model(args, model)
    logger.info("Final model saved.")
if __name__ == "__main__":
    main()