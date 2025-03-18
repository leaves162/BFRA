import os
import shutil
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (MultiStepLR, ExponentialLR,
                                      CosineAnnealingWarmRestarts,
                                      CosineAnnealingLR)
from config import device
import datetime
import collections
import time
import io
from collections import defaultdict, deque
import torch
import torch.distributed as dist
sigmoid = nn.Sigmoid()

def get_initial_optimizer(model_params, args):
    if args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'momentum':
        optimizer = torch.optim.SGD(model_params, lr=args.learning_rate,
                                    momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'ada':
        optimizer = torch.optim.Adadelta(model_params, lr=args.learning_rate)
    else:
        raise RuntimeError('No optimizer has been setted!')
    return optimizer

class UniformStepLR(object):
    def __init__(self, optimizer, args, start_iter):
        self.iter = start_iter
        self.max_iter = args.epochs
        step_iters = self.compute_milestones(args)
        self.lr_scheduler = MultiStepLR(
            optimizer, milestones=step_iters,
            last_epoch=start_iter-1, gamma=1e-1)
    def step(self, _iter):
        self.iter += 1
        self.lr_scheduler.step()
        stop_training = self.iter >= self.max_iter
        return stop_training
    def compute_milestones(self, args):
        # pdb.set_trace()
        lr_decay_step_freq=10
        max_iter = args.epochs
        step_size = max_iter / lr_decay_step_freq
        step_iters = [0]
        while step_iters[-1] < max_iter:
            step_iters.append(step_iters[-1] + step_size)
        return step_iters[1:]

class ExpDecayLR(object):
    def __init__(self, optimizer, args, start_iter):
        self.iter = start_iter
        self.max_iter = args.epochs
        self.start_decay_iter = 10000
        gamma = self.compute_gamma(args)
        schedule_start = max(start_iter - self.start_decay_iter, 0) - 1
        self.lr_scheduler = ExponentialLR(optimizer, gamma=gamma,
                                          last_epoch=schedule_start)
    def step(self, _iter):
        self.iter += 1
        if _iter > self.start_decay_iter:
            self.lr_scheduler.step()
        stop_training = self.iter >= self.max_iter
        return stop_training
    def compute_gamma(self, args):
        last_iter, start_iter = self.max_iter, self.start_decay_iter
        start_lr, last_lr = args.learning_rate, 8e-5
        return np.power(last_lr / start_lr, 1 / (last_iter - start_iter))

class CosineAnnealRestartLR(object):
    def __init__(self, optimizer, args, start_iter):
        self.iter = start_iter
        self.max_iter = args.epochs
        self.lr_scheduler = CosineAnnealingWarmRestarts(optimizer, args.valid_frequency)
        #self.lr_scheduler = CosineAnnealingLR(
        #optimizer, args['train.cosine_anneal_freq'], last_epoch=start_iter-1)

    def step(self, _iter):
        self.iter += 1
        self.lr_scheduler.step(_iter)
        stop_training = self.iter >= self.max_iter
        return stop_training

def lr_scheduler(optimizer, restored_epoch, args):
    if args.lr_scheduler=='cosine':
        scheduler=CosineAnnealRestartLR(optimizer, args, restored_epoch)
    elif args.lr_scheduler=='exp':
        scheduler=ExpDecayLR(optimizer, args, restored_epoch)
    else:
        scheduler=UniformStepLR(optimizer, args, restored_epoch)
    return scheduler

def if_contains_nan(tensor_data):
    if torch.isnan(tensor_data).any():
        raise RuntimeError('tensor has nan values, please check the code!!!')
    return

def if_contains_zero(tensor_data):
    for i in range(tensor_data.shape[0]):
        #print('orig:',tensor_data[i,:].cpu().shape, tensor_data[i,:].cpu())
        #print('zero:',torch.zeros(tensor_data[i,:].shape))
        if torch.equal(tensor_data[i,:].cpu(),torch.zeros(tensor_data[i,:].shape)):
            return True, i
    return False, 0

class WeightAnnealing(nn.Module):
    """WeightAnnealing"""
    def __init__(self, T, alpha=10):
        super(WeightAnnealing, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, t, opt='exp'):
        if t > self.T:
            t=t%self.T
        if opt == 'exp':
            return 1-np.exp(self.alpha*((t)/self.T-1))
        if opt == 'log':
            return np.exp(-(t)/self.T*self.alpha)
        if opt == 'linear':
            return 1-(t)/self.T

def calculate_var(acc_list):
    acc_list=np.array(acc_list) * 100
    acc_mean = acc_list.mean()
    acc_var = (1.96 * acc_list.std()) / np.sqrt(len(acc_list))
    acc_var = round(acc_var, 2)
    acc_mean = round(acc_mean, 2)
    return acc_mean, acc_var

def vote_for_preds(arr):
    preds={}
    for i in arr:
        if i not in list(preds.keys()):
            preds[i]=0
        else:
            preds[i]=preds[i]+1
    #print(preds)
    sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    #print(sorted_preds)
    pred=sorted_preds[0][0]
    return pred









def cosine_sim(embeds, prots):
    prots = prots.unsqueeze(0)
    embeds = embeds.unsqueeze(1)
    return F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30)

class CosineClassifier(nn.Module):
    def __init__(self, n_feat, num_classes):
        super(CosineClassifier, self).__init__()
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        weight = torch.FloatTensor(n_feat, num_classes).normal_(
                    0.0, np.sqrt(2.0 / num_classes))
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-12)
        weight = torch.nn.functional.normalize(self.weight, p=2, dim=0, eps=1e-12)
        cos_dist = x_norm @ weight
        scores = self.scale * cos_dist
        return scores

    def extra_repr(self):
        s = 'CosineClassifier: input_channels={}, num_classes={}; learned_scale: {}'.format(
            self.weight.shape[0], self.weight.shape[1], self.scale.item())
        return s


class CosineConv(nn.Module):
    def __init__(self, n_feat, num_classes, kernel_size=1):
        super(CosineConv, self).__init__()
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        weight = torch.FloatTensor(num_classes, n_feat, 1, 1).normal_(
                    0.0, np.sqrt(2.0/num_classes))
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        x_normalized = torch.nn.functional.normalize(
            x, p=2, dim=1, eps=1e-12)
        weight = torch.nn.functional.normalize(
            self.weight, p=2, dim=1, eps=1e-12)

        cos_dist = torch.nn.functional.conv2d(x_normalized, weight)
        scores = self.scale * cos_dist
        return scores

    def extra_repr(self):
        s = 'CosineConv: num_inputs={}, num_classes={}, kernel_size=1; scale_value: {}'.format(
            self.weight.shape[0], self.weight.shape[1], self.scale.item())
        return s




def labels_to_episode_labels(labels):
    uni_labels = labels.unique()
    eposide_labels = torch.zeros(list(labels.size())).to(labels.device)
    for i in range(len(uni_labels)):
        eposide_labels[labels == uni_labels[i]] = i
    return eposide_labels





def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device, non_blocking=True)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def std(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.std().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.device != 'cuda':
        print('Not using distributed mode')
        args.distributed = False
        return

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

    args.distributed = True

    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)

    #torch.distributed.barrier()
    torch.distributed.barrier(device_ids=[args.gpu])
    torch.cuda.set_device(args.gpu)
    setup_for_distributed(args.rank == 0)
