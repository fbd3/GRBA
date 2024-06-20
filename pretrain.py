import argparse, copy, os, time, warnings, psutil

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

import dgl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from gcc.contrastive.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from gcc.contrastive.memory_moco import MemoryMoCo
from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS,
    GraphClassificationDataset,
    GraphClassificationDatasetLabeled,   
    LoadBalanceGraphDataset,
    Load_injected_GraphDataset,
    worker_init_fn,
)
from gcc.datasets.data_util import batcher, labeled_batcher
from gcc.models import GraphEncoder
from gcc.utils.misc import AverageMeter, adjust_learning_rate, warmup_linear

from config import parse_option_in_config


# ==============================================
# =================== parser ===================
# opt_model_name_and_load_path_and_tb_folder
def option_update(opt):
    opt.model_name = "{}_moco_{}_{}_{}_layer_{}_lr_{}_decay_{}_bsz_{}_hid_{}_samples_{}_nce_t_{}_nce_k_{}_rw_hops_{}_restart_prob_{}_aug_{}_ft_{}_deg_{}_pos_{}_momentum_{}".format(
        opt.exp, opt.moco,
        opt.dataset,
        opt.model, opt.num_layer,
        opt.learning_rate, opt.weight_decay,
        opt.batch_size, opt.hidden_size, opt.num_samples,
        opt.nce_t, opt.nce_k, 
        opt.rw_hops, opt.restart_prob, opt.aug, 
        opt.finetune,
        opt.degree_embedding_size, opt.positional_embedding_size, opt.alpha,
    )
    if opt.load_path is None:
        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)
    else:
        opt.model_folder = opt.load_path

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    return opt
# ==============================================



# copy weights from model to model_ema When pre-training MOCO
def moment_update(model, model_ema, m):
    """ 
        @ model_ema = m * model_ema + (1 - m) model 
    """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)



def clip_grad_norm(params, max_norm):  # default max_norm = 1.0
    """
        @ Clips gradient norm.
    """
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt( sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None) )
    
    
def train_moco( epoch, train_loader, model, model_ema, contrast, criterion, optimizer, sw, opt ):
    """ 
        one epoch training for moco 
    """
    n_batch = train_loader.dataset.total // opt.batch_size
    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.train()

    model_ema.apply(set_bn_train)

    batch_time       = AverageMeter()
    data_time        = AverageMeter()
    loss_meter       = AverageMeter()
    epoch_loss_meter = AverageMeter()
    prob_meter       = AverageMeter()
    graph_size       = AverageMeter()
    gnorm_meter      = AverageMeter()
    max_num_nodes    = 0
    max_num_edges    = 0

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, graph_k = batch
        graph_q.to(torch.device('cpu'))
        graph_k.to(torch.device('cpu'))

        bsz = graph_q.batch_size

        if opt.moco:  # ===================Moco forward=====================
            feat_q = model(graph_q)
            with torch.no_grad():
                feat_k = model_ema(graph_k)

            out = contrast(feat_q, feat_k)
            #//print(feat_k.shape, feat_q.shape)
            prob = out[:, 0].mean()
        else: # ===================Negative sampling forward=====================
            pass
        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)

        # ===================backward=====================
        optimizer.zero_grad()
        loss = criterion(out)
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), opt.clip_norm)

        global_step = epoch * n_batch + idx
        lr_this_step = opt.learning_rate * warmup_linear( global_step / (opt.epochs * n_batch), 0.1 )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        graph_size.update( (graph_q.number_of_nodes() + graph_k.number_of_nodes()) / 2.0 / bsz, 2 * bsz )
        gnorm_meter.update(grad_norm, 1)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())
        # print(max_num_nodes)
        
        if opt.moco:
            moment_update(model, model_ema, opt.alpha)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BatchTim {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DataTime {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "prob {prob.val:.3f} ({prob.avg:.3f})\t"
                "GraphSize {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch, idx + 1, n_batch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    prob=prob_meter,
                    graph_size=graph_size, mem=mem.used / 1024 ** 3, ))
            #  print(out[0].abs().max())

        # tensorboard logger
        if (idx + 1) % opt.tb_freq == 0:
            global_step = epoch * n_batch + idx
            sw.add_scalar("moco_loss", loss_meter.avg, global_step)
            sw.add_scalar("moco_prob", prob_meter.avg, global_step)
            sw.add_scalar("graph_size", graph_size.avg, global_step)
            sw.add_scalar("graph_size/max", max_num_nodes, global_step)
            sw.add_scalar("graph_size/max_edges", max_num_edges, global_step)
            sw.add_scalar("gnorm", gnorm_meter.avg, global_step)
            sw.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
            loss_meter.reset()
            prob_meter.reset()
            graph_size.reset()
            gnorm_meter.reset()
            max_num_nodes, max_num_edges = 0, 0
    return epoch_loss_meter.avg






def main(args):
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    args = option_update(args)
    #//print(args)

    mem = psutil.virtual_memory() # 专门用来获取操作系统以及硬件相关的信息，比如：CPU、磁盘、网络、内存等等。
    # print(psutil.cpu_count())                # 6
    # print(psutil.cpu_count(logical=False))   # 6
    # print(psutil.virtual_memory())           # svmem(total=17179869184, available=8675700736, percent=49.5, used=7589683200, free=2697805824, active=3886612480, inactive=5761380352, wired=3703070720)
    # print(psutil.swap_memory())              # sswap(total=2147483648, used=1332740096, free=814743552, percent=62.1, sin=124188454912, sout=1259970560)
    print("========= before construct dataset, memory used ", mem.used / 1024 ** 3, " GB =========")
    
    # When pre-training at first
    if args.dataset == "dgl":
        print("LOADING DATASET DEFAULT DGL")
        train_dataset = Load_injected_GraphDataset(#Version-Update train_dataset = LoadBalanceGraphDataset(
            rw_hops=args.rw_hops,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
            num_workers=args.num_workers,
            num_samples=args.num_samples,
            dgl_graphs_file="./data/small.bin",
            num_copies=args.num_copies,
        )
    else: #  GraphClassificationDataset and  (not in GRAPH_CLASSIFICATION_DSETS) NodeClassificationDataset
        if args.dataset in GRAPH_CLASSIFICATION_DSETS:
            train_dataset = GraphClassificationDataset(
                dataset=args.dataset,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )

    mem = psutil.virtual_memory()
    print("========= before construct dataloader, memory used ", mem.used / 1024 ** 3, " GB ========")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=labeled_batcher() if args.finetune else batcher(),                      #如果微调 带标签的 batch， 否则 对比学习
        shuffle=True if args.finetune else False,                                          # 如果微调 就打乱
        num_workers=args.num_workers,
        worker_init_fn=None if args.finetune or args.dataset != "dgl" else worker_init_fn, #如果微调 或者不使用 dgl smallbin 就为空
    )
    mem = psutil.virtual_memory()
    print("========= before training, memory used ", mem.used / 1024 ** 3, " GB =========")



    # create model and optimizer
    # n_data = train_dataset.total
    n_data = None

    model, model_ema = [
        GraphEncoder(
            positional_embedding_size=args.positional_embedding_size,
            max_node_freq            =args.max_node_freq,
            max_edge_freq            =args.max_edge_freq,
            max_degree               =args.max_degree,
            freq_embedding_size      =args.freq_embedding_size,
            degree_embedding_size    =args.degree_embedding_size,
            output_dim               =args.hidden_size,
            node_hidden_dim          =args.hidden_size,
            edge_hidden_dim          =args.hidden_size,
            num_layers               =args.num_layer,
            num_step_set2set         =args.set2set_iter,
            num_layer_set2set        =args.set2set_lstm_layer,
            norm                     =args.norm,
            gnn_model                =args.model,
            degree_input             =True,
        )
        for _ in range(2)
    ]

    # copy weights from `model' to `model_ema'
    if args.moco:
        moment_update(model, model_ema, 0)

    # --------------------------------- set the contrast memory and criterion -----------------------------
    contrast = MemoryMoCo( args.hidden_size, n_data, args.nce_k, args.nce_t, use_softmax=True )  
    # -----------------------------------------------32          0.07        ------------------------------
    #//print(contrast)
    criterion = NCESoftmaxLoss() if args.moco else NCESoftmaxLossNS() # 因为是 MOCO 的 使用 NCESoftmaxLoss()

    # Optimizer choice
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr           = args.learning_rate,
            betas        = (args.beta1, args.beta2),
            weight_decay = args.weight_decay,
        )
    else:
        raise NotImplementedError


    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        contrast.load_state_dict(checkpoint["contrast"])
        if args.moco:
            model_ema.load_state_dict(checkpoint["model_ema"])

        print( "=> loaded successfully '{}' (epoch {})".format(args.resume, checkpoint["epoch"]) )
        del checkpoint

    # tensorboard
    #  logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    sw = SummaryWriter(args.tb_folder)
    #  plots_q, plots_k = zip(*[train_dataset.getplot(i) for i in range(5)])
    #  plots_q = torch.cat(plots_q)
    #  plots_k = torch.cat(plots_k)
    #  sw.add_images('images/graph_q', plots_q, 0, dataformats="NHWC")
    #  sw.add_images('images/graph_k', plots_k, 0, dataformats="NHWC")

    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        loss = train_moco( epoch,  train_loader,  model,  model_ema, contrast, criterion, optimizer,  sw,   args )
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        if epoch % args.save_freq == 0: # save model
            print("==> Saving...")
            state = {   "opt": args,
                        "model": model.state_dict(),
                        "contrast": contrast.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,    }
            if args.moco:
                state["model_ema"] = model_ema.state_dict()
                
            save_file = os.path.join(args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch))
            torch.save(state, save_file)
            del state
    # FOR training routine end


if __name__ == "__main__":
    warnings.simplefilter("once", UserWarning)
    args = parse_option_in_config()
    args.gpu = args.gpu[0]
    main(args)