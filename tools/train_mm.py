import os
import torch
import argparse
import yaml
import time
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.datasets import *
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from val_mm import evaluate

from sam2.build_sam import build_sam2
from sam2.sam_lora_language import LoRA_Sam
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import clip

def main(cfg, gpu, save_dir):
    start = time.time()
    best_mIoU = 0.0
    best_epoch = 0
    num_workers = 8
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    resume_path = cfg['MODEL']['RESUME']
    gpus = int(os.environ.get('WORLD_SIZE', 1))
    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'])
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'])
    class_names = trainset.CLASSES
    model_clip, _ = clip.load("ViT-B/32")
    checkpoint = model_cfg['PRETRAINED']
    model_cf = model_cfg['MODEL_CONFIG']
    sam2 = build_sam2(model_cf, checkpoint)
    model = LoRA_Sam(model_clip, sam2, r=train_cfg['RANK'])

    for param in model.sam.obj_ptr_proj.parameters():
        param.requires_grad = False
    for param in model.sam.sam_mask_decoder.iou_prediction_head.parameters():
        param.requires_grad = False
    for param in model.sam.sam_mask_decoder.pred_obj_score_head.parameters():
        param.requires_grad = False
    for param in model.sam.memory_attention.parameters():
        param.requires_grad = False
    for param in model.sam.memory_encoder.parameters():
        param.requires_grad = False
    for param in model.sam.sam_prompt_encoder.parameters():
        param.requires_grad = False
    for param in model.model_clip.parameters():
        param.requires_grad = False

    resume_checkpoint = None
    if os.path.isfile(resume_path):
        resume_checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        msg = model.load_state_dict(resume_checkpoint['model_state_dict'])
        logger.info(msg)
    model = model.to(device)

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE'] // gpus
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    start_epoch = 0
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr,
                              optim_cfg['WEIGHT_DECAY'])

    scheduler = get_scheduler(
        sched_cfg['NAME'], optimizer, int((epochs + 1) * iters_per_epoch),
        sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO']
    )

    if train_cfg['DDP']:
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None
        model = DDP(model, device_ids=[gpu], output_device=0, find_unused_parameters=True)
    else:
        sampler = RandomSampler(trainset)
        sampler_val = None

    if resume_checkpoint:
        start_epoch = resume_checkpoint['epoch'] - 1
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        loss_sam = resume_checkpoint['loss_sam']
        prototype_loss = resume_checkpoint['prototype_loss']
        loss_aux = resume_checkpoint['loss_aux']
        kl_loss = resume_checkpoint['kl_loss']
        loss = resume_checkpoint['loss']
        best_mIoU = resume_checkpoint['best_miou']

    trainloader = DataLoader(
        trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers,
        drop_last=True, pin_memory=False, sampler=sampler
    )

    valloader = DataLoader(
        valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers,
        pin_memory=False, sampler=sampler_val
    )

    scaler = GradScaler(enabled=train_cfg['AMP'])
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer = SummaryWriter(str(save_dir))
        logger.info('================== model structure =====================')
        logger.info(model)
        logger.info('================== training config =====================')
        logger.info(cfg)

    for epoch in range(start_epoch, epochs):
        model.train()
        if train_cfg['DDP']:
            sampler.set_epoch(epoch)

        train_loss = 0.0
        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch,
                    desc=f"Epoch: [{epoch + 1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, (sample, lbl) in pbar:
            optimizer.zero_grad(set_to_none=True)
            sample = [x.to(device) for x in sample]
            lbl = lbl.to(device)

            with autocast(enabled=train_cfg['AMP']):
                m_output, output, prototype_loss, kl_loss = model(sample, lbl, multimask_output=True)
                loss_sam = loss_fn(m_output, lbl)
                loss_aux = loss_fn(output, lbl)
                loss_aux = 1 * loss_aux
                prototype_loss = 1 * prototype_loss.mean()
                kl_loss = 1 * kl_loss.mean()
                loss = loss_sam + loss_aux + prototype_loss + kl_loss
                loss = loss.mean()


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            if lr <= 1e-8:
                lr = 1e-8

            loss_aux += loss_aux.item()
            loss_sam += loss_sam.item()
            kl_loss += kl_loss.item()
            prototype_loss += prototype_loss.item()
            train_loss += loss.item()

            pbar.set_description( f"Epoch: [{epoch + 1}/{epochs}] Iter: [{iter + 1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter + 1):.8f}")

        train_loss /= iter + 1
        loss_aux /= iter + 1
        loss_sam /= iter + 1
        kl_loss /= iter + 1
        prototype_loss /= iter + 1

        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/loss_aux', loss_aux, epoch)
            writer.add_scalar('train/loss_sam', loss_sam, epoch)
            writer.add_scalar('train/kl_loss', kl_loss, epoch)
            writer.add_scalar('train/prototype_loss', prototype_loss, epoch)
        torch.cuda.empty_cache()

        if ((epoch + 1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch + 1) > train_cfg['EVAL_START']) or (
                epoch + 1) == epochs:

            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                acc, macc, _, _, ious, miou = evaluate(model, valloader, device)
                writer.add_scalar('val/mIoU', miou, epoch)
                if miou > best_mIoU:

                    prev_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    prev_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)

                    best_mIoU = miou
                    best_epoch = epoch + 1

                    cur_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    cur_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"

                    torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), cur_best)

                    torch.save({'epoch': best_epoch,
                                'model_state_dict': model.module.state_dict() if train_cfg[
                                    'DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'loss_aux': loss_aux,
                                'loss_sam': loss_sam,
                                'kl_loss': kl_loss,
                                'prototype_loss': prototype_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                }, cur_best_ckp)

                    logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))

                logger.info(f"Current epoch:{epoch} mIoU: {miou} Best mIoU: {best_mIoU}")

        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.close()

        pbar.close()

        end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../configs/pst_rgbt.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    BATCH_SIZE = cfg['TRAIN']['BATCH_SIZE']
    RANK = cfg['TRAIN']['RANK']
    LR = cfg['OPTIMIZER']['LR']
    exp_name = '_'.join(
        [cfg['DATASET']['NAME'], model, modals, str(BATCH_SIZE), str(RANK), str(LR)])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    if os.path.isfile(cfg['MODEL']['RESUME']):
        save_dir = Path(os.path.dirname(cfg['MODEL']['RESUME']))
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')
    main(cfg, gpu, save_dir)
    cleanup_ddp()
