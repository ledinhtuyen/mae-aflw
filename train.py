import argparse
import yaml
import torch
import math
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from dataset.AFLW import AFLW
from utils.utils import get_mae_model, seed_anything, get_logger, pretrain_train_one_epoch, pretrain_val_one_epoch, save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    parser.add_argument('--resume', help='resume from checkpoint', default=None, type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = yaml.load(open(args.cfg, 'r'), Loader=yaml.FullLoader)
    seed_anything()

    step_dict = {
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # Wandb
    config_dict = {
        "learning_rate": (cfg['BASE_LR'] * cfg['BATCH_SIZE'] / 256),
        'weight_decay': cfg['WEIGHT_DECAY'],
        'warmup': cfg['WARMUP_EPOCH'],
        "epochs": cfg['TOTAL_EPOCH']
    }
    logger = get_logger(cfg['PROJECT_NAME'], cfg['WANDB_API_KEY'], config_dict)

    # Data
    train_dataset = AFLW(cfg['TRAIN_ANNOTATION_FILE'], cfg['TRAIN_IMG_DIR'], transform=Compose([ToTensor()])) # Normalize(meanstd['mean'], meanstd['std'])
    val_dataset = AFLW(cfg['TEST_ANNOTATION_FILE'], cfg['TEST_IMG_DIR'], transform=Compose([ToTensor()]))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=True, num_workers=cfg['NUM_WORKERS'])
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=True, num_workers=cfg['NUM_WORKERS'])

    # Model
    model = get_mae_model()

    device = None

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model.to(device)

    # Optimizer, Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr= cfg['BASE_LR'] * cfg['BATCH_SIZE'] / 256, betas=(0.9, 0.95), weight_decay=cfg['WEIGHT_DECAY'])
    lr_func = lambda epoch: min((epoch + 1) / (cfg['WARMUP_EPOCH']) + 1e-8, 0.5 * (math.cos(epoch / cfg['TOTAL_EPOCH'] * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    # Train
    last_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(cfg['CHECKPOINT_PATH'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        last_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print('Resume from checkpoint: {} (epoch {}, loss {})'.format(args.resume, last_epoch, loss))

    for epoch in range(last_epoch, cfg['TOTAL_EPOCH']):
        loss = pretrain_train_one_epoch(model, epoch, train_dataloader, optimizer, step_dict, device, lr_scheduler=lr_scheduler)
        pretrain_val_one_epoch(model, epoch, val_dataset, step_dict, device)
        save_checkpoint(model, optimizer, epoch, loss, step_dict, cfg['CHECKPOINT_PATH'], lr_scheduler=lr_scheduler)

if __name__ == '__main__':
    main()