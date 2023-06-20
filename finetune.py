import argparse
import yaml
import math
import torch
import wandb
from torchvision.transforms import ToTensor, Compose
from dataset.AFLW import AFLW
from torch.utils.data import DataLoader
from utils.utils import get_unetr_model, seed_anything, get_logger, finetune_train_one_epoch, finetune_val_one_epoch, save_checkpoint
from loss.AWingLoss import Loss_weighted
import matplotlib.pyplot as plt

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
  parser.add_argument('--pretrained', help='pretrained model filename', default=None, type=str)
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
  finetune_train_dataset = AFLW(cfg['TRAIN_ANNOTATION_FILE'], cfg['TRAIN_IMG_DIR'], max_samples=cfg['NUM_SAMPLES'],transform=Compose([ToTensor()])) # Normalize(meanstd['mean'], meanstd['std'])
  finetune_val_dataset = AFLW(cfg['TEST_ANNOTATION_FILE'], cfg['TEST_IMG_DIR'], max_samples=cfg['NUM_SAMPLES'], transform=Compose([ToTensor()]))

  finetune_train_dataloader = DataLoader(finetune_train_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=True, num_workers=cfg['NUM_WORKERS'])
  finetune_val_dataloader = DataLoader(finetune_val_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=True, num_workers=cfg['NUM_WORKERS'])

  # Model
  if args.pretrained is not None:
    model = get_unetr_model(pretrained=args.pretrained)
  else:
    model = get_unetr_model()

  # Device
  device = None
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  model.to(device)

  # Loss
  criterion = Loss_weighted()

  # Optimizer, Scheduler
  optimizer = torch.optim.AdamW(model.parameters(), lr= cfg['BASE_LR'] * cfg['BATCH_SIZE'] / 256, betas=(0.9, 0.95), weight_decay=cfg['WEIGHT_DECAY'])
  lr_func = lambda epoch: min((epoch + 1) / (cfg['WARMUP_EPOCH']) + 1e-8, 0.5 * (math.cos(epoch / cfg['TOTAL_EPOCH'] * math.pi) + 1))
  lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

  # Train
  last_epoch = 0
  loss = 0
  nme = 0
  if args.resume is not None and args.pretrained is None:
    checkpoint = torch.load(cfg['CHECKPOINT_PATH'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    nme = checkpoint['nme']
    step_dict = checkpoint['step_dict']
  elif args.resume is not None and args.pretrained is not None:
    print('Need only pretrained or resume, not both.')
    return
  elif args.resume is None and args.pretrained is None:
    print('Need pretrained or resume.')
    return
  
  for epoch in range(last_epoch, cfg['TOTAL_EPOCH']):
    train_loss, nme_train = finetune_train_one_epoch(model, epoch, finetune_train_dataloader, criterion, optimizer, step_dict, device, lr_scheduler)
    val_loss, nme_val = finetune_val_one_epoch(model, epoch, finetune_val_dataloader, criterion, step_dict, device)
    wandb.log({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)
    wandb.log({'nme_train': nme_train, 'nme_val': nme_val}, step=epoch)
    save_checkpoint(model, optimizer, epoch, train_loss, step_dict, cfg['CHECKPOINT_PATH'], nme=nme_train, lr_scheduler=lr_scheduler)

if __name__ == '__main__':
    main()