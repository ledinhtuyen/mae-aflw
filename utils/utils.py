import torch
import numpy as np
import math
import random
import wandb
from tqdm import tqdm
from model.MAE import patchify, unpatchify
from einops import rearrange
from model.MAE import *

def seed_anything(seed=2023):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def get_logger(project_name, api_key, config_dict=None):
    wandb.login(key=api_key)
    wandb.init(project=project_name, config=config_dict, resume=True)
    return wandb

def get_model(encoder_embedding_dim=768, encoder_layers=12, n_heads_encoder_layer=12, decoder_embedding_dim=512, decoder_layers=4, n_heads_decoder_layer=16, patch_size=4, num_patches=16):
    # num_patches (on width or height)= image_size // patch_size
    model = MaskedAutoEncoder(
        Transformer(embedding_dim=encoder_embedding_dim, n_layers=encoder_layers, n_heads=n_heads_encoder_layer, feedforward_dim=encoder_embedding_dim*4),
        Transformer(embedding_dim=decoder_embedding_dim, n_layers=decoder_layers, n_heads=n_heads_decoder_layer, feedforward_dim=decoder_embedding_dim*4),
        encoder_embedding_dim=encoder_embedding_dim, decoder_embedding_dim=decoder_embedding_dim, patch_size=patch_size, num_patches=num_patches
    )
    return model

# ----------------------------------------------------------------
# Adopt from https://github.com/HRNet/HRNet-Facial-Landmark-Detection/blob/master/lib/core/evaluation.py
def compute_nme(preds, meta, IMG_SIZE=64):

    targets = meta['pts'] * IMG_SIZE
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / L

    return rmse

def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    scores: [batch_size, num_joints, height, width]
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def decode_preds(output, res=(64, 64)):
    coords = get_preds(output)  # float type

    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds

# ---------------------------Pretrain-------------------------------------

def pretrain_train_one_epoch(model, epoch, dataloader, optimizer, step_dict, device, lr_scheduler = None):
    model.train()
    losses = []
    print('Epoch ' + str(epoch))
    data_iterator = tqdm(dataloader)
    for x, y, _, _ in data_iterator:
        x = x.to(device)
        image_patches = patchify(x, patch_size=4)
        predicted_patches, mask = model(x)
        loss = torch.sum(torch.mean(torch.square(image_patches - predicted_patches), dim=-1) * mask) / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        data_iterator.set_postfix(loss=np.mean(losses))
        step_dict['train_global_steps'] += 1
    lr_scheduler.step()
    avg_loss = sum(losses) / len(losses)
    print(f'In epoch {epoch}, average traning loss is {avg_loss}.')
    wandb.log({'mae_loss': avg_loss}, step=epoch)

def pretrain_val_one_epoch(model, epoch, val_dataset, step_dict, device):
    model.eval()
    with torch.no_grad():
        val_img = torch.stack([val_dataset[i][0] for i in range(16)])
        val_img = val_img.to(device)
        predicted_val_img, mask = model(val_img)
        mask = unpatchify(mask.unsqueeze(-1).tile(dims=(1,1,48)), 4)
        predicted_val_img = unpatchify(predicted_val_img, 4)
        predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
        img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
        img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
        images = wandb.Image(img)
        wandb.log({"mae_image": images}, step=epoch)

# ---------------------------Finetune-------------------------------------

def finetune_train_one_epoch(model, epoch, dataloader, criterion, optimizer, step_dict, device, lr_scheduler = None):
    model.train()
    nme_count = 0
    nme_batch_sum = 0
    losses = []
    data_iterator = tqdm(dataloader)
    for x, y, M, meta in data_iterator:
        x = x.to(device)
        y = y.to(device)
        M = M.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y, M)

        # compute nme
        preds = decode_preds(y_pred)
        nme_batch = compute_nme(preds, meta)
        nme_batch_sum += nme_batch.sum()
        nme_count += preds.shape[0]

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        data_iterator.set_postfix(loss=np.mean(losses))

        step_dict['train_global_steps'] += 1
        
    if lr_scheduler is not None:
        lr_scheduler.step()

    nme = nme_batch_sum / nme_count
    print(f'Train epoch {epoch}, avg loss : {np.mean(losses)}, nme : {nme}.')
    return np.mean(losses), nme

def finetune_val_one_epoch(model, epoch, dataloader, criterion, step_dict, device):
    model.eval()
    nme_count = 0
    nme_batch_sum = 0
    losses = []
    data_iterator = tqdm(dataloader)
    with torch.no_grad():
        for x, y, M, meta in data_iterator:
            x = x.to(device)
            y = y.to(device)
            M = M.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y, M)

            # compute nme
            preds = decode_preds(y_pred)
            nme_batch = compute_nme(preds, meta)
            nme_batch_sum += nme_batch.sum()
            nme_count += preds.shape[0]

            losses.append(loss.item())
            data_iterator.set_postfix(loss=np.mean(losses))

            step_dict['val_global_steps'] += 1

    nme = nme_batch_sum / nme_count
    print(f'Val epoch {epoch}, avg loss : {np.mean(losses)}, nme : {nme}.')
    return np.mean(losses), nme

def save_checkpoint(model, optimizer, epoch, loss, nme, step_dict, filename, lr_scheduler=None):
    save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'nme': nme,
            'step_dict': step_dict
    }

    # if use cosine annealing lr scheduler, save its state_dict
    if lr_scheduler is not None:
        save_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()

    torch.save(save_dict, filename)