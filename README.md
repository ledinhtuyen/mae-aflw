# MAE on AFLW dataset
## How to use
### 1. Use dataset croped face 64x64 on Kaggle
- Use [this](https://www.kaggle.com/datasets/tuyenldvn/aflw-face-crop)
### 2. Config WANDB_API_KEY on configs/pretrain.yaml or configs/finetune.yaml and config other hyperparameters
- Get API key on [WANDB](https://wandb.ai/)
- Config other hyperparameters in configs/pretrain.yaml or configs/finetune.yaml
### 3. Pretrain
- Run command: `python train.py --cfg configs/pretrain.yaml`
### 4. Finetune
- Run command: `python finetune.py --cfg configs/finetune.yaml --pretrained <path_to_pretrain_model>`
- Resume finetune from checkpoint file: `python finetune.py --cfg configs/finetune.yaml --resume <path_to_checkpoint>`
## Demo
![image](https://github.com/tuyenldhust/mae-aflw/assets/19906050/694769ce-f7aa-4b55-88d2-14b06771dd36)
