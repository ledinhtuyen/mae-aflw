import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
from .MAE import patchify

class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)
    
class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)
    
class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, out_planes),
            SingleConv2DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
    
class UNETR_Encoder(nn.Module):
    def __init__(self, model_pretrained=None, embedding_dim=768, n_heads = 12, n_layers = 12, feedforward_dim = 768*4, patch_size=4, num_patches=16):
        super().__init__()
        if model_pretrained is not None:
          self.layers = model_pretrained.encoder.transformer.layers
          self.encoder_input_projection = model_pretrained.encoder_input_projection
          self.encoder_position_encoding = model_pretrained.encoder_position_encoding
          self.cls_token = model_pretrained.cls_token
        else:
          # If not use pretrained model, use scratch model
          self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=n_heads,
                dim_feedforward=feedforward_dim,
                activation=F.gelu,
                batch_first=True,
                dropout=0.1,
                norm_first=True
            ),
            num_layers=n_layers,
            norm=nn.LayerNorm(embedding_dim, eps=1e-6)
          ).layers
          self.encoder_input_projection = nn.Linear(patch_size * patch_size * 3, embedding_dim)
          self.encoder_position_encoding = nn.Parameter(torch.randn(1, num_patches * num_patches, embedding_dim))
          self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.ext_layers = [3, 6, 9, 12] # Extract features from these layers
    
    def forward(self, images):
        patches = patchify(images)
        projection = self.encoder_input_projection(patches)
        b, n, l = projection.shape
        projection += self.encoder_position_encoding[:, :(n + 1)]
        cls_tokens = einops.repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, projection), dim=1)

        hidden_states = x
        extract_layers = []
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if i + 1 in self.ext_layers:
                extract_layers.append(hidden_states)
        return extract_layers
    
# Tham khảo ở đây (bỏ z3, z9 vì chỉ upsample 2 lần, patch_size = 4)
# https://github.com/tamasino52/UNETR/blob/main/unetr.py

class UNETR(nn.Module):
    def __init__(self, encoder, input_dim=3, output_dim=19, embed_dim=768, patch_dim=(16, 16)):
        # img_size = 64
        # patch_size = 4
        # patch_dim = img_size // patch_size
        super().__init__()
        self.patch_dim = patch_dim
        self.encoder = encoder

        self.decoder0 = \
            nn.Sequential(
                Conv2DBlock(input_dim, 32, 3),
                Conv2DBlock(32, 64, 3)
            )

        self.decoder6 = Deconv2DBlock(embed_dim, 128)

        self.decoder12_upsampler = \
            SingleDeconv2DBlock(embed_dim, 128)


        self.decoder6_upsampler = \
            nn.Sequential(
                Conv2DBlock(256, 128),
                Conv2DBlock(128, 128),
                SingleDeconv2DBlock(128, 64)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv2DBlock(128, 64),
                Conv2DBlock(64, 64),
                SingleConv2DBlock(64, output_dim, 1)
            )

    
    def forward(self, images):
        z = self.encoder(images)
        z0, z3, z6, z9, z12 = images, *z
        z6, z12 = z6[:, 1:, :], z12[:, 1:, :] # remove cls token and not use z3, z9 because only upsample 2 times (patch_size = 4)

        arranger = lambda z_emb: rearrange(z_emb, 'b (x y) d -> b d x y',
                                           x=self.patch_dim[0], y=self.patch_dim[1])
        
        z6, z12 = map(arranger, (z6, z12))
        
        z12 = self.decoder12_upsampler(z12)
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z12], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z6], dim=1))
        return output