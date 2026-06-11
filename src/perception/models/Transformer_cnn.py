import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):

    def __init__(self, in_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)   


class SimpleSelfAttention(nn.Module):

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
   
        B, C, H, W = x.shape

        tokens = x.view(B, C, H * W).permute(0, 2, 1)  

        attn_out, _ = self.attn(tokens, tokens, tokens)

        tokens = self.norm(tokens + attn_out)

        x_out = tokens.permute(0, 2, 1).view(B, C, H, W)
        return x_out

class GraspTransformerCNN(nn.Module):
    def __init__(self, in_channels=4, out_channels=5):
        super().__init__()

        self.encoder = ConvEncoder(in_channels=in_channels)

        self.shrink = nn.AvgPool2d(kernel_size=4)   

        self.attention = SimpleSelfAttention(dim=128, num_heads=2)  

        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, stride=2),  
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
        )


        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):

        f_local = self.encoder(x)   

        small = self.shrink(f_local)       
        f_global = self.attention(small)    
        f_global = F.interpolate(f_global, size=f_local.shape[2:], mode='bilinear')

        fused = torch.cat([f_local, f_global], dim=1)
        fused = self.fusion(fused)

        dec = self.decoder(fused)

        return self.head(dec)


