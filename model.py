import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from config import Image_Height, NUM_CLASSES


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=False),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, (nn.GRU, nn.LSTM)):
        for name, param in module.named_parameters():
            if 'weight' in name:
                init.kaiming_normal_(param, nonlinearity='tanh')
            elif 'bias' in name:
                init.zeros_(param)


def spec_augment(features, time_mask_width=10, freq_mask_width=16):
    features = features.clone()
    B, T, D = features.size()
    # time mask
    t = torch.randint(0, time_mask_width + 1, (1,)).item()
    if t > 0 and T - t > 0:
        t0 = torch.randint(0, T - t + 1, (1,)).item()
        features[:, t0:t0 + t, :] = 0
    # freq mask
    f = torch.randint(0, freq_mask_width + 1, (1,)).item()
    if f > 0 and D - f > 0:
        f0 = torch.randint(0, D - f + 1, (1,)).item()
        features[:, :, f0:f0 + f] = 0
    return features


class CRNN_Model(nn.Module):
    def __init__(self):
        super(CRNN_Model, self).__init__()

        self.Conv_Block = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SEBlock(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SEBlock(512),
            nn.MaxPool2d(2, 2),
        )

        cnn_output_h = Image_Height // 8
        self.Linear_Input_Size = 512 * cnn_output_h

        self.Linear = nn.Sequential(
            nn.Linear(self.Linear_Input_Size, 128),
            nn.Dropout(0.2),
            nn.ReLU()
        )

        self.RNN = nn.GRU(
            128,
            256,
            num_layers=3,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(512)
        self.Classifier = nn.Linear(512, NUM_CLASSES)
        self.aux_classifier = nn.Linear(512, NUM_CLASSES)

        self.residual_proj = nn.Linear(128, 512)  # 预先定义 projection，避免在 forward 中新建

        self.logit_temperature = nn.Parameter(torch.tensor(1.0))

        self.apply(init_weights)

    def forward(self, x):
        x = self.Conv_Block(x)  # [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, W', C, H]
        x = x.view(x.size(0), x.size(1), -1)  # [B, T, Linear_Input_Size]
        x = self.Linear(x)  # [B, T, 128]

        if self.training:
            x = spec_augment(x)  # 用非原地版本

        rnn_out, _ = self.RNN(x)  # [B, T, 512]
        residual = self.residual_proj(x)  # 投影输入到 512
        rnn_out = rnn_out + residual  # residual connection

        x_norm = self.layer_norm(rnn_out)  # [B, T, 512]

        main_logits = self.Classifier(x_norm)  # [B, T, C]
        aux_logits = self.aux_classifier(x_norm)  # [B, T, C]

        temperature = torch.clamp(self.logit_temperature, min=0.5, max=5.0)
        main_logits = main_logits / temperature
        aux_logits = aux_logits / temperature

        main_log_probs = F.log_softmax(main_logits, dim=2)  # [B, T, C]
        aux_log_probs = F.log_softmax(aux_logits, dim=2)

        main_log_probs = main_log_probs.permute(1, 0, 2)  # [T, B, C]
        aux_log_probs = aux_log_probs.permute(1, 0, 2)
        return main_log_probs, aux_log_probs
