import torch
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CaptchaDataset
from model import CRNN_Model
from engine import Train_Fn, Eval_Fn
from config import (
    Bach_Size, Epochs, Learning_Rate, Num_Workers,
    TRAIN_FOLDER, VAL_FOLDER,
    Early_Stop_Patience, Weight_Decay, Seed
)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), labels


def load_model_from_checkpoint(path, device):
    model = CRNN_Model().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    set_seed(Seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 数据集：只用 train 和 val
    train_ds = CaptchaDataset(TRAIN_FOLDER)
    val_ds = CaptchaDataset(VAL_FOLDER)

    train_loader = DataLoader(train_ds, batch_size=Bach_Size, shuffle=True,
                              num_workers=Num_Workers, pin_memory=False,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=Bach_Size, shuffle=False,
                            num_workers=Num_Workers, pin_memory=True,
                            collate_fn=custom_collate_fn)

    model = CRNN_Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    early_stop_count = 0

    latest_ckpt = "latest_crnn.pth"
    best_ckpt = "best_crnn.pth"

    for epoch in range(Epochs):
        print(f"\n=== Epoch {epoch + 1}/{Epochs} ===")
        train_iter = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}", leave=False)
        train_loss = Train_Fn(model, train_iter, optimizer, device)

        # 保存本轮模型（立即落盘）
        torch.save(model.state_dict(), latest_ckpt)

        # 从磁盘加载用于评估（确保用的是写入的权重）
        eval_model = load_model_from_checkpoint(latest_ckpt, device)

        # validation 评估
        val_acc = Eval_Fn(eval_model, val_loader, device)

        print(f"Epoch [{epoch + 1}] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        # scheduler 依据 validation accuracy 调整
        scheduler.step(val_acc)

        # 打印 temperature（如果有）
        if hasattr(model, "logit_temperature"):
            temp = model.logit_temperature.detach().cpu().item()
            print(f"Temperature: {temp:.3f}")

        # 早停 & best model 保存（基于 validation）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_count = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"→ New best VAL model saved (Acc={best_val_acc:.4f})")
        else:
            early_stop_count += 1
            print(f"Early stop count: {early_stop_count}/{Early_Stop_Patience}")
            if early_stop_count >= Early_Stop_Patience:
                print("Early stopping triggered.But continue training.")

    # 训练结束后提示（不再做额外 final eval）
    print("\nTraining finished.")
    if os.path.exists(best_ckpt):
        print(f"Best validation accuracy: {best_val_acc:.4f} (model saved at '{best_ckpt}')")
    else:
        print("No best model was saved.")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
    import warnings
    warnings.filterwarnings("ignore", message="Error fetching version info*")
    torch.set_num_threads(4)
    freeze_support()
    main()
