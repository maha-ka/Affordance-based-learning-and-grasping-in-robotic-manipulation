import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from models.Unet import Unet
from dataset import JacquardDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_PATH = 'JacquardV2_Dataset_1/Jacquard_Dataset_1'
dataset = JacquardDataset(DATA_PATH)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)


bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss(reduction='none')

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device) 

        optimizer.zero_grad()
        outputs = model(inputs)

        heat_logits = outputs[:, 0:1, :, :]
        sin_pred = outputs[:, 1:2, :, :]
        cos_pred = outputs[:, 2:3, :, :]
        w_pred = outputs[:, 3:4, :, :]
        h_pred = outputs[:, 4:5, :, :]

        heat_true = targets[:, 0:1, :, :]
        sin_true = targets[:, 1:2, :, :]
        cos_true = targets[:, 2:3, :, :]
        w_true = targets[:, 3:4, :, :]
        h_true = targets[:, 4:5, :, :]

        loss_heat = bce_loss(heat_logits, heat_true)

        mask = (heat_true > 1e-3).float()  
        denom = mask.sum() + 1e-6

        loss_sin = (mse_loss(sin_pred, sin_true) * mask).sum() / denom
        loss_cos = (mse_loss(cos_pred, cos_true) * mask).sum() / denom
        loss_w = (mse_loss(w_pred, w_true) * mask).sum() / denom
        loss_h = (mse_loss(h_pred, h_true) * mask).sum() / denom

        loss = loss_heat + (loss_sin + loss_cos) * 1.0 + (loss_w + loss_h) * 0.5

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            heat_logits = outputs[:, 0:1, :, :]
            sin_pred = outputs[:, 1:2, :, :]
            cos_pred = outputs[:, 2:3, :, :]
            w_pred = outputs[:, 3:4, :, :]
            h_pred = outputs[:, 4:5, :, :]

            heat_true = targets[:, 0:1, :, :]
            sin_true = targets[:, 1:2, :, :]
            cos_true = targets[:, 2:3, :, :]
            w_true = targets[:, 3:4, :, :]
            h_true = targets[:, 4:5, :, :]

            loss_heat = bce_loss(heat_logits, heat_true)

            mask = (heat_true > 1e-3).float()
            denom = mask.sum() + 1e-6
            loss_sin = (mse_loss(sin_pred, sin_true) * mask).sum() / denom
            loss_cos = (mse_loss(cos_pred, cos_true) * mask).sum() / denom
            loss_w = (mse_loss(w_pred, w_true) * mask).sum() / denom
            loss_h = (mse_loss(h_pred, h_true) * mask).sum() / denom

            loss = loss_heat + (loss_sin + loss_cos) * 1.0 + (loss_w + loss_h) * 0.5
            val_loss += loss.item()
    return val_loss / len(loader)

model = Unet(in_channels=4, out_channels=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
EPOCHS = 50

# run training 
if __name__ == '__main__':
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"model_unet_epoch{epoch+1}.pth")