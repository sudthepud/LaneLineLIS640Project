import os
import random
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torchvision import transforms

resize_height, resize_width = 256, 512
seq_len = 50

class Rescale():
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        return cv2.resize(sample, dsize=self.output_size, interpolation=cv2.INTER_NEAREST)

class TusimpleSeqData(Dataset):
    def __init__(self, dataset_file, transform=None, target_transform=None, training=True, seq_len=50):
        self.transform = transform
        self.target_transform = target_transform
        self.seq_len = seq_len

        img_list, lbl_list = [], []
        with open(dataset_file, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 2:
                    continue
                img_list.append(tokens[0])
                lbl_list.append(tokens[1])

        if training:
            pairs = list(zip(img_list, lbl_list))
            random.shuffle(pairs)
            img_list, lbl_list = zip(*pairs)

        purger = 0.01
        if purger < 1.0 and training:
            keep = int(len(img_list) * purger)
            img_list, lbl_list = img_list[:keep], lbl_list[:keep]

        self._imgs = list(img_list)
        self._lbls = list(lbl_list)

        total_frames = len(self._imgs)
        total_seqs = total_frames // self.seq_len
        self.sequences = [
            list(range(i * self.seq_len, (i + 1) * self.seq_len))
            for i in range(total_seqs)
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        indices = self.sequences[idx]
        imgs, labels = [], []

        for i in indices:
            img = Image.open(self._imgs[i]).convert("RGB")
            if self.transform:
                img = self.transform(img)

            lbl_img = cv2.imread(self._lbls[i], cv2.IMREAD_COLOR)
            if self.target_transform:
                lbl_img = self.target_transform(lbl_img) 
            
            mask = (lbl_img[:, :, :] != [0, 0, 0]).all(axis=2)
            binary = np.zeros((lbl_img.shape[0], lbl_img.shape[1]), dtype=np.uint8)
            binary[mask] = 1
            binary = torch.from_numpy(binary).long()

            imgs.append(img)
            labels.append(binary)

        imgs_seq = torch.stack(imgs, dim=0)
        labels_seq = torch.stack(labels, dim=0)
        return imgs_seq, labels_seq

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}
target_transforms = transforms.Compose([
    Rescale((resize_width, resize_height))
])


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, padding):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.Gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, state):
        h_cur, c_cur = state
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.Gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size, padding)
        self.hidden_channels = hidden_channels

    def forward(self, seq):
        B, T, C, H, W = seq.size()
        device = seq.device
        h = torch.zeros(B, self.hidden_channels, H, W, device=device)
        c = torch.zeros(B, self.hidden_channels, H, W, device=device)

        outputs = []
        for t in range(T):
            h, c = self.cell(seq[:, t], (h, c))
            outputs.append(h)
        return torch.stack(outputs, dim=1)  
    
class LaneLinesRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        self.conv_lstm = ConvLSTM(input_channels=32, hidden_channels=32)

        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(16,  2, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.relu = nn.ReLU()

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.size()
        feats = []
        # encode each frame
        for t in range(T):
            x = self.relu(self.conv1(x_seq[:, t]))
            x = self.relu(self.conv2(x))
            feats.append(x)
        feats = torch.stack(feats, dim=1)

        h_seq = self.conv_lstm(feats)

        logits_seq, preds_seq = [], []
        for t in range(T):
            h = h_seq[:, t]
            d = self.relu(self.deconv1(h))
            logits = self.deconv2(d)
            pred   = torch.argmax(logits, dim=1, keepdim=True)
            logits_seq.append(logits)
            preds_seq.append(pred)

        logits_seq = torch.stack(logits_seq, dim=1)
        preds_seq  = torch.stack(preds_seq,  dim=1)
        return {"binary_seg_logits": logits_seq, "binary_seg_pred": preds_seq}


class LaneSegRNNLightning(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model   = LaneLinesRNN()
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr      = lr

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, out, target):
        logits = out["binary_seg_logits"]
        B, T, C, H, W = logits.size()
        logits = logits.view(B * T, C, H, W)
        target = target.view(B * T, H, W)
        loss = self.loss_fn(logits, target) * 10
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.compute_loss(out, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.compute_loss(out, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

def run_training():
    train_file = 'archive/TUSimple/train_set/training/train.txt'
    val_file   = 'archive/TUSimple/train_set/training/val.txt'

    train_ds = TusimpleSeqData(
        train_file,
        transform=data_transforms['train'],
        target_transform=target_transforms,
        training=True,
        seq_len=seq_len
    )
    val_ds = TusimpleSeqData(
        val_file,
        transform=data_transforms['val'],
        target_transform=target_transforms,
        training=False,
        seq_len=seq_len
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=4)

    print(f"[INFO] Loaded {len(train_ds)} sequences for training.")

    model      = LaneSegRNNLightning()
    logger     = CSVLogger("logs", name="laneseg_rnn")
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best_rnn_model"
    )

    trainer = pl.Trainer(
        max_epochs=2,
        logger=logger,
        callbacks=[checkpoint],
        accelerator="auto",
        devices=1
    )
    trainer.fit(model, train_loader, val_loader)

    test(checkpoint.best_model_path)

def test(model_ckpt_path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LaneSegRNNLightning.load_from_checkpoint(model_ckpt_path)
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_path = '0001.png'
    inp = Image.open(img_path).convert("RGB")
    input_tensor = transform(inp).unsqueeze(0).to(DEVICE)
    input_seq = input_tensor.unsqueeze(1)

    with torch.no_grad():
        output = model(input_seq)

    logits    = output['binary_seg_logits'][0, 0].cpu().numpy()

    pred_mask = output['binary_seg_pred'][0, 0].squeeze(0).cpu().numpy()
    inp_np  = np.array(inp.resize((resize_width, resize_height)))
    overlay = inp_np.copy()
    overlay[pred_mask > 0] = [0, 0, 255]

    os.makedirs('test_output', exist_ok=True)
    cv2.imwrite("test_output/input.jpg", inp_np[:, :, ::-1])
    cv2.imwrite("test_output/binary_prediction.jpg", pred_mask * 255)
    cv2.imwrite("test_output/input_with_prediction_overlay.jpg", overlay[:, :, ::-1])

    for i in range(logits.shape[0]):
        norm = cv2.normalize(logits[i], None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f"test_output/binary_logits_channel_{i}.jpg", norm.astype(np.uint8))

    print("Prediction visualization complete — see test_output/")

if __name__ == "__main__":
    run_training()
