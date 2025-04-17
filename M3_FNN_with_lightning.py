import os
import torch
import cv2
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

resize_height, resize_width = 256, 512

input_dim = 3 * resize_height * resize_width
output_dim = 2 * resize_height * resize_width

class Rescale():
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        return cv2.resize(sample, dsize=self.output_size, interpolation=cv2.INTER_NEAREST)

class TusimpleData(Dataset):
    def __init__(self, dataset_file, n_labels=3, transform=None, target_transform=None, training=True, optuna=False):
        self._gt_img_list = []
        self._gt_label_binary_list = []
        self.transform = transform
        self.target_transform = target_transform
        self.n_labels = n_labels

        with open(dataset_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()
                self._gt_img_list.append(info_tmp[0])
                self._gt_label_binary_list.append(info_tmp[1])

        self._shuffle()

        purger = 1
        if purger < 1.0 and training:
            subset_size = int(len(self._gt_img_list) * purger)
            self._gt_img_list = self._gt_img_list[:subset_size]
            self._gt_label_binary_list = self._gt_label_binary_list[:subset_size]

    def _shuffle(self):
        zipped = list(zip(self._gt_img_list, self._gt_label_binary_list))
        random.shuffle(zipped)
        self._gt_img_list, self._gt_label_binary_list = zip(*zipped)

    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        img = Image.open(self._gt_img_list[idx])
        label_img = cv2.imread(self._gt_label_binary_list[idx], cv2.IMREAD_COLOR)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label_img = self.target_transform(label_img)

        label_binary = np.zeros((label_img.shape[0], label_img.shape[1]), dtype=np.uint8)
        mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
        label_binary[mask] = 1
        label_binary = torch.from_numpy(label_binary).long()
        return img, label_binary

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
target_transforms = transforms.Compose([
    Rescale((resize_width, resize_height))
])

class LaneLinesFNN(nn.Module):
    def __init__(self, hidden1=1024, hidden2=256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        logits = logits.view(-1, 2, resize_height, resize_width)
        pred = torch.argmax(logits, dim=1, keepdim=True)
        return {"binary_seg_logits": logits, "binary_seg_pred": pred}


class LaneSegLightning(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.model = LaneLinesFNN()
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, out, target):
        logits = out["binary_seg_logits"]
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
    val_file = 'archive/TUSimple/train_set/training/val.txt'

    train_ds = TusimpleData(train_file, transform=data_transforms['train'], target_transform=target_transforms, training=True)
    val_ds = TusimpleData(val_file, transform=data_transforms['val'], target_transform=target_transforms, training=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    print(f"[INFO] Loaded {len(train_ds)} samples for training.")

    model = LaneSegLightning()
    logger = CSVLogger("logs", name="laneseg")
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best_model")

    trainer = pl.Trainer(max_epochs=10, logger=logger, callbacks=[checkpoint], accelerator="auto", devices=1)
    trainer.fit(model, train_loader, val_loader)

    test(checkpoint.best_model_path)

def test(model_ckpt_path):
    if not os.path.exists('test_output'):
        os.makedirs('test_output')

    img_path = '0001.png'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LaneSegLightning.load_from_checkpoint(model_ckpt_path)
    model.eval()
    model.freeze()
    model = model.to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    inp = Image.open(img_path)
    input_tensor = transform(inp).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)

    binary_logits = output['binary_seg_logits']
    binary_pred = output['binary_seg_pred']
    binary_logits_np = binary_logits.detach().cpu().numpy()
    binary_pred_np = binary_pred.detach().cpu().numpy()

    input_img = np.array(inp.resize((resize_width, resize_height)))
    overlay = input_img.copy()
    overlay[binary_pred_np[0, 0, :, :] > 0] = [0, 0, 255]

    cv2.imwrite("test_output/input.jpg", input_img)
    cv2.imwrite("test_output/binary_prediction.jpg", binary_pred_np[0, 0] * 255)
    cv2.imwrite("test_output/input_with_prediction_overlay.jpg", overlay)

    for i in range(binary_logits_np.shape[1]):
        logits = binary_logits_np[0, i, :, :]
        logits_norm = cv2.normalize(logits, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(f"test_output/binary_logits_channel_{i}.jpg", logits_norm)

    print("✅ Prediction visualization complete — see test_output/")

if __name__ == "__main__":
    run_training()
