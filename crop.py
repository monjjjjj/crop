import torch
import os
import timm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from torchvision import transforms, datasets
import numpy as np
import time
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
import pathlib

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

train_path = "../Han/data_spilt_91/train"
val_path = "../Han/data_spilt_91/val"
EPOCHS = 20
BATCH_SIZE = 16
sizeW, sizeH = 384, 384
data_size = 320

train_transform = transforms.Compose([transforms.Resize((sizeW, sizeH)),
                                      transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0)),
                                      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.RandomRotation(degrees=(-30, 30)),
                                      transforms.CenterCrop((data_size, data_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize((sizeW, sizeH)),
                                     transforms.CenterCrop((data_size, data_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


train_data = datasets.ImageFolder(train_path, transform=train_transform)
val_data = datasets.ImageFolder(val_path, transform=test_transform)
print(train_data.classes, "\n")
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
NUM_CLASSES = len(train_data.classes)
mixup_ft = Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
    prob=0.1, switch_prob=0.5, mode='batch',
    label_smoothing=0.1, num_classes= NUM_CLASSES
)

def build_model():
    model = timm.create_model("convnext_base", pretrained=True, num_classes=NUM_CLASSES)
    for name, param in model.named_parameters():
        param.requires_grad = True
    print(model.get_classifier())
    model = model.to(device)
    print(model)
    return model


def train(model, x_train, epochs, batch_size=8):
    def acc_cal(pred, t):
        matrix = np.argmax(pred, axis=1)
        compare = matrix - t
        accuracy = (t.shape[0] - np.count_nonzero(compare)) / t.shape[0]
        return accuracy
    train_state = []
    torch.cuda.empty_cache()
    early_stopping_count = -1
    early_stopping_loss = 0
    
    train_loss, train_acc = 0, 0
    count, count2 = 0, 0
    dicts = {}
    for x_batch, y_batch in x_train:
        optimizer.zero_grad()
        batch_data = x_batch.to(device)
        target = y_batch.to(device)
        samples, targets = mixup_ft(batch_data, target)
        outputs = model(samples)
        loss = criterion_train(outputs, targets)
        train_loss += loss.item()
        train_pred = outputs.detach().cpu().numpy()
        train_trg = target.cpu().numpy()
        acc = acc_cal(train_pred, train_trg)
        train_acc += acc
        count += 1
        loss.backward()
        optimizer.step()
        if count % 10 == 0:
            #t = round(batch_t - epoch_t)
            loss_per_10_batch = round(train_loss / count, 5)
            acc_per_10_batch = round(train_acc / count, 5)
            print("[{}] {}/{} train_loss:{:>9.5f}, train_acc:{:>9.5f}".format(epoch, count,len(x_train), loss_per_10_batch, acc_per_10_batch), end="\r")
    avg_loss = train_loss / count
    avg_acc = train_acc / count
    print("[{}] {}/{} train_loss:{:>9.5f}, train_acc:{:>9.5f}".format(epoch, len(x_train),len(x_train), round(avg_loss, 5), round(avg_acc, 5)))
    dicts["train_loss"] = avg_loss
    dicts["train_acc"] = avg_acc
    if early_stopping_loss < avg_loss:
        early_stopping_count += 1
        torch.save(model.state_dict(), f"./path/timm_model_{epoch}({early_stopping_count}).pth")
        if early_stopping_count >= 3:
            return model
    else:
        early_stopping_count = 0
    early_stopping_loss = avg_loss
    torch.save(model.state_dict(), f"./path/timm_model_{epoch}.pth")
    return train_state


def validation(model, x_val, epochs, batch_size=8):
    def acc_cal(pred, t):
        matrix = np.argmax(pred, axis=1)
        compare = matrix - t
        accuracy = (t.shape[0] - np.count_nonzero(compare)) / t.shape[0]
        return accuracy
    model.eval()
    val_state = []
    torch.cuda.empty_cache()
    all_time = time.time()
    val_loss, val_acc = 0, 0
    count, count2 = 0, 0
    dicts = {}
    with torch.no_grad():
        for x_batch, y_batch in x_val:
            batch_data = x_batch.to(device)
            target = y_batch.to(device)
            outputs = model(batch_data)
            loss = criterion_test(outputs, target)
            val_loss += loss.item()
            val_pred = outputs.detach().cpu().numpy()
            val_trg = target.cpu().numpy()
            acc = acc_cal(val_pred, val_trg)
            val_acc += acc
            count += 1
            if count % 10 == 0:
                #t = round(batch_t - epoch_t)
                loss_per_10_batch = round(val_loss / count, 5)
                acc_per_10_batch = round(val_acc / count, 5)
                print("[{}] {}/{} val_loss:{:>9.5f}, val_acc:{:>9.5f}".format(epoch, count, len(x_val), loss_per_10_batch, acc_per_10_batch), end="\r")
    avg_loss = val_loss / count
    avg_acc = val_acc / count
    t = time.time() - all_time
    print("[{}] {}/{} loss:{:>9.5f}, acc:{:>9.5f}".format(epoch, len(x_val), len(x_val), round(avg_loss, 5), round(avg_acc, 5)))
    return val_state

model = build_model()
criterion_train = SoftTargetCrossEntropy()
criterion_test = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)


for epoch in range(EPOCHS):
    train_state = train(model, train_loader, epochs=epoch, batch_size=BATCH_SIZE)
    val_state = validation(model, test_loader, epochs=epoch, batch_size=BATCH_SIZE)


torch.save(model.state_dict(), './convnext_base.pth')
