import torch
import pandas as pd
import pathlib
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets
from torchvision import transforms
import timm
import time

def build_model():
    model = timm.create_model("convnext_base", pretrained = True, num_classes = 33)
    for name, param in model.named_parameters():
        param.requires_grad = True
    print(model.get_classifier())
    model = model.to(device)
    print(model)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 128
# tuple: idx0=image, idx1=label
TRAIN_DATA_PATH = "../YuAn/crops_dataset"
TEST_DATA_PATH = "../YuAn/public_test"
PATH = "./path/timm_model_0(0).pth"
sizeW, sizeH = 320, 320
train_data = datasets.ImageFolder(TRAIN_DATA_PATH)
category_dict = {j:i for i, j in train_data.class_to_idx.items()}
test_data = datasets.ImageFolder(TEST_DATA_PATH, 
                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Resize((sizeW, sizeH))]))
filename = [imgPath.replace("../YuAn/public_test/public_test/", "") for imgPath, fakeLabel in test_data.imgs]
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
category = len(train_data.classes)
model = build_model()
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()
lst = []
with torch.no_grad():
    c = 0
    for data in test_loader:
        images, _ = data
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        lst.extend(predicted.tolist())
        c += 1
        print(f"{c}/{len(test_loader)}", end="\r")

lst = [category_dict.get(i) for i in lst]
df = pd.DataFrame(data={"filename": filename, "label": lst})
df.to_csv("./submission.csv", index=False)