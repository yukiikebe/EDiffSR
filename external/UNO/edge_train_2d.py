import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from functools import reduce
from functools import partial
import random
import gc
from timeit import default_timer
from utilities3 import *
from Adam import Adam
from PIL import Image
import os

def make_coord(shape, ranges=None, flatten=False):
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = (-1, 1) if ranges is None else ranges[i]
        r = torch.linspace(v0, v1, steps=n)
        coord_seqs.append(r)
    coords = torch.meshgrid(*coord_seqs, indexing='ij')
    coord = torch.stack(coords, dim=-1)  # [H, W, 2]
    if flatten:
        coord = coord.view(-1, 2)
    return coord

def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    ntrain,
    nval,
    ntest,
    weight_path,
    T_f=1,  
    step=1,
    batch_size=16,
    epochs=500,
    learning_rate=0.001,
    scheduler_step=100,
    scheduler_gamma=0.5,
    device="cuda",
    weight_decay=1e-3,
):
    optimizer = Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=False
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )
    Min_error_t = float("inf")
    myloss = LpLoss(size_average=False)
    
    weight_dir = "./weights"
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    weight_path = os.path.join(weight_dir, weight_path)

    model.to(device)

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0.0

        for x, y in train_loader:
            x = x.to(device)  # LR edge image: [B, 3, 16, 16]
            y = y.to(device)  # HR edge image: [B, 3, 256, 256]
            # Save images for checking
            if ep == 0:  
                save_dir = "./saved_images"
                os.makedirs(save_dir, exist_ok=True)
                for i in range(min(5, x.size(0))):  # Save up to 5 samples
                    x_img = (x[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [1, 16, 16]
                    y_img = (y[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [256, 256]
                    # print("x_img shape and y_shape",x_img.shape, y_img.shape)
                    x_img = Image.fromarray(x_img)  
                    y_img = Image.fromarray(y_img)  
                    
                    x_img.save(f"{save_dir}/x_sample_{i}.png")
                    y_img.save(f"{save_dir}/y_sample_{i}.png")
                    
            B, _, H_hr, W_hr = y.shape

            x = x.permute(0, 2, 3, 1)  # [B, 16, 16, 1]

            coord = make_coord((H_hr, W_hr)).unsqueeze(0).repeat(B, 1, 1, 1).to(device)  # [B, H, W, 2]

            model.gen_feat(x)
            pred = model.query_features(coord)  # [B, H, W, 1]
            pred = pred.permute(0, 3, 1, 2)  # [B, 1, H, W] for loss

            loss = myloss(pred.reshape(B, -1), y.reshape(B, -1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_l2 += loss.item()
            del x, y, pred

        gc.collect()

        # Validation
        model.eval()
        val_l2 = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                B, _, H_hr, W_hr = y.shape
                x = x.permute(0, 2, 3, 1)
                coord = make_coord((H_hr, W_hr)).unsqueeze(0).repeat(B, 1, 1, 1).to(device)

                model.gen_feat(x)
                pred = model.query_features(coord)
                pred = pred.permute(0, 3, 1, 2)

                loss = myloss(pred.reshape(B, -1), y.reshape(B, -1))
                val_l2 += loss.item()
                del x, y, pred

        scheduler.step()
        t2 = default_timer()

        print(f"Epoch {ep+1} | Time: {t2 - t1:.2f}s | Train Loss: {train_l2/ntrain:.6f} | Val Loss: {val_l2/nval:.6f}")

        if val_l2 / nval < Min_error_t:
            torch.save(model.state_dict(), weight_path)
            print(f"Model saved (val improvement: {Min_error_t - val_l2 / nval:.6f})")
            Min_error_t = val_l2 / nval

    # Test
    print("Training completed. Evaluating on test set...")
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            B, _, H_hr, W_hr = y.shape
            x = x.permute(0, 2, 3, 1)
            coord = make_coord((H_hr, W_hr)).unsqueeze(0).repeat(B, 1, 1, 1).to(device)

            model.gen_feat(x)
            pred = model.query_features(coord)
            pred = pred.permute(0, 3, 1, 2)

            loss = myloss(pred.reshape(B, -1), y.reshape(B, -1))
            test_l2 += loss.item()
            del x, y, pred

    print(f"Test Loss: {test_l2 / ntest:.6f}")
