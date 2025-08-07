import pprint
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
from tqdm import tqdm

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
    epochs=105,
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
    # myloss = LpLoss(p=1, size_average=False)
    loss_fn = torch.nn.L1Loss()
    
    weight_dir = "./weights"
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    weight_path = os.path.join(weight_dir, weight_path)

    model.to(device)

    for ep in tqdm(range(epochs), desc="Epochs"):
        model.train()
        t1 = default_timer()
        train_l2 = 0.0

        for x, y in train_loader:
            x = x.to(device)  # Upsampling LR image: [B, 3, 256, 256]
            y = y.to(device)  # HR image: [B, 3, 256, 256]
                    
            B, _, H_hr, W_hr = y.shape

            x = x.permute(0, 2, 3, 1)

            pred = model(x)
            # print(f"pred shape: {pred.shape}, y shape: {y.shape}")
            
            # pred = pred.permute(0, 3, 1, 2)

            save_dir = f"./saved_images/SR_images/epoch_{ep}"
            os.makedirs(save_dir, exist_ok=True)
            existing_images = [f for f in os.listdir(save_dir) if f.startswith("pred_sample_") and f.endswith(".png")]
            
            if len(existing_images) < 3:
                num_to_save = min(3 - len(existing_images), pred.size(0))
                for i in range(num_to_save):
                    # Save predicted image
                    pred_img = pred[i].detach().cpu()
                    pred_img = torch.clamp(pred_img, 0.0, 1.0)
                    pred_img = (pred_img * 255).numpy().astype(np.uint8)
                    pred_img_pil = Image.fromarray(pred_img)
                    pred_img_pil.save(f"{save_dir}/pred_sample_{len(existing_images) + i}.png")

                    # Save ground truth image
                    y_img = y[i].detach().cpu().permute(1, 2, 0)
                    y_img = (y_img * 255).clamp(0, 255).to(torch.uint8)
                    y_img_pil = Image.fromarray(y_img.numpy())
                    y_img_pil.save(f"{save_dir}/y_sample_{len(existing_images) + i}.png")
            
            # print(f"Epoch {ep+1}, Batch {x.shape[0]}: Predicted and ground truth images saved.")
            # print(f"pred shape: {pred.shape}, y shape: {y.shape}")
            # print(f"pred dtype: {pred.dtype}, y dtype: {y.dtype}")
            # print("pred min, max:", pred.min().item(), pred.max().item())
            # print("y min, max:", y.min().item(), y.max().item())
            # loss = myloss(pred, y)
            pred = pred.permute(0, 3, 1, 2)  # [B, C, H, W]
            loss = loss_fn(pred, y)

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

                pred = model(x)
                pred = pred.permute(0, 3, 1, 2)

                # loss = myloss(pred, y)
                loss = loss_fn(pred, y)
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

            pred = model(x)
            print(f"pred shape: {pred.shape}, y shape: {y.shape}")
            pred = pred.permute(0, 3, 1, 2)

            # loss = myloss(pred, y)
            loss = loss_fn(pred, y)
            test_l2 += loss.item()
            del x, y, pred

    print(f"Test Loss: {test_l2 / ntest:.6f}")
