import torch
import matplotlib.pyplot as plt
from utilities3 import LpLoss 
from navier_stokes_uno2d_pretrain import UNO, UNO_P
import os 
from data_load_img import *

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

def test_model(model:UNO, test_loader, device="cuda", save_dir=None):
    model.to(device)
    model.eval()
    loss_fn = LpLoss(size_average=False)

    total_l2 = 0.0
    count = 0

    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x = x.to(device)  # [B, 1, 64, 64]
            y = y.to(device)  # [B, 1, 256, 256]

            B, _, H_hr, W_hr = y.shape
            x = x.permute(0, 2, 3, 1)  # [B, 64, 64, 1]
            coord = make_coord((H_hr, W_hr)).unsqueeze(0).repeat(B, 1, 1, 1).to(device)

            model.gen_feat(x)
            pred = model.query_features(coord)  # [B, H, W, 1]
            pred = pred.permute(0, 3, 1, 2)     # [B, 1, H, W]

            # Loss calculation
            loss = loss_fn(pred.reshape(B, -1), y.reshape(B, -1))
            total_l2 += loss.item()
            count += B

            # Optional: Save images
            if save_dir and idx < 5:  # Save first 5
                for i in range(B):
                    # print("shape pred", pred[i].shape)
                    plt.imsave(f"{save_dir}/pred_{idx}_{i}.png", pred[i].permute(1,2,0).cpu().numpy(),)
                    plt.imsave(f"{save_dir}/gt_{idx}_{i}.png", y[i].permute(1,2,0).cpu().numpy())

    print(f"Test L2 Loss: {total_l2 / count:.6f}")

def main():
    train_lr_path = os.path.expanduser("~/EDiffSR/Orthophotos_patches_png_scale16_split_ratio/LR/test/Farmland")
    train_hr_path = os.path.expanduser("~/EDiffSR/Orthophotos_patches_png_scale16_split_ratio/HR/test/Farmland")

    dataset = EdgeDataset(lr_dir=train_lr_path, hr_dir=train_hr_path)
    test_size = len(dataset) * 0.5
    test_dataset = torch.utils.data.random_split(dataset, [int(test_size), len(dataset) - int(test_size)])[0]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    width = 32
    inwidth = 7
    model = UNO(inwidth, width)
    model.load_state_dict(torch.load("./weights//UNO-10e3.pt"))
    
    save_dir="./test_results_RGB"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    test_model(
        model,
        test_loader,
        device="cuda",
        save_dir=save_dir  # Directory to save test results
    )
    
if __name__ == "__main__":
    main()
