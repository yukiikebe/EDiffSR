import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def read_band(dataset, band_number):
    return dataset.read(band_number).astype('float32')

def calculate_ndvi(file_path, output_folder):
    with rasterio.open(file_path) as dataset:
        R = read_band(dataset, 1) 
        N = read_band(dataset, 4) 
        
        ndvi = (N - R) / (N + R)
        ndvi = np.clip(ndvi, -1, 1)  
        
        output_tiff = os.path.join(output_folder, "NDVI.tif")
        with rasterio.open(
            output_tiff, 'w',
            driver='GTiff',
            height=ndvi.shape[0],
            width=ndvi.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs=dataset.crs,
            transform=dataset.transform,
        ) as dst:
            dst.write(ndvi, 1)
        print(f"NDVI GeoTIFF saved at: {output_tiff}")
        
    
        plt.figure(figsize=(10, 10))
        plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='NDVI')
        # plt.title('NDVI from Drone Imagery')
        plt.axis('off')
        output_png = os.path.join(output_folder, "NDVI.png")
        plt.savefig(output_png, bbox_inches='tight', pad_inches=0.1)
        print(f"NDVI plot saved at: {output_png}")

# 実行部分
file_path = "../Orthophotos_patches_tiff_scale16_combine_20_SR/merged_farmland.tif" 
output_folder = "../Orthophotos_patches_tiff_scale16_combine_20_SR"
os.makedirs(output_folder, exist_ok=True)
calculate_ndvi(file_path, output_folder)
