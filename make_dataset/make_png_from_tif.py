import rasterio
import numpy as np
from PIL import Image

# Open the TIFF file
tif_path = "../Orthophotos_patches_tiff/HR/Farmland/2000.tif"
output_png = "../output.png"

with rasterio.open(tif_path) as src:
    # Read a specific band (e.g., band 1)
    band1 = src.read(3)

    # Normalize to 0-255 (if needed)
    band1 = (band1 - band1.min()) / (band1.max() - band1.min()) * 255
    band1 = band1.astype(np.uint8)

    # Save as PNG
    Image.fromarray(band1).save(output_png)

print(f"Saved single-band image as {output_png}")

