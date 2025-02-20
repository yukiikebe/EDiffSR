import rasterio

def identify_raster_bands(tiff_path):
    """
    Identifies the bands in a raster TIFF file and prints their corresponding names.
    """
    with rasterio.open(tiff_path) as dataset:
        num_bands = dataset.count  # Number of bands
        band_names = dataset.descriptions  # Band descriptions (if available)
        color_interpretations = dataset.colorinterp  # Band color interpretation
        
        print(f"Number of bands: {num_bands}\n")
        
        band_mapping = {}

        for i in range(num_bands):
            band_index = i + 1  # Raster bands start from 1
            
            # Get band interpretation (if available)
            color_name = str(color_interpretations[i]).replace("ColorInterp.", "")

            # Use dataset.descriptions if available, otherwise use colorinterp
            if band_names and band_names[i]:
                band_mapping[f"Band {band_index}"] = band_names[i]
            else:
                band_mapping[f"Band {band_index}"] = color_name

        # Print the mapping
        for band, name in band_mapping.items():
            print(f"{band}: {name}")

# Example usage
tiff_file = "results/sisr/Test-AID-farm-multiband-fliprot-iter500000-wholeimage20-re/AID/0007.tif"  # Replace with your TIFF file path
identify_raster_bands(tiff_file)
