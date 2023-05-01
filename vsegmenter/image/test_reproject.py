import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

input_raster_path = '/media/gus/data/viticola/OUTPUT.tif'
output_raster_path = '/media/gus/data/viticola/OUTPUT_222.tif'

# Define the source and destination CRS (you can use EPSG codes or any other supported CRS format)
src_crs = 'EPSG:4326'  # WGS 84
dst_crs = 'EPSG:3857'  # Web Mercator

# Read the input raster data
with rasterio.open(input_raster_path) as src:
    # Calculate the default transform to reproject the raster
    transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)

    # Update the metadata with the new CRS, transform, width, and height
    out_meta = src.meta.copy()
    out_meta.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    # Read the input raster data into a numpy array
    src_array = src.read()

    # Create an empty numpy array with the same shape as the source array to store the reprojected data
    dst_array = np.empty_like(src_array)

    # Reproject the raster data
    reproject(
        src_array, dst_array,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )

    # Write the reprojected data to the output raster file
    with rasterio.open(output_raster_path, 'w', **out_meta) as dst:
        dst.write(dst_array)
