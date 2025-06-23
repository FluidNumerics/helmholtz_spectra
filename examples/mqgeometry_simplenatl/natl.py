import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point, Polygon
from shapely.vectorized import contains

# Define grid over North Atlantic
lon = np.linspace(-100, -10, 512)  # west to east
lat = np.linspace(0, 60, 256)     # south to north
lon2d, lat2d = np.meshgrid(lon, lat)

dlon = lon[1] - lon[0]  # grid spacing in longitude
dlat = lat[1] - lat[0]  # grid spacing in latitude
dx = dlon * 111320  # approximate conversion from degrees to meters
dy = dlat * 110574  # approximate conversion from degrees to meters
print(f"Grid spacing: dx = {dx:.2f} m, dy = {dy:.2f} m")

# Load land feature from Natural Earth
land_geom = list(cfeature.NaturalEarthFeature(
    category='physical', name='land', scale='110m').geometries())

# Flatten grid for vectorized masking
points = np.column_stack((lon2d.ravel(), lat2d.ravel()))

# Initialize mask with 1 (ocean)
mask = np.ones(points.shape[0], dtype=np.uint8)

# Rasterize: mark land as 0
for geom in land_geom:
    mask[contains(geom, points[:, 0], points[:, 1])] = 0


# Define a bounding box (loose definition of the Bahamas)
# Adjust as needed for better precision
bahamas_poly = Polygon([
    (-79.5, 20.5),  # SW
    (-72.5, 20.5),  # SE
    (-72.5, 27.5),  # NE
    (-79.5, 27.5),  # NW
    (-79.5, 20.5)   # Close loop
])

# Vectorized check: set mask = 1 where point is inside the polygon
mask[contains(bahamas_poly, points[:, 0], points[:, 1])] = 1

bahamas_poly = Polygon([
    (-85.0, 15.0),  # SW
    (-60.0, 15.0),  # SE
    (-60.0, 23.2),  # NE
    (-85.0, 23.2),  # NW
    (-85.0, 15.0)   # Close loop
])

# Vectorized check: set mask = 1 where point is inside the polygon
mask[contains(bahamas_poly, points[:, 0], points[:, 1])] = 1

# Mask out lower left corner in the pacific
pacific_poly = Polygon([
    (-101.0, -1.0),  # SW
    (-65.0, -1.0),  # SE
    (-101.0, 20.0),  # NW
    (-101.0, -1.0)   # Close loop
])

# Vectorized check: set mask = 1 where point is inside the polygon
mask[contains(pacific_poly, points[:, 0], points[:, 1])] = 0


# Close up some bits in canada
canada_poly = Polygon([
    (-100.0, 50.0),  # SW
    (-65.0, 50.0),  # SE
    (-65.0, 62.0),  # NE
    (-100.0, 62.0),  # NW
    (-100.0, 50.0)   # Close loop
])
mask[contains(canada_poly, points[:, 0], points[:, 1])] = 0


# Smooth out coastlines with median filter

# Reshape to 2D
mask2d = mask.reshape(lat2d.shape)


# Example mask2d: binary 2D array (0 = land, 1 = ocean)

# Step 1: Smooth with a uniform kernel (e.g., 3x3 or 5x5 mean filter)
smoothed = scipy.ndimage.median_filter(mask2d.astype(float), size=9)

# Step 2: Threshold — convert to binary mask
# You can use a low threshold (e.g., > 0.1) to fill coastlines
mask2d = (smoothed > 0.01).astype(np.uint8)


# Plot for verification
plt.figure(figsize=(10, 6))
plt.pcolormesh(lon2d, lat2d, mask2d, cmap="gray_r")
plt.title("North Atlantic Land/Ocean Mask (0 = Land, 1 = Ocean)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(label="Mask Value")
plt.savefig("natl_mask.png", dpi=300)
