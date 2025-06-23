#!/usr/bin/env python
import torch
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from helmholtz_spectra.tuml import TUML
from helmholtz_spectra.nma import NMA
import os
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from MQGeometry.qgm import QGFV
import scipy.ndimage
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point, Polygon
from shapely.vectorized import contains


torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
print(f" Device: {device}, Data type: {dtype}",flush=True)

# grid
n_ens = 1
nl = 3
nx = 512
ny = 256
dt = 4000
west = -100
east = -10
south = 0
north = 60
zwcl = 45.0 # Latitude of zero wind curl line


###########################################################################################
# North Atlantic domain parameters
# Define grid over North Atlantic
lon = np.linspace(west, east, nx)  # west to east
lat = np.linspace(south, north, ny)     # south to north
lon2d, lat2d = np.meshgrid(lon, lat)

dlon = lon[1] - lon[0]  # grid spacing in longitude
dlat = lat[1] - lat[0]  # grid spacing in latitude
dx = dlon * 111320  # approximate conversion from degrees to meters
dy = dlat * 110574  # approximate conversion from degrees to meters
Lx = dx* nx
Ly = dy * ny
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


# Time stepping parameters
# time params
t = 0
n_days = 10000 # the number of days to run the simulation
save_interval_days = 50 # save every 100
plot_interval_days = 25 # plot every 20
freq_log = 1000 # Log frequency (in iterations)


case_dir = "./"
if not os.path.exists(case_dir):
    os.makedirs(case_dir)

# vertex grid
xv = torch.linspace(0, Lx, nx+1, dtype=torch.float64, device=device)
yv = torch.linspace(0, Ly, ny+1, dtype=torch.float64, device=device)
x, y = torch.meshgrid(xv, yv, indexing='ij')

# layer thickness
H = torch.zeros(nl,1,1, dtype=dtype, device=device)
H[0,0,0] = 400.
H[1,0,0] = 1100.
H[2,0,0] = 2600.

# gravity
g_prime = torch.zeros(nl,1,1, dtype=dtype, device=device)
g_prime[0,0,0] = 9.81
g_prime[1,0,0] = 0.025
g_prime[2,0,0] = 0.0125

# Coriolis beta plane
f0 = 9.375e-5 # mean coriolis (s^-1)
beta = 1.754e-11 # coriolis gradient (m^-1 s^-1)

# forcing
yc = 0.5 * (yv[1:] + yv[:-1]) # cell centers
tau0 = 0.08 / 1000
zwcl_m = zwcl*110574
shift = zwcl_m - 0.5*Ly
jshift = int(shift / dy)  # shift in grid points
curl_tau = -tau0*2*torch.pi/Ly*torch.sin(2*torch.pi*(yc-shift)/(Ly-shift))
curl_tau[0:jshift] = 0.0  # set curl to zero below the southern zero wind curl line
curl_tau = curl_tau.tile((nx, 1)).unsqueeze(0).repeat(n_ens, 1, 1, 1)
# drag
delta_ek = 2.
bottom_drag_coef = delta_ek / H[-1].cpu().item() * f0 / 2
side_drag_coef = 1.0e-7 # side drag coefficient (1/s)

# Domain mask
mask = torch.from_numpy(mask2d).to(device=device, dtype=dtype).T  # Convert to tensor
dA = dx * dy
integral_D1 = curl_tau.sum(dim=(-2, -1)) * dA
print("Original ∬_D f dA", integral_D1.squeeze())


# Integrate over masked region:
#D0_mask = torch.ones(nx, ny, device=device)
D0_mask = (1.0 - mask).to(device=device, dtype=dtype)  # D0 is the masked region
# Expand mask to match curl_tau shape
D0_mask_expanded = D0_mask.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, nx, ny)

curl_tau_D0 = curl_tau * D0_mask_expanded  # shape: (n_ens, 1, nx, ny)

integral_D0 = curl_tau_D0.sum(dim=(-2, -1)) * dA  # shape: (n_ens, 1)
print("Original ∬_D0 f dA", integral_D0.squeeze())

area_D0 = D0_mask.sum() * dA  # scalar
area_D1 = mask.sum() * dA  # scalar

# Area-averaged curl over D0
C = integral_D0 / area_D1  # shape: (n_ens, 1)
C = C.squeeze()  # shape: (n_ens,)
print("Contanst", C)

curl_tau = curl_tau + C.view(-1, 1, 1, 1)
curl_tau = mask.to(device=device) * curl_tau
integral_D1 = curl_tau.sum(dim=(-2, -1)) * dA
print("∬_D1 f' dA (should be ~0):", integral_D1.squeeze())

param = {
    'nx': nx,
    'ny': ny,
    'nl': nl,
    'n_ens': n_ens,
    'mask': mask,
    'Lx': Lx,
    'Ly': Ly,
    'flux_stencil': 5,
    'H': H,
    'g_prime': g_prime,
    'tau0': tau0,
    'f0': f0,
    'beta': beta,
    'bottom_drag_coef': bottom_drag_coef,
    'device': 'cpu',
    'dtype': dtype,
    'dt': dt, # time-step (s)
    'case_directory': case_dir,
}

# Write the sparse matrices to file
nma_obj = NMA(param,model=TUML)
nma_obj.construct_splig()
nma_obj.write() # Save the nma_obj to disk in the case directory

# Save a few plots for reference
# Plot the mask
plt.figure()
plt.imshow(nma_obj.mask_d,interpolation='nearest', aspect='equal')
plt.colorbar(fraction=0.046,location='right')
plt.savefig(f'{case_dir}/dirichlet-mask.png')
plt.close()

plt.figure()
plt.imshow(nma_obj.mask_n,interpolation='nearest', aspect='equal')
plt.colorbar(fraction=0.046,location='right')
plt.savefig(f'{case_dir}/neumann-mask.png')
plt.close()

# Update the MQGeometry device
param['device'] = device
qg = QGFV(param)
qg.set_wind_forcing(curl_tau)

#######################
# Define the immersed boundry method drag field

qbound = qg.masks.q_distbound1.cpu().numpy().squeeze()
# Step 1: Smooth with a uniform kernel (e.g., 3x3 or 5x5 mean filter)
smoothed = scipy.ndimage.uniform_filter(qbound, size=9)

# Step 2: Threshold — convert to binary mask
# You can use a low threshold (e.g., > 0.1) to fill coastlines
ibm_dragfield = smoothed*qg.masks.q.cpu().numpy().squeeze() #(smoothed > 0.01).astype(np.uint8)
# Rescal the ibm_dragfield to the range [0, 1]
ibm_dragfield = (ibm_dragfield - ibm_dragfield.min()) / (ibm_dragfield.max() - ibm_dragfield.min())*side_drag_coef

qg.side_wall_drag = torch.from_numpy(ibm_dragfield).to(device=device, dtype=dtype).unsqueeze(0).repeat(n_ens, nl, 1, 1)
# Save a few plots for reference
# Plot the mask
plt.figure()
plt.imshow(ibm_dragfield,interpolation='nearest', aspect='equal')
plt.colorbar(fraction=0.046,location='right')
plt.savefig(f'{case_dir}/ibm-side-drag.png')
plt.close()

fname = os.path.join(case_dir, f'psi_mask.npy')
np.save(fname, qg.masks.psi.cpu().numpy().astype('float32'))
print(f'saved \psi mask to {fname}',flush=True)

fname = os.path.join(case_dir, f'q_mask.npy')
np.save(fname, qg.masks.q.cpu().numpy().astype('float32'))
print(f'saved q mask to {fname}',flush=True)

t0 = time.time()
n_steps = int(n_days*24*3600 / dt)
freq_save = int(save_interval_days*24*3600 / dt)
freq_plot = int(plot_interval_days*24*3600 / dt)

# time integration
for n in range(1, n_steps+1):
    qg.step() # one RK3 integration step
    t += dt

    if n % 500 == 0 and torch.isnan(qg.psi).any():
        raise ValueError(f'Stopping, NAN number in p at iteration {n}.')

    if freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
        w_over_f0 = (qg.laplacian_h(qg.psi, qg.dx, qg.dy) / qg.f0 * qg.masks.psi).cpu()
        f,a = plt.subplots(1,1, figsize=(20,10))
        im = a.imshow(w_over_f0[0,0].T, cmap='bwr', vmin=-0.2, vmax=0.2, origin='lower')
        f.colorbar(im )
        f.suptitle(f'Upper layer relative vorticity (units of $f_0$), {t/(365*86400):.2f} yrs')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.savefig(os.path.join(case_dir, f'vort_{n:06d}.png'), dpi=300)
        plt.close(f)

        f,a = plt.subplots(1,1, figsize=(20,10))
        im = a.imshow(qg.psi[0,0].cpu().T, cmap='bwr', vmin=-90.0e3, vmax=90.0e3, origin='lower')
        f.colorbar(im )
        f.suptitle(f'Upper layer stream function, {t/(365*86400):.2f} yrs')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.savefig(os.path.join(case_dir, f'psi_{n:06d}.png'), dpi=300)
        plt.close(f)

    if freq_log > 0 and n % freq_log == 0:
        print(f'{n=:06d}, t={t/(365*24*60**2):.2f} yr, '\
              f'q: {qg.q.sum().cpu().item():+.5E}, '\
              f'qabs: {qg.q.abs().sum().cpu().item():+.5E}',flush=True)

    if freq_save > 0 and n % freq_save == 0:
        n_days = int(t % (24*3600) )
        fname = os.path.join(case_dir, f'psi_{n:06d}d.npy')
        np.save(fname, qg.psi.cpu().numpy().astype('float32'))
        print(f'saved psi to {fname}',flush=True)

        fname = os.path.join(case_dir, f'vort_{n:06d}d.npy')
        vorticity = (qg.laplacian_h(qg.psi, qg.dx, qg.dy) * qg.masks.psi).cpu()
        np.save(fname, vorticity.numpy().astype('float32'))
        print(f'saved vorticity to {fname}',flush=True)

total_time = time.time() - t0
print(total_time,flush=True)
print(f'{total_time // 3600}h {(total_time % 3600) // 60} min',flush=True)


