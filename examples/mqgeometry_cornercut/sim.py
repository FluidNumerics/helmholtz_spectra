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

torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
print(f" Device: {device}, Data type: {dtype}",flush=True)

# grid
n_ens = 1
nl = 3
nx = 256
ny = 256
dt = 4000
Lx = 5120.0e3
Ly = 5120.0e3
dx = Lx / nx
dy = Ly / ny

# Time stepping parameters
# time params
t = 0
n_days = 4000 # the number of days to run the simulation
save_after = 2000 # save after this many days
save_interval_days = 10 # save every 10 days
plot_interval_days = 10 # save every 10 days
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
curl_tau = -tau0*2*torch.pi/Ly*torch.sin(2*torch.pi*yc/Ly).tile((nx, 1))
curl_tau = curl_tau.unsqueeze(0).repeat(n_ens, 1, 1, 1)
# drag
delta_ek = 2.
bottom_drag_coef = delta_ek / H[-1].cpu().item() * f0 / 2

# Domain mask
mask = torch.ones(nx, ny)
for i in range(nx//4):
    for j in range(ny//4):  
        mask[i,j] = 0.

dA = dx * dy
integral_D1 = curl_tau.sum(dim=(-2, -1)) * dA
print("Original ∬_D f dA", integral_D1.squeeze())


# Integrate over masked region:
D0_mask = torch.ones(nx, ny, device=device)
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
save_after_steps = int(save_after*24*3600 / dt)

print(f'Simulation parameters:\n'
      f'  n_ens: {n_ens}, nx: {nx}, ny: {ny}, nl: {nl}\n'
      f'  dt: {dt} s, n_steps: {n_steps}\n'
      f'  save_after_steps: {save_after_steps}, freq_save: {freq_save}, freq_plot: {freq_plot}, freq_log: {freq_log}\n'
      f'  case_dir: {case_dir}', flush=True)

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
        im = a.imshow(qg.psi[0,0].cpu().T, cmap='bwr', origin='lower')
        f.colorbar(im )
        f.suptitle(f'Upper layer stream function, {t/(365*86400):.2f} yrs')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.savefig(os.path.join(case_dir, f'psi_{n:06d}.png'), dpi=300)
        plt.close(f)

    if freq_log > 0 and n % freq_log == 0:
        # Compute upper layer energy
        u, v = qg.grad_perp(qg.psi, qg.dx, qg.dy)
        ke = torch.sqrt( torch.square(u[...,0,0:-1,:]) + torch.square(v[...,0,:,0:-1])).sum().cpu().item()

        print(f'{n=:06d}, t={t/(365*24*60**2):.2f} yr, '\
              f'q: {qg.q.sum().cpu().item():+.5E}, '\
              f'qabs: {qg.q.abs().sum().cpu().item():+.5E}, '\
              f'ke0: {ke:+.5E}',flush=True)

    if freq_save > 0 and n % freq_save == 0 and n >= save_after_steps:
        fname = os.path.join(case_dir, f'psi_{n:06d}d.npy')
        np.save(fname, qg.psi.cpu().numpy().astype('float32'))
        print(f'saved psi to {fname}',flush=True)

total_time = time.time() - t0
print(total_time,flush=True)
print(f'{total_time // 3600}h {(total_time % 3600) // 60} min',flush=True)


