#!/usr/bin/env python
#

# Notes
#
# Assumptions
# ---------------
#    Uniform grid spacing in each direction ( dx = constant, dy = constant )
#
#    The dirichlet modes are associated with the rotational part of the flow (stream function)
#    and are defined on the vorticity points.
# 
#    The neumann modes are associated with the divergent part of the flow (velocity potential)
#    and are defined on the tracer points    
#
#
# Grid
# -------
#   The mask applies to vorticity points on the
#   arakawa c-grid (z-points below).
#
#
#   Vorticity points are in the range (0:nx,0:ny) [nx+1, ny+1] grid points
#   Boundary values for the vorticity 
#
#   Vorticity points are suffixed with "g", e.g. "xg" and "yg" refer to
#   zonal and meridional positions at vorticity points   
#
#   Tracer points have a range of (0,nx-1,0:ny-1) [nx,ny] grid points
#   
#   Tracer points are suffixed with "c", e.g. "xc" and "yc" refer to
#   zonal and meridional positions at tracer points   
#
#   Tracer points are defined offset to the north and east from a vorticity
#   point by half the tracer cell width with the same (i,j) index. 
#   For example, xc(i,j) = xg(i,j) + dx*0.5 and yc(i,j) = yg(i,j) + dy*0.5
#
#
#     z(i,j+1) ----- z(i+1,j+1)
#       |                 |
#       |                 |
#       |                 |
#       |      t(i,j)     |
#       |                 |
#       |                 |
#       |                 |
#     z(i,j) -------- z(i+1,j)
#
#
# Masking
# ------------
#   A mask value of 0 corresponds to a wet cell (this cell is not masked)
#   A mask value of 1 corresponds to a dry cell (this cell is masked)
#   This helps with working with numpy's masked arrays
#


import numpy as np
import torch
import torch.nn.functional as F

from MQGeometry.qgm import QGFV, Masks

def dfdx_c(f,dx):
    """Calculates the x-derivative of a function on tracer points
    and returns a function on u-points.Homogeneous neumann boundary
    conditions are assumed on the grid boundaries."""
    return F.pad( 
        (f[...,1:,:]-f[...,:-1,:])/dx, (0,0,1,1), mode='constant',value=0.
    )

def dfdx_v(f,dx):
    """Calculates the x-derivative of a function on v points
    and returns a function on u-points.Homogeneous neumann boundary
    conditions are assumed on the grid boundaries."""
    return F.pad( 
        (f[...,1:,:]-f[...,:-1,:])/dx, (0,0,1,1), mode='constant',value=0.
    )

def dfdx_u(f,dx):
    """Calculates the x-derivative of a function on u points
    and returns a function on tracer-points."""
    return (f[...,1:,:]-f[...,:-1,:])/dx

def dfdy_c(f,dy):
    """Calculates the y-derivative of a function on tracer points
    and returns a function on v-points. Homogeneous neumann boundary
    conditions are assumed on the grid boundaries."""
    return F.pad( 
        (f[...,:,1:]-f[...,:,:-1])/dy, (1,1,0,0), mode='constant',value=0.
    )

def dfdy_u(f,dy):
    """Calculates the y-derivative of a function on tracer points
    and returns a function on u-points. Homogeneous neumann boundary
    conditions are assumed on the grid boundaries."""
    return F.pad( 
        (f[...,:,1:]-f[...,:,:-1])/dy, (1,1,0,0), mode='constant',value=0.
    )

def dfdy_v(f,dy):
    """Calculates the y-derivative of a function on v points
    and returns a function on tracer points."""
    return (f[...,:,1:]-f[...,:,:-1])/dy

def laplacian_c(f, masku, maskv, dx, dy):
    """2-D laplacian on the tracer points. On tracer points, we are
    working with the divergent modes, which are associated with neumann
    boundary conditions. """
    return dfdx_u( dfdx_c(f,dx)*masku, dx ) + dfdy_v( dfdy_c(f,dy)*maskv, dy )

def TtoU(f):
    """Interpolates from arakawa c-grid tracer point to u-point.
    Input is first padded in the x-direction to prolong the data
    past the boundaries, consistent with homogeneous neumann conditions
    for data at tracer points. """
    fpad = F.pad( f[...,:,:], (0,0,1,1), mode='replicate')
    return 0.5*(fpad[...,1:,:]+fpad[...,:-1,:])

def TtoV(f):
    """Interpolates from arakawa c-grid tracer point to v-point.
    Input is first padded in the x-direction to prolong the data
    past the boundaries, consistent with homogeneous neumann conditions
    for data at tracer points. """
    fpad = F.pad( f, (1,1,0,0), mode='replicate')
    return 0.5*(fpad[...,:,1:]+fpad[...,:,:-1])   

def UtoT(f):
    return 0.5*(f[...,1:,:]+f[...,:-1,:])

def VtoT(f):
    return 0.5*(f[...,:,1:]+f[...,:,:-1])    


class TUML(QGFV):
   
    @property
    def area_n(self):
        return self.masks.q*self.dx*self.dy # area of neumann mode cells

    @property
    def area_d(self):
        return self.masks.psi*self.dx*self.dy # area of dirichlet mode cells
    
    def get_uv(self):
        return self.grad_perp(self.psi, self.dx, self.dy)
    
    def vorticity(self,u,v):
        return (dfdx_v(v,self.dx) - dfdy_u(u,self.dy))*self.masks.psi
    
    def divergence(self,u,v):
        return (dfdx_u(u,self.dx) + dfdy_v(v,self.dy))*self.masks.q

    def apply_laplacian_n(self,x):
        """Laplacian with neumann boundary conditions"""
        return -laplacian_c(x,self.masks.u,self.masks.v,self.dx,self.dy)*self.masks.q.squeeze()

    def apply_laplacian_d(self,f):
        """Laplacian with dirichlet boundary conditions"""
        fm_g = self.masks.psi.squeeze()*f # Mask the data to apply homogeneous dirichlet boundary conditions
        return -self.masks.psi.squeeze()*self.laplacian_h(fm_g,self.dx,self.dy)
    
    def total_area_d(self):
        return torch.sum( self.area_d ) 
    
    def total_area_n(self):
        return torch.sum( self.area_n ) 
    
    def area_integral_n(self,f):
        return torch.sum(  f*self.dx*self.dy*self.masks.q )

    def area_integral_d(self,f):
        return torch.sum(  f*self.dx*self.dy*self.masks.psi )
    
    def dfdx_n(self,f):
        """Takes the x-derivative of a function defined at the same points
        as the neumann modes.
        For this model, we return the derivative back on the U-points"""
        return dfdx_c(f,self.dx)

    def dfdy_n(self,f):
        """Takes the y-derivative of a function defined at the same points
        as the neumann modes.
        
        For this model, we return the derivative back on V-points"""
        return dfdy_c(f,self.dy)

    def dfdx_d(self,f):
        """Takes the x-derivative of a function defined at the same points
        as the dirichlet modes.

        For this model, we return the derivative back on the V-points"""
        return dfdx_u(f,self.dx)

    def dfdy_d(self,f):
        """Takes the y-derivative of a function defined at the same points
        as the dirichlet modes.
        For this model, we return the derivative back on the V-points
        """
        return dfdy_v(f,self.dy)
    
    def map_T_to_U(self,f):
        return TtoU(f)

    def map_T_to_V(self,f):
        return TtoV(f)

    def map_U_to_T(self,f):
        return UtoT(f)

    def map_V_to_T(self,f):
        return VtoT(f)
