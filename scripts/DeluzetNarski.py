# DeluzetNarski.py

# What this does: tests straight- (lam=0) and curved-fieldlines cases of anisotropic diffusion
# after "A two field iterated AP method for highly anisotropic elliptic equations"
# by Deluzet and Narski.
# This shows the issues explored in the paper:
# 1. Null space problem best seen for lam=0 straight fieldlines: eps 1e-14 is OK-ish, eps 10^-15 is clearly wrong.
# 2. "Locking" problem for lam!=0 and low orders: solution amplitude comes out much smaller than should be.

from firedrake import *

# MODEL PARAMETERS
lam = Constant(0.0)  # parameter for varying B-field away from x-direction; 10 gives reasonable bent fieldlines, 0 is x-direction only
eps = Constant(1.0e-14)  # anisotropy parameter, 1.0 is isotropic, intended to be decreased to small positive values
# Note if changing mesh need to make sure indices in bcB and bcT are correct - see comment.
# END OF MODEL PARAMETERS

#square mesh (works)
#import numpy as np
#xcoords = np.linspace(-0.5, 0.5, 20 + 1, dtype=np.double)
#ycoords = np.linspace(-0.5, 0.5, 20 + 1, dtype=np.double)
#mesh = TensorRectangleMesh(xcoords, ycoords, quadrilateral="True")

mesh = Mesh("unit_square.msh")  # fairly crude triangle mesh

V = FunctionSpace(mesh, "CG", 2)

u = TrialFunction(V)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)

bcB = DirichletBC(V, 0.0, 11)   # y = 0 homogeneous Dirichlet, 11 on unit_square.msh, 3 on TensorRectangleMesh
bcT = DirichletBC(V, 0.0, 13)   # y = 1 homogeneous Dirichlet, 13 on unit_square.msh, 4 on TensorRectangleMesh

solution = Function(V)
solution.interpolate(0.125-0.5*y*y)
bcLR = DirichletBC(V, solution, (1,2))  # optional additional Dirichlet BC for lam=0.0 case: removes null space problem

# magnetic field lines direction unit vector
bhat = as_vector([1/sqrt(1+lam*lam*x*x*y*y*(1-x)*(1-x)*(1-y)*(1-y)),lam*x*y*(1-x)*(1-y)/sqrt(1+lam*lam*x*x*y*y*(1-x)*(1-x)*(1-y)*(1-y))])

k_par = Constant(1.0)
k_per = eps  # THIS IS EPSILON IN ANISO DIFFUSION TENSOR

flux = k_par * bhat * dot(bhat, grad(u)) + k_per * (grad(u) - bhat * dot(bhat, grad(u)))

a = inner(flux, grad(v))*dx
f = Function(V)
f.interpolate(0.0*x+eps)
L = inner(f,v)*dx

T = Function(V)

solve( a==L, T, bcs=[bcB, bcT])  # optionally add bcLR to remove null space problem and locking problem

File("DeluzetNarski.pvd").write(T)  

# output magnetic field lines
VB=VectorFunctionSpace(mesh,"CG", 1)
BField = Function(VB)
BField.interpolate(bhat)
File("DeluzetNarski_BField.pvd").write(BField)
