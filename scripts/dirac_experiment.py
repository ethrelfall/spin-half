# dirac_experiment.py

# What this does: is set up to run a spin-half equation on square giving same result as scalar problem similar to Deluzet-Narski.
# Solution is u=v=(1/8-1/2*y^2) sin(m pi x).
# Is not isotropic (at least as supplied, with sqrt_eps=1), rather is intended to test solutions to spin-half equation.

# Known problems
# 1. Running on a mesh of regular squares or right triangles gives nonsense output (see square mesh below, and change identifiers in BCs).
# 2. Low-accuracy computations, e.g. orders 1-3 for unit_square.msh, gives obvious artifacts

from firedrake import *

#square mesh (doesn't work!)
#import numpy as np
#xcoords = np.linspace(-0.5, 0.5, 20 + 1, dtype=np.double)
#ycoords = np.linspace(-0.5, 0.5, 20 + 1, dtype=np.double)
#mesh = TensorRectangleMesh(xcoords, ycoords, quadrilateral="True")

mesh = Mesh("unit_square.msh")  # crude triangle mesh
#mesh = Mesh("unit_square_test.msh")  # more refined triangle mesh

m = Constant(1.0)  # mode order across x; m=2 is one wavelength; higher mode order here needs higher order elements
sqrt_eps = 1.0e-0  # square root of anisotropy in scalar problem

V1 = FunctionSpace(mesh, "CG", 4)
V2 = FunctionSpace(mesh, "CG", 4)
V = V1*V2

uv = TrialFunction(V)
u, v = split(uv)
w1, w2 = TestFunctions(V)
x,y = SpatialCoordinate(mesh)

a = (grad(u)[0]*w1 + sqrt_eps*grad(v)[1]*w1 + sqrt_eps*grad(u)[1]*w2 - grad(v)[0]*w2)*dx
L = ((m*pi*cos(m*pi*x)*(0.125-0.5*y*y)+sqrt_eps*sin(m*pi*x)*(-y))*w1+(sqrt_eps*(-y)*sin(m*pi*x)-m*pi*cos(m*pi*x)*(0.125-0.5*y*y))*w2)*dx

g = Function(V)

bc1 = DirichletBC(V.sub(0), 0.0, (11,13))  #11,13 on unit_square.msh; 3,4 on TensorRectangleMesh
bc2 = DirichletBC(V.sub(1), 0.0, (11,13))

solve(a==L, g, bcs=[bc1, bc2])

up, vp = g.split()
File("dirac_experiment.pvd").write(up, vp)
