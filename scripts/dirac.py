# dirac.py

# What this does: is set up to run a spin-half equation on square giving same result as scalar problem similar to Deluzet-Narski.
# Solution is u=v=1/8-1/2 y^2.

# Known problems
# 1. Running on a mesh of regular squares or right triangles gives nonsense output (see square mesh below, and change identifiers in BCs).
# 2. Nice answers for order-2 elements (run as supplied) go away for higher orders.

from firedrake import *

sqrt_eps = Constant(1.0e-10)  # is supposed to correspond to sqrt of eps in Deluzet-Narski

#square mesh (doesn't work!)
#import numpy as np
#xcoords = np.linspace(-0.5, 0.5, 20 + 1, dtype=np.double)
#ycoords = np.linspace(-0.5, 0.5, 20 + 1, dtype=np.double)
#mesh = TensorRectangleMesh(xcoords, ycoords, quadrilateral="True")

mesh = Mesh("unit_square.msh")  # this is a crude triangle mesh
#mesh = Mesh("unit_square_test.msh")  # this is a more refined triangle mesh

V1 = FunctionSpace(mesh, "CG", 2)
V2 = FunctionSpace(mesh, "CG", 2)
V = V1*V2

uv = TrialFunction(V)
u, v = split(uv)
w1, w2 = TestFunctions(V)
x,y = SpatialCoordinate(mesh)

a = (grad(u)[0]*w1 + sqrt_eps*grad(v)[1]*w1 + sqrt_eps*grad(u)[1]*w2 - grad(v)[0]*w2)*dx
L = -sqrt_eps*y*(w1+w2)*dx

g = Function(V)

bc1 = DirichletBC(V.sub(0), 0.0, (11,13))  #11,13 on unit_square.msh, 3,4 on internally-generated mesh
bc2 = DirichletBC(V.sub(1), 0.0, (11,13))

solve(a==L, g, bcs=[bc1, bc2])

up, vp = g.split()
File("dirac.pvd").write(up, vp)
