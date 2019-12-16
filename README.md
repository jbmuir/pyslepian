# slepians - Python implementation of Cartesian Slepian functions for arbitrary triangulations

This is a very bare-bones implementation for calculating slepian functions for triangulated subdomains of the 2D real domain. It hasn't been maintained since 2016 but may be a useful reference or base point for others who wish to further implement this functionality in Python.

## Usage

```
import pyslepians
import numpy as np
import triangle

eastings = np.random.normal(0, 1, 100)
northings = np.random.normal(0, 1, 100)

z = eastings + 1j*northings #creating a complex array is a trick to get distance
d = len(z)
m, n = np.meshgrid(z, z)
distance_s_matrix = np.abs(m-n)**2
coords=np.array(list(zip(eastings, northings)))
hull = triangle.convex_hull(coords)
vertices = np.array([coords[index] for index in hull[:,0]])
S = 80 #shannon number; effective size of the parametrization
maxarea = 0.25 #maximum area of the mesh cells; may need to be decreased to allow slepian construction to work. 
points_dict = {"vertices":vertices,"segments":np.array([[i,i+1] for i in range(len(vertices)-1)] + [[len(vertices)-1,0]])}
quad_rules = pyslepians.get_tri_quadrature(points_dict, maxarea)
w, A = pyslepians.compute_slepians_at_points(quad_rules, S, eastings, northings) # concentration values and slepian function evaluations at given points
```