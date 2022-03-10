import numpy as np

import openmc
import pytest


# check that we can successfully read wwinp files with the following contents:
#
#  - neutrons on a rectilinear mesh
#  - neutrons and photons on a rectilinear mesh

# check that the following raises the correct exceptions (for now):
#
#  - wwinp file with multiple time steps
#  - wwinp file with cylindrical or spherical mesh

mesh = openmc.RectilinearMesh()
mesh.x_grid = np.asarray([-100, -99, -97, -79.36364, -61.72727, -44.09091, -26.45455, -8.818182,
                          8.818182, 26.45455, 44.09091, 61.72727, 79.36364, 97, 99, 100])
mesh.y_grid = np.asarray([-100, -50, -13.33333, 23.33333, 60, 70, 80, 90, 100])
mesh.z_grid = np.asarray([-100, -66.66667, -33.33333, 0.0, 33.33333, 66.66667, 100])
e_bounds = np.asarray([0.0, 100000.0, 146780.0])


def test_wwinp_reader():
    ww = openmc.wwinp_to_wws('wwinp_n')[0]
    # check the mesh grid
    np.testing.assert_allclose(mesh.x_grid, ww.mesh.x_grid, rtol=1e-6)
    np.testing.assert_allclose(mesh.y_grid, ww.mesh.y_grid, rtol=1e-6)
    np.testing.assert_allclose(mesh.z_grid, ww.mesh.z_grid, rtol=1e-6)

    # check the energy bounds
    np.testing.assert_array_equal(e_bounds, ww.energy_bins)
