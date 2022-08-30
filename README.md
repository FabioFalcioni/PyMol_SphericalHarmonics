# PyMol Spherical Harmonics
This repository containts the Plugin 'spherical_harmonics.py' for PyMol (tested on version 2.5.0 Open-Source).
This adds the functionality of plotting spherical harmonics functions on any atom of a molecule loaded in PyMol.
The functionality is extremely helpful for visualising ***Atomic Orbitals*** or ***Spherical Multipole Moments***.

![alt text](https://github.com/FabioFalcioni/PyMol_SphericalHarmonics/blob/main/spherical_harmonics.gif)

# Usage
To load the `spherical_harmonics.py` script as a PyMol plugin, please follow this [link](https://pymolwiki.org/index.php/Plugins).

A typical PyMol session to run the spherical harmonics plugin will look like this:
|![alt text](https://github.com/FabioFalcioni/PyMol_SphericalHarmonics/blob/main/example.png) |
|:--:|
| <b>Figure showing an example usage of the spherical_harmonics.py plugin. The image shows a Qzz (i.e. m=2, l=0) component of the Quadrupole of benzene carbons. The 'test' molecule is the text.xyz in the ***test*** folder</b>|

## Variables
1. **selected_atoms** : PyMol selection of atoms onto which the chosen spherical harmonic function will be plotted.
2. **molobj** : PyMol molecule object. This can be a sub-selection of your system or your entire system as loaded.
3. **r** : magnitude of the chosen spherical harmonic function.
4. **m** : order of the spherical harmonic function.
5. **l** : degree of the spherical harmonic function.
6. **n_points** : number of points for the meshgrid.
7. **cmap** (=*viridis* by default): any colormap from the matplotlib package. Examples can be found [here](https://matplotlib.org/stable/gallery/color/colormap_reference.html).
8. **ax** (=0 by default): shows global cartesian axes.

# To do
- Add triangular meshgrid for the CGO object.
- Add a QTWidget in PyMol GUI

# License
The MIT License makes this plugin available for everyone. You are more than welcome to help with the development of this repository.
Please cite this github page if you use the plugin for your studies/research.
