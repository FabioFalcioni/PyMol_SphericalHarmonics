from cmath import cos
from multiprocessing.sharedctypes import Value
from turtle import position
from xdrlib import ConversionError
from pymol import cgo
from pymol import cmd
from pymol.cgo import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm
import math as m
from typing import List, Dict

def showaxes():
    """showaxes is a function that loads XYZ global axes as CGO objects into PyMol viewport
    """    
    w = 0.06 # cylinder width 
    l = 0.75 # cylinder length
    h = 0.25 # cone hight
    d = w * 1.618 # cone base diameter

    obj = [CYLINDER, 0.0, 0.0, 0.0,   l, 0.0, 0.0, w, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        CYLINDER, 0.0, 0.0, 0.0, 0.0,   l, 0.0, w, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        CYLINDER, 0.0, 0.0, 0.0, 0.0, 0.0,   l, w, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        CONE,   l, 0.0, 0.0, h+l, 0.0, 0.0, d, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 
        CONE, 0.0,   l, 0.0, 0.0, h+l, 0.0, d, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 
        CONE, 0.0, 0.0,   l, 0.0, 0.0, h+l, d, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    cmd.load_cgo(obj, 'global_axes')

def unit_vector(vector: np.array) -> np.array:
    """unit_vector generates a unit vector given any numpy array

    :return: normalised vector
    """ 
    return vector / np.linalg.norm(vector)

def global_to_local(local_origin: np.array, local_axis: np.ndarray, global_coordinates: np.ndarray) -> np.ndarray:
    """global_to_local converts global coordinates to local coordinate system

    :param local_origin: n origin of the local axis system
    :param local_axis: _description_
    :param global_coordinates: _description_
    :return: _description_
    """ 
    return np.matmul(local_axis.T, (global_coordinates - local_origin).T).T

def global_to_local_frame(local_frame: List[int], global_coordinates: np.ndarray) -> np.ndarray:
    """global_to_local_frame converts global coordinates to local frame coordinates defined by three atoms

    :param local_frame: local_frame is 
    :param global_coordinates: _description_
    :return: _description_
    """    
    origin_atom = np.array(local_frame[0])
    x_axis_atom = np.array(local_frame[1])
    xy_plane_atom = np.array(local_frame[2])

    lx = unit_vector(x_axis_atom - origin_atom)
    lz = unit_vector(np.cross(unit_vector(xy_plane_atom - origin_atom), unit_vector(x_axis_atom - origin_atom)))
    ly = unit_vector(np.cross(lz,unit_vector(x_axis_atom - origin_atom)))

    local_axis = np.array([lx, ly, lz])
    return global_to_local(np.zeros(3), local_axis, global_coordinates)

def spherical_harmonics(local_frame: List[int], r: float, m: int, l: int, n_points: int, colormap: str):
    """spherical_harmonics _summary_

    :param local_frame: _description_
    :type local_frame: List[int]
    :param r: _description_
    :type r: float
    :param m: _description_
    :type m: int
    :param l: _description_
    :type l: int
    :param n_points: _description_
    :type n_points: int
    :param colormap: _description_
    :type colormap: str
    :return: _description_
    :rtype: _type_
    """    
    theta = np.linspace(0, np.pi, n_points)
    phi = np.linspace(0, 2*np.pi, n_points)
    # Create a 2-D meshgrid of (theta, phi) angles.
    theta, phi = np.meshgrid(theta, phi)
    # Calculate the Cartesian coordinates of each point in the mesh.
    xyz = np.array([ r * np.sin(theta) * np.sin(phi),
                     r * np.sin(theta) * np.cos(phi),
                     r * np.cos(theta)])
    Y = sph_harm(abs(m), l, phi, theta)
    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real
    Yx, Yy, Yz = np.abs(Y) * xyz
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap))
    cmap.set_clim(-0.5, 0.5)
    colors = cmap.to_rgba(Y.real)
    list_of_colors = []
    coords = []
    for color in colors:
        for c in color:
            list_of_colors.append(c[:3])
    coords = np.vstack((Yx.ravel(), Yy.ravel(), Yz.ravel())).T
    coords = global_to_local_frame(local_frame, coords)
    return coords, list_of_colors

def alf_axes(ref_atom, ax1,ax2,ax3):
    alf = [BEGIN, LINES]
    [alf.append(l) for l in [COLOR, 1.0, 0.0, 0.0]]
    [alf.append(l) for l in ref_atom]
    [alf.append(l) for l in ax1]
    [alf.append(l) for l in [COLOR, 0.0, 1.0, 0.0]]
    [alf.append(l) for l in ref_atom]
    [alf.append(l) for l in ax2]
    [alf.append(l) for l in [COLOR, 0.0, 0.0, 1.0]]
    [alf.append(l) for l in ref_atom]
    [alf.append(l) for l in ax3]
    cmd.load_cgo(alf, 'ALF')

def create_obj(coords, colors, define_normals=False):
    obj = [BEGIN, POINTS]

    for i in range(0, len(coords), 3):
        tri = np.array([coords[i], coords[i+1], coords[i+2]])
        v = tri[1] - tri[0]
        w = tri[2] - tri[0]
        normal = unit_vector(np.cross(v, w))

        obj.append(COLOR)
        obj.extend(colors[i])
        if define_normals:
            obj.append(NORMAL)
            obj.extend(normal)
        obj.append(VERTEX)
        obj.extend(coords[i])

        obj.append(COLOR)
        obj.extend(colors[i+1])
        if define_normals:
            obj.append(NORMAL)
            obj.extend(normal)
        obj.append(VERTEX)
        obj.extend(coords[i+1])

        obj.append(COLOR)
        obj.extend(colors[i+2])
        if define_normals:
            obj.append(NORMAL)
            obj.extend(normal)
        obj.append(VERTEX)
        obj.extend(coords[i+2])
    obj.append(END)

    return obj

def calculate_alf_cahn_ingold_prelog(iatom: int, obj_coords: np.ndarray, obj_atom_masses: List[float], connectivity: np.ndarray) -> List[int]:
    import itertools as it

    def _priority_by_mass(atoms: List[int]) -> float:
        """Returns the sum of masses of a list of Atom instances
        Args:
            :param: `atoms` a list of Atom instances:
        Returns:
            :type: `float`
            The sum of the masses of the Atom instances that were given in the input `atoms`.
        """
        return sum(obj_atom_masses[a] for a in atoms)

    def _get_priority(atom: int, level: int):
        """Returns the priority of atoms on a given level."""
        atoms = [atom]
        for _ in range(level):
            atoms_to_add = []
            for a in atoms:
                atoms_to_add.extend(
                    bonded_atom
                    for bonded_atom in _get_bonded_atoms(a)
                    if bonded_atom not in atoms
                )

            atoms += atoms_to_add

        return _priority_by_mass(atoms)

    def _max_priority(atoms: List[int]):
        """Returns the Atom instance that has the highest priority in the given list.
            Args:
            :param: `atoms` a list of Atom instances:
        Returns:
            :type: `Atom` instance
                The atom instance with the highest priority by mass.
        """
        prev_priorities = []
        level = it.count(0)
        while True:
            next_lvl = next(level)  # starts at 0
            priorities = [_get_priority(atom, next_lvl) for atom in atoms]
            if (
                priorities.count(max(priorities)) == 1
                or prev_priorities == priorities
            ):
                break
            else:
                prev_priorities = priorities
        return atoms[priorities.index(max(priorities))]

    def _get_bonded_atoms(i: int) -> List[int]:
        """_get_bonded_atoms _summary_

        :param i: _description_
        :return: _description_
        """        
        return [j for j in range(len(connectivity)) if connectivity[i, j] == 1]

    def _calculate_alf(iatom) -> List[int]:
        """Returns a list consisting of the x-axis and xy-plane Atom instances, which
        correspond to the atoms of first and second highest priorty as determined by the
        Cahn-Ingold-Prelog rules."""
        alf = [iatom]
        # we need to get 2 atoms - one for x-axis and one for xy-plane. If the molecule is 2d (like HCl), then we only need 1 atom.
        n_atoms_in_alf = 2 if len(obj_coords) > 2 else 1
        if len(obj_coords) == 1:
            raise ValueError(
                "ALF cannot be calculated because there is only 1 atom. Two or more atoms are necessary."
            )

        for _ in range(n_atoms_in_alf):
            # make a list of atoms to which the central atom is bonded to that are not in alf
            queue = [a for a in _get_bonded_atoms(iatom) if a not in alf]

            # if queue is empty, then we add the bonded atoms of the atoms that the atom of interest is connected to
            if not queue:
                queue = list(
                    it.chain.from_iterable(
                        _get_bonded_atoms(a) for a in _get_bonded_atoms(iatom)
                    )
                )
                # again remove atoms if they are already in alf
                queue = [a for a in queue if a not in alf]
            if not queue:
                raise ValueError("Check that the selection is bonded to other atoms in order to find a correct local frame.")
            max_priority_atom = _max_priority(queue)
            alf.append(max_priority_atom)
        return alf

    return [a for a in _calculate_alf(iatom)]

def wrapper(selected_atoms, molobj, r, m, l, n_points,cmap = 'viridis', ax=0):
    '''
DESCRIPTION

    Plot a spherical harmonic function onto selected atoms of a specific molecule object.

USAGE

    spherical_harmonics selected_atoms, molobj, r, m, l, n_points [, cmap [, ax ]]
    '''
    if (int(n_points)%3 != 0):
        raise ValueError("Number of points for the meshgrid needs to be a multiple of 3")
    if ax:
        showaxes()
    #Automated search of the ALF
    bonds = [bond.index for bond in cmd.get_model(molobj).bond]
    natoms = cmd.get_model(molobj).nAtom
    connectivity = np.zeros((natoms, natoms))
    for (iatom, jatom) in bonds:
        connectivity[iatom, jatom] = 1
        connectivity[jatom, iatom] = 1
    obj_coords = cmd.get_model(molobj).get_coord_list()
    obj_atom_masses = [atom.get_mass() for atom in cmd.get_model(molobj).atom]
    selected_atoms_index = [ a[1] - 1 for a in cmd.index(selected_atoms)]
    selected_atoms_coords = [coord for coord in cmd.get_model(selected_atoms).get_coord_list()]
    v = cmd.get_view()
    for selected_atom_index in selected_atoms_index:
        alf = calculate_alf_cahn_ingold_prelog(selected_atom_index, obj_coords, obj_atom_masses, connectivity)

        local_frame = [obj_coords[i] for i in alf]

        OBJECT_coords, OBJECT_colors = spherical_harmonics(local_frame, float(r),int(m),int(l),
                                                    int(n_points), colormap = cmap)
        object = create_obj(OBJECT_coords,OBJECT_colors)
        name = f'atom{selected_atom_index+1}sph_{m}_{l}'
        
        cmd.load_cgo(object, name)
        ref_atom = selected_atoms_coords[selected_atoms_index.index(selected_atom_index)]
        cmd.translate(object=name, vector = ref_atom, camera=0)
    cmd.set_view(v)
   
    
cmd.extend('spherical_harmonics',wrapper)

def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt('Spherical Harmonics', )

def spherical_harmonics_dialog():
    from pymol.Qt import QtWidgets
    QtWidgets.QMessageBox,information(None,'Spherical Harmonics','Plot Spherical Harmonics into local frame of an atom')

