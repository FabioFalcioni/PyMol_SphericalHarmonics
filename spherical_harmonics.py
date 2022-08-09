from cmath import cos
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
    v = cmd.get_view()
    obj = [
        cgo.BEGIN, cgo.LINES,
        cgo.COLOR, 1.0, 0.0, 0.0,
        cgo.VERTEX,   0.0, 0.0, 0.0,
        cgo.VERTEX,  5.0, 0.0, 0.0,
        cgo.COLOR, 0.0, 1.0, 0.0,
        cgo.VERTEX, 0.0,   0.0, 0.0,
        cgo.VERTEX, 0.0,  5.0, 0.0,
        cgo.COLOR, 0.0, 0.0, 1.0,
        cgo.VERTEX, 0.0, 0.0,   0.0,
        cgo.VERTEX, 0.0, 0.0,   5.0,
        cgo.END
    ]
    cmd.load_cgo(obj, 'axes')
    cmd.set_view(v)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
                     
def rotate_test(selection, coords):
    a = unit_vector([1.0,0.0,0.0])
    selection_coords = list(cmd.get_model(selection).get_coord_list())
    b = unit_vector(np.subtract(selection_coords[-1],selection_coords[0]))
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
    v_x = skew(v)
    I = np.identity(3)
    R = I + v_x + (np.dot(v_x,v_x)) * ((1-c)/(s**2))
    test = np.dot(R,a)
    new_coords = []
    for coord in coords:
        new_coords.append(np.dot(R,coord))
    return new_coords


def euler_to_quaternion(phi, theta, psi):
 
        qw = m.cos(phi/2) * m.cos(theta/2) * m.cos(psi/2) + m.sin(phi/2) * m.sin(theta/2) * m.sin(psi/2)
        qx = m.sin(phi/2) * m.cos(theta/2) * m.cos(psi/2) - m.cos(phi/2) * m.sin(theta/2) * m.sin(psi/2)
        qy = m.cos(phi/2) * m.sin(theta/2) * m.cos(psi/2) + m.sin(phi/2) * m.cos(theta/2) * m.sin(psi/2)
        qz = m.cos(phi/2) * m.cos(theta/2) * m.sin(psi/2) - m.sin(phi/2) * m.sin(theta/2) * m.cos(psi/2)
 
        return [qw, qx, qy, qz]

def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def Rx(theta):
  return np.array([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
def Rz(theta):
  return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
def Ry(theta):
  return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])

def c_matrix(local_frame):
    selection_coords = [list(cmd.get_model(l).get_coord_list())[0] for l in local_frame]
    ref_atom = np.array(selection_coords[0])
    x_axis_atom = np.array(selection_coords[1])
    xy_plane_atom = np.array(selection_coords[2])

    # c_matrix = np.empty((3,3))

    # row1 = (x_axis_atom - ref_atom) / np.linalg.norm(x_axis_atom - ref_atom)
    x_axis = x_axis_atom - ref_atom
    xy_plane = xy_plane_atom - ref_atom
    
    sigma_fflux = -np.dot(x_axis, xy_plane) / np.dot(x_axis, x_axis)
    y_vec = sigma_fflux * x_axis + xy_plane
    # z_axis = np.cross(x_axis,y_vec)
    # #row2 = y_vec / np.linalg.norm(y_vec)
    # row2 = (y_vec) / np.sqrt(np.dot(y_vec,y_vec))
    # row3 = np.cross(row1,row2)
    # c_matrix[0, :] = row1
    # c_matrix[1, :] = row2
    # c_matrix[2, :] = row3
    # print(np.linalg.det(c_matrix))



    # r12 = x_axis_atom - ref_atom
    # r13 = xy_plane_atom - ref_atom

    # mod_r12 = np.linalg.norm(r12)

    # r12 /= mod_r12

    # ex = r12
    # s = sum(ex * r13)
    # ey = r13 - s * ex

    # ey /= np.sqrt(sum(ey * ey))
    # ez = np.cross(ex, ey)

    # c_matrix = np.array([ex, ey, ez])
    # print(np.linalg.det(c_matrix))

    #Rotating global Vx onto local Vx
    bx = unit_vector(x_axis_atom - ref_atom)
    by = unit_vector(y_vec)
    bz = np.cross(bx,by)
    # # bz = np.cross(unit_vector(xy_plane_atom - ref_atom), unit_vector(x_axis_atom - ref_atom))
    # # by = np.cross(bz,unit_vector(x_axis_atom - ref_atom))
    phi, theta, psi = get_angles(bx,by,bz)
    print(phi,theta,psi)
    # # c_matrix = euler_to_quaternion(phi,theta,psi)
    # cy = np.matmul(Rz(psi),by)
    # cz = np.matmul(Ry(theta),np.matmul(Rz(psi),bz))
    # new_theta = get_angles(bx, cy, bz)[1]
    # print(new_theta)
    # new_phi = get_angles(bx,bx,cz)[2]
    # print(new_phi)

    c_matrix = euler_to_quaternion(phi,theta,psi)

    # c_matrix = np.matmul(Rz(psi),Ry(new_theta),Rz(new_phi))

    
    #Rotating global Vy onto local Vy
    
    # if bz[-1] < 0:
    #     theta = np.arccos(np.dot(bz, az)) + np.pi/2
    # else:
    #     theta = np.arccos(np.dot(bz, az)) 
    # R2 = np.array(
    #     [
    #         [1, 0, 0],
    #         [0, np.cos(theta), -np.sin(theta)],
    #         [0, np.sin(theta), np.cos(theta)],
    #     ]
    # )
    # print(np.linalg.det(R2))
    # c_matrix = np.matmul(R2, R1,R2)
    # az = unit_vector(np.array([0,0,1]))
    # bz = unit_vector(np.cross(xy_plane_atom - ref_atom, x_axis_atom - ref_atom))
    # v2 = np.cross(az,bz)
    # s2 = np.linalg.norm(v2)
    # c2 = np.dot(az,bz)
    # v_z = skew(v2)
    # R2 = np.eye(3) + v_z + (np.dot(v_z,v_z)) * ((1-c2)/(s2**2))
    # print(np.linalg.det(R2))
    return c_matrix

def global_to_local(local_origin, local_axis, global_coordinates):
    """global_to_local converts global coordinates to local coordinate system

    :param local_origin: n origin of the local axis system
    :param local_axis: _description_
    :param global_coordinates: _description_
    :return: _description_
    """ 
    return np.matmul(local_axis.T, (global_coordinates - local_origin).T).T


def global_to_local_frame(local_frame, global_coordinates):
    origin_atom = np.array(local_frame[0])
    x_axis_atom = np.array(local_frame[1])
    xy_plane_atom = np.array(local_frame[2])

    bx = unit_vector(x_axis_atom - origin_atom)
    bz = unit_vector(np.cross(unit_vector(xy_plane_atom - origin_atom), unit_vector(x_axis_atom - origin_atom)))
    by = unit_vector(np.cross(bz,unit_vector(x_axis_atom - origin_atom)))

    local_axis = np.array([bx, by, bz])
    return global_to_local(np.zeros(3), local_axis, global_coordinates)


def get_angles(local_x,local_y,local_z):
    global_x = np.array([1, 0, 0])
    global_y = np.array([0, 1, 0])
    global_z = np.array([0, 0, 1])
    # temp = np.empty((3,3))
    # temp[:, 0] = global_x
    # temp[:, 1] = local_x
    # temp[:, 2] = np.cross(global_x,local_x)
    cosine = np.dot(global_x,local_x)
    sine = np.linalg.norm(np.cross(global_x,local_x))
    phi = np.arctan2(sine,cosine)
    #phi = np.arccos(np.dot(global_x,local_x))
    
    # temp = np.empty((3,3))
    # temp[:, 0] = global_y
    # temp[:, 1] = local_y
    # temp[:, 2] = np.cross(global_y,local_y)
    cosine = np.dot(global_y,local_y)
    sine = np.linalg.norm(np.cross(global_x,local_x))
    theta = np.arctan2(sine,cosine)
    #theta = np.arccos(np.dot(global_y,local_y))

    # temp = np.empty((3,3))
    # temp[:, 0] = global_z
    # temp[:, 1] = local_z
    # temp[:, 2] = np.cross(global_z,local_z)
    cosine = np.dot(global_z,local_z)
    sine = np.linalg.norm(np.cross(global_x,local_x))
    psi = np.arctan2(sine,cosine)
    #psi = np.arccos(np.dot(global_z,local_z))

    return (phi,theta,psi)
    

def spherical_harmonics(local_frame,r,m,l, n_points, colormap):
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
    # C = c_matrix(local_frame)
    for color in colors:
        for c in color:
            list_of_colors.append(c[:3])

    coords = np.vstack((Yx.ravel(), Yy.ravel(), Yz.ravel())).T
    coords = global_to_local_frame(local_frame, coords)
    # new_coords = []
    # for coord in coords:
    #     new_coords.append(qv_mult(C,tuple(coord)))
    # coords = np.matmul(C, coords).T

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

    def _get_bonded_atoms(i):
        return [j for j in range(len(connectivity)) if connectivity[i, j] == 1]

    def _calculate_alf(iatom) -> List["Atom"]:
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
            max_priority_atom = _max_priority(queue)
            alf.append(max_priority_atom)
        return alf

    return [a for a in _calculate_alf(iatom)]

def wrapper(selected_atom, molobj, r, m, l, n_points,cmap = 'viridis', ax=0):
    selected_atom_coords = cmd.get_model(selected_atom).get_coord_list()
    if len(selected_atom_coords) != 1:
        raise ValueError("Selection must be only one atom")
    if ax:
        showaxes()
    bonds = [bond.index for bond in cmd.get_model(molobj).bond]
    natoms = cmd.get_model(molobj).nAtom
    connectivity = np.zeros((natoms, natoms))
    for (iatom, jatom) in bonds:
        connectivity[iatom, jatom] = 1
        connectivity[jatom, iatom] = 1

    selected_atom_index = cmd.index(selected_atom)[0][-1] - 1
    obj_coords = cmd.get_model(molobj).get_coord_list()
    obj_atom_masses = [atom.get_mass() for atom in cmd.get_model(molobj).atom]

    alf = calculate_alf_cahn_ingold_prelog(selected_atom_index, obj_coords, obj_atom_masses, connectivity)
    local_frame = [obj_coords[i] for i in alf]

    OBJECT_coords, OBJECT_colors = spherical_harmonics(local_frame, float(r),int(m),int(l),
                                                int(n_points), colormap = cmap)
    object = create_obj(OBJECT_coords,OBJECT_colors)
    if ax:
        showaxes()
    name = f'atom{selected_atom_index+1}sph_{m}_{l}'
    v = cmd.get_view()
    cmd.load_cgo(object, name)
    ref_atom = selected_atom_coords[0]
    cmd.translate(object=name, vector = ref_atom, camera=0)
    cmd.set_view(v)
   
    
cmd.extend('spherical_harmonics',wrapper)

def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt('Spherical Harmonics', )

def spherical_harmonics_dialog():
    from pymol.Qt import QtWidgets
    QtWidgets.QMessageBox,information(None,'Spherical Harmonics','Plot Spherical Harmonics into local frame of an atom')

