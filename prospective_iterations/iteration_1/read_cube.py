import numpy as np


#adapted form ase.io.cube (https://wiki.fysik.dtu.dk/ase/_modules/ase/io/cube.html)
def read_cube(fn, read_data=True, program=None, verbose=False):
    """Read data from CUBE file.

    fn : str or file
        Location of the cubefile.
    read_data : boolean
        If set true, the actual cube file content, i.e. an array
        containing the electronic density (or something else )on a grid
        and the dimensions of the corresponding voxels are read.
    program: str
        Use program='castep' to follow the PBC convention that first and
        last voxel along a direction are mirror images, thus the last
        voxel is to be removed.  If program=None, the routine will try
        to catch castep files from the comment lines.
    verbose : bool
        Print some more information to stdout.

    Returns a dict with the following keys:

    * 'data' : (Nx, Ny, Nz) ndarray
    * 'origin': (3,) ndarray, specifying the cube_data origin.
    """

    Bohr=0.529177249
    dct={}

    with open(fn,"r") as fileobj:
        readline = fileobj.readline
        line = readline()  # the first comment line
        line = readline()  # the second comment line

        # The second comment line *CAN* contain information on the axes
        # But this is by far not the case for all programs
        axes = []
        if 'OUTER LOOP' in line.upper():
            axes = ['XYZ'.index(s[0]) for s in line.upper().split()[2::3]]
        if not axes:
            axes = [0, 1, 2]

        # castep2cube files have a specific comment in the second line ...
        if 'castep2cube' in line:
            program = 'castep'
            if verbose:
                print('read_cube identified program: castep')

        # Third line contains actual system information:
        line = readline().split()
        natoms = int(line[0])

        # Origin around which the volumetric data is centered
        # (at least in FHI aims):
        origin = np.array([float(x) * Bohr for x in line[1::]])

        cell = np.empty((3, 3))
        shape = []

        # the upcoming three lines contain the cell information
        dct['uvecs'] = []
        for i in range(3):
            n, x, y, z = [float(s) for s in readline().split()]
            shape.append(int(n))

            # different PBC treatment in castep, basically the last voxel row is
            # identical to the first one
            if program == 'castep':
                n -= 1
            cell[i] = n * Bohr * np.array([x, y, z])
            dct['uvecs'].append(np.array([x, y, z]) * Bohr)
        dct['shape']=shape
        
        numbers = np.empty(natoms, int)
        positions = np.empty((natoms, 3))
        dct['atoms']=[]
        for i in range(natoms):
            line = readline().split()
            numbers[i] = int(line[0])
            positions[i] = [float(s) for s in line[2:]]
            dct['atoms'].append((numbers[i], positions[i] * Bohr))

        dct['cell'] = cell

        if read_data:
            data = np.array([float(s)
                             for s in fileobj.read().split()]).reshape(shape)
            if axes != [0, 1, 2]:
                data = data.transpose(axes).copy()

            if program == 'castep':
                # Due to the PBC applied in castep2cube, the last entry along each
                # dimension equals the very first one.
                data = data[:-1, :-1, :-1]

            dct['data'] = data
            dct['origin'] = origin

    return dct



######################################################3
#adapted form ase.io.cube (https://wiki.fysik.dtu.dk/ase/_modules/ase/io/cube.html)
def write_cube(fileobj, grid, comment=None):
    """
    Function to write a cube file.
    """
    Bohr=0.529177249
    
    data=grid['data']
    if data is None:
        data = np.ones((2, 2, 2))
    data = np.asarray(data)

    if data.dtype == complex:
        data = np.abs(data)

    if comment is None:
        comment = ''
    else:
        comment = comment.strip()
    fileobj.write(comment)

    fileobj.write('\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')

    origin=grid['origin']
    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.asarray(origin) / Bohr

    fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n'
                  .format(len(grid['atoms']), *origin))

    uvecs=grid['uvecs']
    for i in range(3):
        d = uvecs[i] / Bohr
        fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(grid['shape'][i], *d))

    positions = [ a[1]/Bohr for a in grid['atoms']]
    numbers = [ a[0] for a in grid['atoms']]
    for Z, (x, y, z) in zip(numbers, positions):
        fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}{4:12.6f}\n'
                      .format(Z, 0.0, x, y, z))

    data.tofile(fileobj, sep='\n', format='%e')
