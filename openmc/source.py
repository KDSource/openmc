from enum import Enum
from math import cos, sin, pi
from numbers import Real
from xml.etree import ElementTree as ET

import numpy as np
import h5py
import pandas as pd

import openmc.checkvalue as cv
from openmc.stats.multivariate import UnitSphere, Spatial
from openmc.stats.univariate import Univariate
from ._xml import get_text


class Source:
    """Distribution of phase space coordinates for source sites.

    Parameters
    ----------
    space : openmc.stats.Spatial
        Spatial distribution of source sites
    angle : openmc.stats.UnitSphere
        Angular distribution of source sites
    energy : openmc.stats.Univariate
        Energy distribution of source sites
    filename : str
        Source file from which sites should be sampled
    library : str
        Path to a custom source library
    parameters : str
        Parameters to be provided to the custom source library

        .. versionadded:: 0.12
    strength : float
        Strength of the source
    particle : {'neutron', 'photon'}
        Source particle type

    Attributes
    ----------
    space : openmc.stats.Spatial or None
        Spatial distribution of source sites
    angle : openmc.stats.UnitSphere or None
        Angular distribution of source sites
    energy : openmc.stats.Univariate or None
        Energy distribution of source sites
    file : str or None
        Source file from which sites should be sampled
    library : str or None
        Path to a custom source library
    parameters : str
        Parameters to be provided to the custom source library
    strength : float
        Strength of the source
    particle : {'neutron', 'photon'}
        Source particle type

    """

    def __init__(self, space=None, angle=None, energy=None, filename=None,
                 library=None, parameters=None, strength=1.0, particle='neutron'):
        self._space = None
        self._angle = None
        self._energy = None
        self._file = None
        self._library = None
        self._parameters = None

        if space is not None:
            self.space = space
        if angle is not None:
            self.angle = angle
        if energy is not None:
            self.energy = energy
        if filename is not None:
            self.file = filename
        if library is not None:
            self.library = library
        if parameters is not None:
            self.parameters = parameters
        self.strength = strength
        self.particle = particle

    @property
    def file(self):
        return self._file

    @property
    def library(self):
        return self._library

    @property
    def parameters(self):
        return self._parameters

    @property
    def space(self):
        return self._space

    @property
    def angle(self):
        return self._angle

    @property
    def energy(self):
        return self._energy

    @property
    def strength(self):
        return self._strength

    @property
    def particle(self):
        return self._particle

    @file.setter
    def file(self, filename):
        cv.check_type('source file', filename, str)
        self._file = filename

    @library.setter
    def library(self, library_name):
        cv.check_type('library', library_name, str)
        self._library = library_name

    @parameters.setter
    def parameters(self, parameters_path):
        cv.check_type('parameters', parameters_path, str)
        self._parameters = parameters_path

    @space.setter
    def space(self, space):
        cv.check_type('spatial distribution', space, Spatial)
        self._space = space

    @angle.setter
    def angle(self, angle):
        cv.check_type('angular distribution', angle, UnitSphere)
        self._angle = angle

    @energy.setter
    def energy(self, energy):
        cv.check_type('energy distribution', energy, Univariate)
        self._energy = energy

    @strength.setter
    def strength(self, strength):
        cv.check_type('source strength', strength, Real)
        cv.check_greater_than('source strength', strength, 0.0, True)
        self._strength = strength

    @particle.setter
    def particle(self, particle):
        cv.check_value('source particle', particle, ['neutron', 'photon'])
        self._particle = particle

    def to_xml_element(self):
        """Return XML representation of the source

        Returns
        -------
        element : xml.etree.ElementTree.Element
            XML element containing source data

        """
        element = ET.Element("source")
        element.set("strength", str(self.strength))
        if self.particle != 'neutron':
            element.set("particle", self.particle)
        if self.file is not None:
            element.set("file", self.file)
        if self.library is not None:
            element.set("library", self.library)
        if self.parameters is not None:
            element.set("parameters", self.parameters)
        if self.space is not None:
            element.append(self.space.to_xml_element())
        if self.angle is not None:
            element.append(self.angle.to_xml_element())
        if self.energy is not None:
            element.append(self.energy.to_xml_element('energy'))
        return element

    @classmethod
    def from_xml_element(cls, elem):
        """Generate source from an XML element

        Parameters
        ----------
        elem : xml.etree.ElementTree.Element
            XML element

        Returns
        -------
        openmc.Source
            Source generated from XML element

        """
        source = cls()

        strength = get_text(elem, 'strength')
        if strength is not None:
            source.strength = float(strength)

        particle = get_text(elem, 'particle')
        if particle is not None:
            source.particle = particle

        filename = get_text(elem, 'file')
        if filename is not None:
            source.file = filename

        library = get_text(elem, 'library')
        if library is not None:
            source.library = library

        parameters = get_text(elem, 'parameters')
        if parameters is not None:
            source.parameters = parameters

        space = elem.find('space')
        if space is not None:
            source.space = Spatial.from_xml_element(space)

        angle = elem.find('angle')
        if angle is not None:
            source.angle = UnitSphere.from_xml_element(angle)

        energy = elem.find('energy')
        if energy is not None:
            source.energy = Univariate.from_xml_element(energy)

        return source


class ParticleType(Enum):
    NEUTRON = 0
    PHOTON = 1
    ELECTRON = 2
    POSITRON = 3


class SourceParticle:
    """Source particle

    This class can be used to create source particles that can be written to a
    file and used by OpenMC

    Parameters
    ----------
    r : iterable of float
        Position of particle in Cartesian coordinates
    u : iterable of float
        Directional cosines
    E : float
        Energy of particle in [eV]
    wgt : float
        Weight of the particle
    delayed_group : int
        Delayed group particle was created in (neutrons only)
    surf_id : int
        Surface ID where particle is at, if any.
    particle : ParticleType
        Type of the particle

    """
    def __init__(self, r=(0., 0., 0.), u=(0., 0., 1.), E=1.0e6, wgt=1.0,
                 delayed_group=0, surf_id=0, particle=ParticleType.NEUTRON):
        self.r = tuple(r)
        self.u = tuple(u)
        self.E = float(E)
        self.wgt = float(wgt)
        self.delayed_group = delayed_group
        self.surf_id = surf_id
        self.particle = particle

    def to_tuple(self):
        """Return source particle attributes as a tuple

        Returns
        -------
        tuple
            Source particle attributes

        """
        return (self.r, self.u, self.E, self.wgt,
                self.delayed_group, self.surf_id, self.particle.value)


def write_source_file(source_particles, filename, **kwargs):
    """Write a source file using a collection of source particles

    Parameters
    ----------
    source_particles : iterable of SourceParticle
        Source particles to write to file
    filename : str or path-like
        Path to source file to write
    **kwargs
        Keyword arguments to pass to :class:`h5py.File`

    See Also
    --------
    openmc.SourceParticle

    """
    # Create compound datatype for source particles
    pos_dtype = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
    source_dtype = np.dtype([
        ('r', pos_dtype),
        ('u', pos_dtype),
        ('E', '<f8'),
        ('wgt', '<f8'),
        ('delayed_group', '<i4'),
        ('surf_id', '<i4'),
        ('particle', '<i4'),
    ])

    # Create array of source particles
    cv.check_iterable_type("source particles", source_particles, SourceParticle)
    arr = np.array([s.to_tuple() for s in source_particles], dtype=source_dtype)

    # Write array to file
    kwargs.setdefault('mode', 'w')
    with h5py.File(filename, **kwargs) as fh:
        fh.attrs['filetype'] = np.string_("source")
        fh.create_dataset('source_bank', data=arr, dtype=source_dtype)

def read_source_file(input_file, output_range = {}, set_range_first = True,
                     translation = None, rotation = None, **kwargs):
    """ Read a .h5 source file and return a DataFrame in MCPL format

    Parameters
    ----------
    input_file: str of path-like
        Path to original source file
    output_range: dict
        Range of the variables
        It must be defined like {'var':[var_min, var_max]}
        List of possible variables: type, E, x, y, z, u, v, w, wgt
    set_range_first: bool
        Define if the setting of the variables ranges must be before or after the translation and rotation
    translation: list
        Translation for the position variables
    rotation:
        Rotation for the position and direction variables
    **kwargs
        Keyword arguments to pass to :class:`h5py.File`

    """

    ### Read the .h5 file
    kwargs.setdefault('mode', 'r')
    with h5py.File(input_file, **kwargs) as fh:
        print("Reading {:s} source file...".format(input_file))

        df = pd.DataFrame(columns=['id','type','E','x','y','z','u','v','w','t', 'wgt','px','py','pz','userflags'])
    
        ### Change from OpenMC int type to MCPL PDG code
        df['type'] = fh['source_bank']['particle']
        df.loc[df['type']==0, 'type'] = 2112   # neutron
        df.loc[df['type']==1, 'type'] = 22     # photon
        df.loc[df['type']==2, 'type'] = None   # electron
        df.loc[df['type']==3, 'type'] = None   # positron
        df = df[df['type']!=None]
        print("Number of total particles in source file: {}".format(len(df)))

        df['id'] = df.index
        df['E'] = fh['source_bank']['E']*1e-6
        df['x'] = fh['source_bank']['r']['x']
        df['y'] = fh['source_bank']['r']['y']
        df['z'] = fh['source_bank']['r']['z']
        df['u'] = fh['source_bank']['u']['x']
        df['v'] = fh['source_bank']['u']['y']
        df['w'] = fh['source_bank']['u']['z']
        df['wgt'] = fh['source_bank']['wgt']
        df['t'] = 0.0
        df['px'] = 0.0
        df['py'] = 0.0
        df['pz'] = 0.0
        df['userflags'] = '0x00000000'

        ### Check and set the ranges of the variables BEFORE the translation and rotation of the variables
        if set_range_first == True:
            for pvar, (pmin, pmax) in output_range.items():
                if pmin != None:
                    df=df[df[pvar]>=pmin]
                if pmax != None:
                    df=df[df[pvar]<=pmax]        

        ### Execute the translation of the position variables
        if translation != None:
            df['x'] += translation[0]
            df['y'] += translation[1]
            df['z'] += translation[2]

        ### Execute the rotation of the position and direction variables
        if rotation != None:
            phi, theta, psi = np.array(rotation)*(pi/180.)
            c3, s3 = cos(phi), sin(phi)
            c2, s2 = cos(theta), sin(theta)
            c1, s1 = cos(psi), sin(psi)
            rotation_matrix = np.array([[c1*c2, c1*s2*s3 - c3*s1, s1*s3 + c1*c3*s2],
                                        [c2*s1, c1*c3 + s1*s2*s3, c3*s1*s2 - c1*s3],
                                        [-s2, c2*s3, c2*c3]])
            df['x'], df['y'], df['z'] = np.dot(rotation_matrix, np.array([df['x'], df['y'], df['z']]))
            df['u'], df['v'], df['w'] = np.dot(rotation_matrix, np.array([df['u'], df['v'], df['w']]))

        ### Round position and direction variables
        df = df.round({'x': 5, 'y': 5, 'z': 5, 'u': 5, 'v': 5, 'w': 5})

        ### Normalize the direction vector
        df['u'], df['v'], df['w'] = (df['u']/(df['u']**2+df['v']**2+df['w']**2)**0.5,
                                     df['v']/(df['u']**2+df['v']**2+df['w']**2)**0.5,
                                     df['w']/(df['u']**2+df['v']**2+df['w']**2)**0.5)

        ### Check and set the ranges of the variables AFTER the translation and rotation of the variables
        if set_range_first == False:
            for pvar, (pmin, pmax) in output_range.items():
                if pmin != None:
                    df=df[df[pvar]>=pmin]
                if pmax != None:
                    df=df[df[pvar]<=pmax]
        
        df['id'] = np.arange(len(df))
        print("Number of particles in DataFrame: {}".format(len(df)))
        return df

def h5_to_ssv(input_file, output_file, output_range = {}, set_range_first = True,
              translation = None, rotation = None, **kwargs):
    """ Transform a .h5 source file into .ssv format using MCPL format

    Parameters
    ----------
    input_file: str of path-like
        Path to original source file
    output_file: str or path-like
        Path to`source file to write
    output_range: dict
        Range of the variables
        It must be defined like {'var':[var_min, var_max]}
        List of possible variables: type, E, x, y, z, u, v, w, wgt
    translation: list
        Translation for the position variables
    rotation:
        Rotation for the position and direction variables
    **kwargs
        Keyword arguments to pass to :class:`h5py.File`

    """ 

    ### Write the .ssv file
    with open(output_file,'w') as fo:
        ### Read the .h5 file
        df = read_source_file(input_file, output_range, set_range_first, translation, rotation, **kwargs) 
        fo.write('#MCPL-ASCII\n')
        fo.write('#GENERATED FROM OPENMC\n')
        fo.write('#NPARTICLES: {:d}\n'.format(len(df)))
        fo.write('#END-HEADER\n')
        fo.write('index\tpdgcode\tekin[MeV]\tx[cm]\ty[cm]\tz[cm]\tux\tuy\tuz\ttime[ms]\tweight\tpol-x\tpol-y\tpol-z\tuserflags\n')        
        for i,s in enumerate(df.values):
            fo.write('{:.0f}\t{:.0f}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:s}\n'.format(*[j for j in s]))

        print("Saved original {:s} source file into new {:s} source file.".format(input_file, output_file))