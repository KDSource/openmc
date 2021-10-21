import os
import subprocess
from enum import Enum
from math import cos, sin, pi
import numpy as np
import h5py
import pandas as pd
from uncertainties import ufloat
import matplotlib.pyplot as plt

import openmc
from openmc.source import ParticleType
import openmc.checkvalue as cv

import mcpl

class PDGCode(Enum):
    NEUTRON = 2112
    PHOTON = 22
    ELECTRON = 11
    POSITRON = -11

class SurfaceSource:
    """ Read a surface source file and return a DataFrame in MCPL format

    Parameters
    ----------
    filepath: str
        Path to original source file
    translation: list
        Translation for the position variables [cm]
        Default is None
    rotation: list
        Rotation for the position and direction variables [deg]
        Default is None
    domain: dict
        Range of the variables: var_min <= 'var' <= var_max
        It must be defined like: {'var':[var_min, var_max]}
        List of possible variables: type [PDG], E [MeV], x [cm], y [cm], z [cm], u, v, w, wgt, R [cm], theta [rad], phi [rad], psi [rad], log(E0/E)
    normal_direction: str
        Versor which is normal to the surface source
    set_domain_first: bool
        Define if the variables domains and the normal direction must be set before or after the translation and rotation    
    E0: float
        Reference energy to calculate the lethargy log(E0/E)
        Default is 20 MeV
    **kwargs
        Keyword arguments to pass to :class:`h5py.File`
    """

    def __init__(self, filepath, translation=None, rotation=None, domain=None, normal_direction='w', set_domain_first=True, E0=20):
        # Initialize SurfaceSource class atributes
        self._filename = str(filepath)
        self._extension = os.path.splitext(self._filename)[-1]
        self.translation = translation
        self.rotation = rotation
        self.domain = domain
        self.set_domain_first = set_domain_first
        self.E0 = E0
        self.uvw = normal_direction

        self._df = self.__read__()

    def __read__(self):
        if self._extension == '.h5':
            self._read_from_h5()

        elif self._extension == '.gz' or self._extension == '.mcpl':
            self._read_from_mcpl()

        else:
            self._read_from_ascii()

        if self.uvw == 'u':
            X, Y, Z, U, V, W = 'y', 'z', 'x', 'v', 'w', 'u'
        elif self.uvw == 'v':
            X, Y, Z, U, V, W = 'z', 'x', 'y', 'w', 'u', 'v'
        elif self.uvw == 'w':
            X, Y, Z, U, V, W = 'x', 'y', 'z', 'u', 'v', 'w'

        df = self._df
        df['R'] = (df[X].to_numpy()**2+df[Y].to_numpy()**2)**0.5
        df['theta'] = (np.arctan2(df[Y].to_numpy(), df[X].to_numpy()))
        df['mu'] = df[W].to_numpy()
        df['psi'] = (np.arccos(df[W].to_numpy()))
        df['phi'] = (np.arctan2(df[V].to_numpy(), df[U].to_numpy()))
        df['log(E0/E)'] = np.log10(self.E0/df['E'].to_numpy())
        self._df = df

        if self.set_domain_first and self.domain!=None:
            self._domain()
        if self.translation!=None:
            self._translation()
        if self.rotation!=None:
            self._rotation()
        if not self.set_domain_first and self.domain!=None:
            self._domain()
        self._normalize_direction()

        if self.rotation!=None or self.translation!=None:
            df = self._df
            df['R'] = (df[X].to_numpy()**2+df[Y].to_numpy()**2)**0.5
            df['theta'] = (np.arctan2(df[Y].to_numpy(), df[X].to_numpy()))
            df['mu'] = df[W].to_numpy()
            df['psi'] = (np.arccos(df[W].to_numpy()))
            df['phi'] = (np.arctan2(df[V].to_numpy(), df[U].to_numpy()))
            self._df = df
        
        return self._df

    def _read_from_h5(self):
        with h5py.File(self._filename, 'r') as fh:
            df = pd.DataFrame(columns=['id','type','E','x','y','z','u','v','w','t', 'wgt','px','py','pz','userflags'])
    
            ### Change from OpenMC ParticleType type to MCPL PDGCode
            df['type'] = fh['source_bank']['particle']
            df.loc[df['type']==ParticleType.NEUTRON.value, 'type'] = PDGCode.NEUTRON.value   # neutron
            df.loc[df['type']==ParticleType.PHOTON.value, 'type'] = PDGCode.PHOTON.value     # photon
            df.loc[df['type']==ParticleType.ELECTRON.value, 'type'] = PDGCode.ELECTRON.value     # electron
            df.loc[df['type']==ParticleType.POSITRON.value, 'type'] = PDGCode.POSITRON.value    # positron

            df['id'] = df.index
            df['E'] = fh['source_bank']['E']*1e-6  # MeV
            df['x'] = fh['source_bank']['r']['x']  # cm
            df['y'] = fh['source_bank']['r']['y']  # cm
            df['z'] = fh['source_bank']['r']['z']  # cm
            df['u'] = fh['source_bank']['u']['x']
            df['v'] = fh['source_bank']['u']['y']
            df['w'] = fh['source_bank']['u']['z']
            df['wgt'] = fh['source_bank']['wgt']
            df['t'] = 0.0
            df['px'] = 0.0
            df['py'] = 0.0
            df['pz'] = 0.0
            df['userflags'] = 0x00000000

            self._df = df

    def _read_from_ascii(self):
        df = pd.DataFrame(np.loadtxt(self._filename, skiprows=5), columns=['id','type','E','x','y','z','u','v','w','t', 'wgt','px','py','pz','userflags'])
        
        self._df = df

    def _read_from_mcpl(self):
        mcpl_file = mcpl.MCPLFile(self._filename)
        plist = []
        for i,p in enumerate(mcpl_file.particles):
            plist.append([i, p.pdgcode, p.ekin, p.x, p.y, p.z, p.ux, p.uy, p.uz, p.time, p.weight, p.polx, p.poly, p.polz, p.userflags])
        df = pd.DataFrame(plist, columns=['id','type','E','x','y','z','u','v','w','t', 'wgt','px','py','pz','userflags'])
        self._df = df

    def _write_to_h5(self, output_file, **kwargs):
        df = self._df
        df.loc[df['type']==PDGCode.NEUTRON.value, 'type'] = ParticleType.NEUTRON.value   # neutron
        df.loc[df['type']==PDGCode.PHOTON.value, 'type'] = ParticleType.PHOTON.value     # photon
        df.loc[df['type']==PDGCode.ELECTRON.value, 'type'] = ParticleType.ELECTRON.value     # electron
        df.loc[df['type']==PDGCode.POSITRON.value, 'type'] = ParticleType.POSITRON.value    # positron

        ### 'id','type','E','x','y','z','u','v','w','t', 'wgt','px','py','pz','userflags'

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

        # cv.check_iterable_type("source particles", source_particles, SourceParticle)
        # arr = np.array([s.to_tuple() for s in source_particles], dtype=source_dtype)
        arr = np.array([((s[3], s[4], s[5]), (s[6], s[7], s[8]), (s[2]), (s[10]), (0), (0), (s[1])) for s in df.values], dtype=source_dtype)

        kwargs.setdefault('mode', 'w')
        with h5py.File(output_file, **kwargs) as fh:
            fh.attrs['filetype'] = np.string_("source")
            fh.create_dataset('source_bank', data=arr, dtype=source_dtype)

    def _write_to_ascii(self, output_file):        
         ## Write the ASCII-based format file
        with open(output_file,'w') as fo:
            df = self._df

            fo.write('#MCPL-ASCII\n')
            fo.write('#GENERATED FROM OPENMC\n')
            fo.write('#NPARTICLES: {:d}\n'.format(len(df)))
            fo.write('#END-HEADER\n')
            fo.write("index     pdgcode               ekin[MeV]                   x[cm]          "
                        +"         y[cm]                   z[cm]                      ux                  "
                        +"    uy                      uz                time[ms]                  weight  "
                        +"                 pol-x                   pol-y                   pol-z  userflags\n")     
            
            fmtstr="%5i %11i %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g 0x%08x\n"
            for i,s in enumerate(df.values):
                # fo.write('{:.0f}\t{:.0f}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:s}\n'.format(*[j for j in s]))
                fo.write(fmtstr%(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], s[13], int(s[14])))   

    def _write_to_mcpl(self, output_file):
        self._write_to_ascii('temp.txt')
        subprocess.call(['ascii2mcpl', 'temp.txt', output_file])
        subprocess.call(['rm', 'temp.txt'])          

    def _translation(self):
        cv.check_length('tracks translation', self.translation, 3)

        df = self._df

        df['x'] = df['x'].to_numpy() + self.translation[0]
        df['y'] = df['y'].to_numpy() + self.translation[1]
        df['z'] = df['z'].to_numpy() + self.translation[2]

        self._df = df

    def _rotation(self):  
        cv.check_length('tracks rotation', self.rotation, 3)

        df = self._df

        phi, theta, psi = np.array(self.rotation)*(pi/180.)
        c3, s3 = cos(phi), sin(phi)
        c2, s2 = cos(theta), sin(theta)
        c1, s1 = cos(psi), sin(psi)
        rotation_matrix = np.array([[c1*c2, c1*s2*s3 - c3*s1, s1*s3 + c1*c3*s2],
                                    [c2*s1, c1*c3 + s1*s2*s3, c3*s1*s2 - c1*s3],
                                    [-s2, c2*s3, c2*c3]])
        df['x'], df['y'], df['z'] = rotation_matrix @ np.array([df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()])
        df['u'], df['v'], df['w'] = rotation_matrix @ np.array([df['u'].to_numpy(), df['v'].to_numpy(), df['w'].to_numpy()])
        
        self._df = df
    
    def _normalize_direction(self):
        ### Normalize the direction vector
        df = self._df

        unorm = df['u'].to_numpy()**2 + df['v'].to_numpy()**2 + df['w'].to_numpy()**2
        
        df['u'], df['v'], df['w'] = (df['u'].to_numpy()/unorm**0.5,
                                     df['v'].to_numpy()/unorm**0.5,
                                     df['w'].to_numpy()/unorm**0.5)

        self._df = df

    def _domain(self):
        df = self._df

        for pvar, (pmin, pmax) in self.domain.items():
            if pmin != None:
                df=df[df[pvar]>=pmin]
            if pmax != None:
                df=df[df[pvar]<=pmax]

        self._df = df

    def get_pandas_dataframe(self):
        return self._df

    def get_distribution_1D(self, var, bins, factor=1.0, filters={}, total=False):
        df = self._df

        for pvar, (pmin, pmax) in filters.items():
            if pmin != None:
                df=df[df[pvar]>=pmin]
            if pmax != None:
                df=df[df[pvar]<=pmax]
        
        if type(bins)==int:
            bins = np.linspace(df[var].min(), df[var].max(), bins)
        
        if var=='psi' or var=='phi' or var=='theta':
            bins = np.deg2rad(bins)
        
        x = 0.5*(bins[1:] + bins[:-1])
        dx = bins[1:]-bins[:-1]

        p_mean = np.histogram(a=df[var], bins=bins, weights=df['wgt'])[0]
        p_stdv = np.histogram(a=df[var], bins=bins, weights=df['wgt']**2)[0]**0.5
        p_total = ufloat(sum(df['wgt']), sum(df['wgt']**2)**0.5)

        p_mean = p_mean * factor / dx
        p_stdv = p_stdv * factor / dx
        with np.errstate(divide='ignore', invalid='ignore'):
            p_erel = np.nan_to_num(p_stdv / p_mean)
        p_total = p_total * factor

        if var=='psi' or var=='phi' or var=='theta':
            bins = np.rad2deg(bins)
            x = 0.5*(bins[1:] + bins[:-1])
        
        if var=='mu':
            bins = np.rad2deg(np.arccos(bins))
            x = 0.5*(bins[1:] + bins[:-1])
            var = 'psi'        

        p_df = pd.DataFrame()
        # p_df['{:s}-min'.format(var)] = bins[:-1].ravel()
        # p_df['{:s}-max'.format(var)] = bins[1:].ravel()
        p_df['{:s}'.format(var)] = x.ravel()
        
        p_df['mean'] = p_mean.ravel()
        p_df['stdv'] = p_stdv.ravel()
        p_df['erel'] = p_erel.ravel()

        if total:
            return p_total
        else:
            return p_df, bins

    def get_distribution_2D(self, var1, var2, bins1, bins2, factor=1.0, filters={}, total=False):
        df = self._df

        for pvar, (pmin, pmax) in filters.items():
            if pmin != None:
                df=df[df[pvar]>=pmin]
            if pmax != None:
                df=df[df[pvar]<=pmax]

        if type(bins1)==int:
            bins1 = np.linspace(df[var1].min(), df[var1].max(), bins1)
        if type(bins2)==int:
            bins2 = np.linspace(df[var2].min(), df[var2].max(), bins2)

        if var1=='psi' or var1=='phi' or var1=='theta':
            bins1 = np.deg2rad(bins1)
        if var2=='psi' or var2=='phi' or var2=='theta':
            bins2 = np.deg2rad(bins2)

        x = 0.5*(bins1[1:] + bins1[:-1])
        y = 0.5*(bins2[1:] + bins2[:-1])

        dx = bins1[1:]-bins1[:-1]        
        dy = bins2[1:]-bins2[:-1]
        dx,dy = np.meshgrid(dx, dy, indexing='xy')

        p_mean = np.histogram2d(x=df[var1], y=df[var2], bins=[bins1, bins2], weights=df['wgt'])[0].T
        p_stdv = np.histogram2d(x=df[var1], y=df[var2], bins=[bins1, bins2], weights=df['wgt']**2)[0].T**0.5
        p_total = ufloat(sum(df['wgt']), sum(df['wgt']**2)**0.5)

        p_mean = p_mean * factor / (dx * dy)
        p_stdv = p_stdv * factor / (dx * dy)
        with np.errstate(divide='ignore', invalid='ignore'):
            p_erel = np.nan_to_num(p_stdv / p_mean)
        p_total = p_total * factor   

        if var1=='psi' or var1=='phi' or var1=='theta':
            bins1 = np.rad2deg(bins1)
            x = 0.5*(bins1[1:] + bins1[:-1])        
        if var1=='mu':
            bins1 = np.rad2deg(np.arccos(bins1))
            x = 0.5*(bins1[1:] + bins1[:-1])
            var1 = 'psi' 

        if var2=='psi' or var2=='phi' or var2=='theta':
            bins2 = np.rad2deg(bins2)
            y = 0.5*(bins2[1:] + bins2[:-1])        
        if var2=='mu':
            bins2 = np.rad2deg(np.arccos(bins2))
            y = 0.5*(bins2[1:] + bins2[:-1])
            var2 = 'psi'  

        x,y = np.meshgrid(x, y, indexing='xy')           
        
        p_df = pd.DataFrame()
        # p_df['{:s}-min'.format(var1)] = np.meshgrid(bins1[:-1], bins2[:-1], indexing='xy')[0].ravel()
        # p_df['{:s}-max'.format(var1)] = np.meshgrid(bins1[1:], bins2[1:], indexing='xy')[0].ravel()
        # p_df['{:s}-min'.format(var2)] = np.meshgrid(bins1[:-1], bins2[:-1], indexing='xy')[1].ravel()
        # p_df['{:s}-max'.format(var2)] = np.meshgrid(bins1[1:], bins2[1:], indexing='xy')[1].ravel()
        p_df['{:s}'.format(var1)] = x.ravel()
        p_df['{:s}'.format(var2)] = y.ravel()
        
        p_df['mean'] = p_mean.ravel()
        p_df['stdv'] = p_stdv.ravel()
        p_df['erel'] = p_erel.ravel()     

        if total:
            return p_total
        else:
            return p_df, bins1, bins2

    def plot_distribution_1D(self, var, bins, factor=1.0, filters={}, **kwargs):
        df, bins = self.get_distribution_1D(var, bins, factor, filters, total=False)
        # df.loc[df['stdv']>=abs(df['mean']), 'stdv'] = 0.99999*df['stdv']
        if var=='mu':
            var='psi'
        plt.errorbar(df[var], df['mean'], df['stdv'], **kwargs)

    def plot_distribution_2D(self, var1, var2, bins1, bins2, factor=1.0, filters={}, zlabel=None, **kwargs):
        df, bins1, bins2 = self.get_distribution_2D(var1, var2, bins1, bins2, factor, filters, total=False)
        len1=len(bins1)-1
        len2=len(bins2)-1
        
        z_mean = df['mean'].to_numpy().reshape(len2, len1)
        # z_stdv = df['mean'].to_numpy().reshape(len(x), len(y))
        
        plt.pcolor(0.5*(bins1[1:]+bins1[:-1]), 0.5*(bins2[1:]+bins2[:-1]), z_mean, shading='auto', **kwargs)
        cbar=plt.colorbar(label=zlabel)
        lvls=np.linspace(cbar.vmin, cbar.vmax, 5)        
        cntrs=plt.contour(0.5*(bins1[1:]+bins1[:-1]), 0.5*(bins2[1:]+bins2[:-1]), z_mean, colors='grey', levels=lvls, vmin=cbar.vmin, vmax=cbar.vmax)
        cbar.add_lines(cntrs)

    def save_source_file(self, filename):
        new_extension = os.path.splitext(filename)[-1]

        if new_extension == '.h5':
            self._write_to_h5(filename)
        
        elif new_extension == '.mcpl':
            self._write_to_mcpl(filename)

        else:
            self._write_to_ascii(filename)