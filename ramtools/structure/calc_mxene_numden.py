import numpy as np
import mdtraj as md
import warnings
import mdtraj.core.element as Element
import matplotlib.pyplot as plt

def calc_number_density(coord_file, trj_file, bin_width, area, dim, box_range, data_path, resnames):
    """
    Calculate a 1-dimensional number density profile for each residue specifically for mxene water adsorption study

    Parameters
    ----------
    gro_file: str
        GROMACS '.gro' file to load 
    trj_file: str
        Trajectory to load
    top_file: str
        GROMACS '.top' file to load
    bin_width: int
        Width (nm) of numpy histogram bins
    dim: int
        Dimension to calculate number density profile (0,1 or 2)
    box_range: array
        Range of coordinates in 'dim' to evaluate
    data_path: str
        Path to save txt file out to
    resnames: dict
        Contains residues to look at

    
    Attributes
    ----------
    """

    first_frame = md.load_frame(trj_file, top=coord_file, index=0)

    open('{0}/resnames.txt'.format(data_path), 'w').close()

    for resname, restype in resnames.items():
        traj = md.load(trj_file, top=coord_file,
            atom_indices=first_frame.topology.select('name {}'
                .format(restype)))

        indices = [[at.index for at in compound.atoms]
            for compound in list(traj.topology.residues)]

        if 0 in [x.mass for x in
            [atom.element for atom in traj.topology.atoms]]:
            warnings.warn("mdtraj found zero mass, setting element to hydrogen", UserWarning)
            for atom in traj.topology.atoms:
                if atom.element in [Element.virtual, Element.virtual_site]:
                    atom.element = Element.hydrogen

        x = np.histogram(traj.xyz[:, 1:, dim].reshape((-1, 1)),
            bins=np.linspace(box_range[0], box_range[1],
            num=1+round((box_range[1]-box_range[0])/bin_width)))

        np.savetxt('{0}/{1}-number-density.txt'.format(data_path, resname),
            np.vstack([x[1][:-1]+np.mean(x[1][:2])-box_range[0],
            x[0]/(area*bin_width*(len(traj)-1))]).transpose())

        with open('{0}/resnames.txt'.format(data_path), "a") as myfile:
            myfile.write(resname + '\n')

def plot_mxene_numden(resnames, ylim, filename='number_density.pdf'):
    """
    function to plot number density profiles from txt files

    Paramters
    ---------
    resnames: list
        list of resnames to find corresponding txt files
    filename: str
        Name of PDF file to return
    
    Returns
    -------
    Number density profile in PDF format
    """
    def get_color(atom_name):
        """ Get color for matplotlib plot for each atom """
        color_dict = {
                'k': 'C0',
                'li': 'C1',
                'water': 'C2',
                'water_o': 'C9',
                'water_h': 'black',
                'O': 'C3',
                'OH': 'C4',
                'F': 'C7'
                }

        return color_dict[atom_name]

    def get_alpha(atom_name):
        """ Get color for matplotlib plot for each atom """
        alpha_dict = {
                'k': 1,
                'li': 1,
                'water': 1,
                'water_o': 1,
                'water_h': 1,
                'O': 0.2,
                'OH': 0.2,
                'F': 0.2
                }

        return alpha_dict[atom_name]

    fig, ax = plt.subplots()
    for f in resnames:
        data = np.loadtxt('{}-number-density.txt'.format(f))
        ax.plot(data[:,0], data[:,1],
                label=f,
                color=get_color(f),
                alpha=get_alpha(f)
                )

    plt.xlabel('Position on Surface (nm)')
    plt.ylabel('Number Density (nm^-3)')
    plt.ylim(ylim)
    plt.legend()
    plt.savefig(filename)
