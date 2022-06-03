import numpy as np
import param_bases as parbase
import param_pairs as parpair


#############
# functions
##############

def Rx(deg):
    rad = deg*np.pi/180.
    cs = np.cos(rad)
    sn = np.sin(rad)
    return np.array([
        [1., 0, 0],
        [0, cs,-sn],
        [0, sn, cs]
    ])

def Ry(deg):
    rad = deg*np.pi/180.
    cs = np.cos(rad)
    sn = np.sin(rad)
    return np.array([
        [cs, 0, sn],
        [0, 1., 0],
        [-sn, 0, cs]
    ])

def Rz(deg):
    rad = deg*np.pi/180.
    cs = np.cos(rad)
    sn = np.sin(rad)
    return np.array([
        [cs,-sn, 0],
        [sn, cs, 0],
        [0, 0, 1.]      
    ])

def pdb_line(atom, resid, base, i, segname='DNA1'):
    return f"ATOM{atom:>7} {base.atoms[i]:^4} {base.name:>3}   {resid:>3} " \
    + f"{base.xyz[i][0]:11.3f}{base.xyz[i][1]:8.3f}{base.xyz[i][2]:8.3f}" \
    + f"  1.00  1.00      {segname:>4}\n"

####################################
# Classes
###################
class Base:
    def __init__(self, name, strand='I', label = 1):

        if name == 'g': 
            name = 'G'

        letters = {
            "A": "ADE", 
            "C": "CYT",
            "c": "C_",  
            "G": "GUA",
            "T": "THY", 
            "U": "URA"
        }
        self.name = letters[name]
        if name == 'c':
            label_name = f"C{label}"
            self.atoms = list(parbase.Coor_labels[label_name].keys())
            self.xyz = np.array(list(parbase.Coor_labels[label_name].values()))
        else:
            self.atoms = list(parbase.Coor_bases[name].keys())
            self.xyz = np.array(list(parbase.Coor_bases[name].values()))
        
        #flip base around x-axis if on strand II
        if strand == 'II':
            self.xyz = self.xyz @ Rx(180.)

    def write_pdb(self, file=None):
        if file == None:
            file_name = 'pdbs/b' + self.name + '.pdb'
        else:
            file_name = 'pdbs/' + file + '.pdb'
        out = open(file_name,'w')
        atom = 1
        for i in range(2,len(self.atoms)):
            line = pdb_line(atom, 1, self, i, segname='DNA1')
            out.write(line)
            atom += 1
        out.close()


class Pair:
    def __init__(self, name, na_type, label=1):

        if na_type == 'DNA':
            letters = {
                'A': (0,'T'), 'T': (1,'A'), 
                'C': (2,'G'), 'G': (3,'C'),
                'c': (2,'G'), 'g': (3,'c')
            }
        elif na_type == 'RNA':
            letters = {
                'A': (0,'U'), 'U': (1,'A'), 
                'C': (2,'G'), 'G': (3,'C'),
                'c': (2,'G'), 'g': (3,'c')
            }
        b, WC = letters[name]

        self.bases  = [Base(name,'I',label), Base(WC,'II',label)]
        #global transformation for base pair step
        self.Tg = np.eye(3)
        self.rg = np.zeros((1,3))
        # calculate (but DO NOT implement) local transformation 
        # of bases in the pair
        omega = parpair.pair_geom[na_type]["Propeller"][b] 
        sigma = parpair.pair_geom[na_type]["Opening"][b]
        kappa = parpair.pair_geom[na_type]["Buckle"][b]
        Sy    = parpair.pair_geom[na_type]["Stretch"][b]
        Sz    = parpair.pair_geom[na_type]["Stagger"][b]
        Sx    = parpair.pair_geom[na_type]["Shear"][b]
        if omega == 0 and kappa == 0:
            gamma = 0.0
            phi_prime = 0.0
        else:
            gamma = np.sqrt(kappa**2 + sigma**2)
            phi_prime = 180.0/np.pi*np.arccos(kappa/gamma)
        if sigma >= 0.0:
            phi_prime = np.abs(phi_prime)
        elif sigma < 0.0:
            phi_prime = -np.abs(phi_prime)
        # local transformation for bases in the pair
        self.dr = np.array([Sx, Sy, Sz]).reshape(1,3)
        self.T1 = Ry(-phi_prime) @ Rx( gamma/2) @ Ry(phi_prime + omega/2)
        self.T2 = Ry(-phi_prime) @ Rx(-gamma/2) @ Ry(phi_prime - omega/2)

    def unflat(self):
        #UNDO the base pair step
        self.bases[0].xyz[2:] = (self.bases[0].xyz[2:] - self.rg) @ self.Tg 
        self.bases[1].xyz[2:] = (self.bases[1].xyz[2:] - self.rg) @ self.Tg 
        # UNFLAT the bases 
        self.bases[0].xyz[2:] = self.bases[0].xyz[2:] @ self.T1.T + self.dr/2.
        self.bases[1].xyz[2:] = self.bases[1].xyz[2:] @ self.T2.T - self.dr/2.
        #REDO the base pair step
        self.bases[0].xyz[2:] = self.bases[0].xyz[2:] @ self.Tg.T + self.rg
        self.bases[1].xyz[2:] = self.bases[1].xyz[2:] @ self.Tg.T + self.rg


    def stats(self):
        distC1 = np.linalg.norm(self.bases[1].xyz[1] - self.bases[0].xyz[1])
        distNs = np.linalg.norm(self.bases[1].xyz[2] - self.bases[0].xyz[2])
        if self.bases[0].name == 'ADE' or self.bases[0].name == 'GUA':
            distCs = np.linalg.norm(self.bases[1].xyz[-1] - self.bases[0].xyz[3])
        else:
            distCs = np.linalg.norm(self.bases[1].xyz[3] - self.bases[0].xyz[-1])

        print(f"C1'-C1': {distC1:5.1f}")
        print(f"RN9-YN1: {distNs:5.1f}")
        print(f"RC8-YC6: {distCs:5.2f}")
     

    def write_pdb(self, file=None):
        if file == None:
            file_name = 'pdbs/bp' + self.bases[0].name[0] + self.bases[1].name[0] + '.pdb'
        else:
            file_name = 'pdbs/' + file 
        out = open(file_name,'w')
        atom = 1
        #first base in pair
        for i in range(2,len(self.bases[0].atoms)):
            line = pdb_line(atom, 1, self.bases[0], i, segname='DNA1')
            out.write(line)
            atom += 1
        #second base in pair
        for i in range(2,len(self.bases[1].atoms)):
            line = pdb_line(atom, 1, self.bases[1], i, segname='DNA2')
            out.write(line)
            atom += 1
        out.close()


####
# step
###
def helix_axis(pair1, pair2):
    """
    Determines helix axis from given base pairs at positions i and i+1
    following Rosenberg 1976
    """
    # 1) Select equivalent atoms
    equivalent_atoms_i = np.vstack((
        pair1.bases[0].xyz[0], #center
        pair1.bases[0].xyz[1], pair1.bases[1].xyz[1], # C1'/C1'
        pair1.bases[0].xyz[3], pair1.bases[1].xyz[3]  # RN9/YN1
    ))
    equivalent_atoms_ip1 = np.vstack((
        pair2.bases[0].xyz[0], #center
        pair2.bases[0].xyz[1], pair2.bases[1].xyz[1], #C1'/C1'
        pair2.bases[0].xyz[3], pair2.bases[1].xyz[3]  # RN9/YN1
    ))
    #calculate vectors from one set of atoms to the other
    Del_i_ip1 = (equivalent_atoms_ip1 - equivalent_atoms_i)
    #middle point of these vectors
    Avg_i_ip1 = (equivalent_atoms_ip1 + equivalent_atoms_i)/2.
    
    # 2) determine helix axis (normal) and translation along it (a)
    pinvD  = np.linalg.pinv(Del_i_ip1)
    normal =  np.sum(pinvD, axis=1)
    # translation distance along helix axis
    rise = 1/np.linalg.norm(normal)
    #this is the RISE
    #print(rise)

    # 3) find the origin of rotation as the intersection of the bisector normals
    # a) generate all line direction vectors
    n = np.cross(normal, Del_i_ip1) 
    n /= np.linalg.norm(n, axis=1, keepdims=True) # normalized
    # b) generate the array of all projectors 
    # I - n*n.T
    projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis,:]
    # c) generate matrix A and vector b for the least squares
    A = projs.sum(axis=0)
    b = ( projs @ Avg_i_ip1[:,:,np.newaxis] ).sum(axis=0)
    # d) solve the least squares problem 
    origin = np.linalg.lstsq(A, b, rcond=None)[0]

    # 4) determine the angle of rotation about the helix axis
    #this is the TWIST

    return np.squeeze(origin), normal, rise


def base_step(letter, pair1, form, label):
    na_type = form[:3]

    #first make a flat pair
    pair2 = Pair(letter, na_type, label)

    if na_type == 'DNA':
        label2index = {'A': 0,'T': 1,'C': 2,'G': 3}
    elif na_type == 'RNA':
        label2index = {'A': 0,'U': 1,'C': 2,'G': 3}

    b1 = label2index[pair1.bases[0].name[0]]
    b2 = label2index[pair2.bases[0].name[0]]

    Omega = parpair.step_geometry[form]["Twist"][b1][b2]
    rho   = parpair.step_geometry[form]["Roll"][b1][b2]
    tau   = parpair.step_geometry[form]["Tilt"][b1][b2]
    Dz    = parpair.step_geometry[form]["Rise"][b1][b2]
    Dy    = parpair.step_geometry[form]["Slide"][b1][b2]
    Dx    = parpair.step_geometry[form]["Shift"][b1][b2]

    if tau != 0 or rho != 0:
        Gamma = np.sqrt(rho**2 + tau**2)
        phi = 180.0/np.pi*np.arccos(rho/Gamma)
    else:
        Gamma = 0.0
        phi = 0.0
    if tau >= 0.0:
        phi = np.abs(phi)
    elif tau < 0.0:
        phi = -np.abs(phi)

    T_mst_i = Rz(Omega/2 - phi) @ Ry(Gamma/2) @ Rz(phi)
    r_ip1_i = np.array([Dx, Dy, Dz]).reshape(1,3) @ T_mst_i.T

    T_ip1_i = Rz(Omega/2 - phi) @ Ry(Gamma) @ Rz(Omega/2 + phi) #Typo in paper
    # in the paper this is given as 
    #T_ip1_i = Rz(Omega/2 - phi) @ Ry(Gamma) @ Rz(Omega/2 - phi)    

    pair2.Tg = pair1.Tg @ T_ip1_i  
    pair2.rg = pair1.rg + r_ip1_i @ pair1.Tg.T

    pair2.bases[0].xyz = pair2.bases[0].xyz @ pair2.Tg.T + pair2.rg
    pair2.bases[1].xyz = pair2.bases[1].xyz @ pair2.Tg.T + pair2.rg
    
    #########
    # identify helix origin 
    # and translation vector (also rotation axis)
    origin, normal, rise = helix_axis(pair1, pair2)
    
    return pair2, origin, normal, rise

################
# Helix
#####

class Helix():
    def __init__(self, sequence, form='DNAseqB', label=1, verbose=False):
        self.seq     = sequence
        self.form    = form
        self.length  = len(sequence)
        self.pairs   = []
        self.origins = []
        self.normals = []
        nodes   = []

        rises = []

        na_type = form[:3]

        #first make a flat pair
        pair = Pair(sequence[0], na_type, label)
        self.pairs.append(pair)
        #collect nodes of flat pair
        nodes.append([pair.bases[0].xyz[0], 
            pair.bases[0].xyz[1], pair.bases[1].xyz[1]])

        #proceed with the rest of the sequence
        for letter in sequence[1:]:
            #base_step always makes a flat pair
            pair2, origin, normal, rise = base_step(letter, pair, form, label)
            rises.append(rise)
            #collect nodes of flat pair
            nodes.append([pair2.bases[0].xyz[0], 
                pair2.bases[0].xyz[1], pair2.bases[1].xyz[1]])
            #as well as origin and normal calculated from flat pairs
            self.origins.append(origin)
            self.normals.append(normal)

            pair = pair2
            self.pairs.append(pair)

        #pair.stats()
        self.nodes = np.array(nodes)
        #print("nodes shape:")
        #print(self.nodes.shape)
        if verbose:
            rises = np.array(rises)
            np.set_printoptions(precision=6,suppress=True)
            print(rises)
            print(np.mean(rises))


    def unflat(self):
        for pair in self.pairs:
            pair.unflat()


    def write_pdb(self, file=None):
        if file == None:
            if self.length > 6:
                file_name = 'pdbs/hx'+self.form + '_' + self.seq[:2] + '-' + str(self.length-4) + '-' + self.seq[-2:] + '.pdb'
            else:
                file_name = 'pdbs/hx'+self.form + '_' + self.seq + '.pdb'
        else:
            file_name = 'pdbs/' + file# + '.pdb'
        out = open(file_name,'w')
        atom, resid = 1, 1
        #go over strand I
        for pair in self.pairs:
            base = pair.bases[0]
            for i in range(2,len(base.atoms)):
                line = pdb_line(atom, resid, base, i, segname='DNA1')
                out.write(line)
                atom += 1
            resid += 1
        resid = 1
        #go over strand II
        for pair in reversed(self.pairs):
            base = pair.bases[1]
            for i in range(2,len(base.atoms)):
                line = pdb_line(atom, resid, base, i, segname='DNA2')
                out.write(line)
                atom += 1
            resid += 1
        out.close()


    def write_axis_pdb(self, file='pdbs/axis.pdb'):
        file_name = file
        out = open(file_name,'w')
        atom, resid = 1, 1
        
        for origin in self.origins:
            line = f"ATOM{atom:>7} {'S':^4} {'GUA':>3}   {resid:>3} " \
    + f"{origin[0]:11.3f}{origin[1]:8.3f}{origin[2]:8.3f}" \
    + f"  1.00  1.00      {'DNAx':>4}\n"
            out.write(line)
            atom += 1
            resid += 1
        for i in range(atom-2):
            line = f"CONECT{i+1:>5}{i+2:>5}\n"
            out.write(line)
        out.close()


    def write_center_pdb(self, file='pdbs/center.pdb'):
        file_name = file
        out = open(file_name,'w')
        atom, resid = 1, 1
        #go over strand I
        for pair in self.pairs:
            base = pair.bases[0]
            line = pdb_line(atom, resid, base, 0, segname='DNAc')
            out.write(line)
            atom += 1
            resid += 1
        for i in range(atom-2):
            line = f"CONECT{i+1:>5}{i+2:>5}\n"
            out.write(line)
        out.close()


    def write_nodes_pdb(self, file='nodes'):
        file_name = 'pdbs/' + file + '_nodes.pdb'
        out = open(file_name,'w')

        types = ['S','C','C']
        atom, resid = 1, 1
        #go over strand I
        for j,node in enumerate(self.nodes):
            name = self.pairs[j].bases[0].name
            for i in range(3):
                line = f"ATOM{atom:>7} {types[i]:^4} {name:>3}   {resid:>3} " \
                + f"{node[i][0]:11.3f}{node[i][1]:8.3f}{node[i][2]:8.3f}" \
                + f"  1.00  1.00      {'DNAx':>4}\n"
                out.write(line)
                atom += 1
            resid += 1
        out.close()


    def write_models_pdb(self, vecs, vals):
        return True

