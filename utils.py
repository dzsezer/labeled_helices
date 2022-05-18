import numpy as np
import param_bases as parbase
import param_pairs as parpair



def rigid_transform_3D(A, B):
    # Input: expects 3xN matrix of points
    # B = R@A + t
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector

    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

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
    def __init__(self, name, strand='I'):

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
        self.atoms = list(parbase.Coordinates[name].keys())
        self.xyz = np.array(list(parbase.Coordinates[name].values()))
        
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
        for i in range(1,len(self.atoms)):
            line = pdb_line(atom, 1, self, i, segname='DNA1')
            out.write(line)
            atom += 1
        out.close()


class Pair:
    def __init__(self, name, na_type, flat):

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

        self.bases  = [Base(name,'I'), Base(WC,'II')]

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

        #transformations for bases in the pair
        self.dr = np.array([Sx, Sy, Sz]).reshape(1,3)
        self.T1 = Ry(-phi_prime) @ Rx( gamma/2) @ Ry(phi_prime + omega/2)
        self.T2 = Ry(-phi_prime) @ Rx(-gamma/2) @ Ry(phi_prime - omega/2)
        #global transformations for base pair step
        self.Tg = np.eye(3)
        self.rg = np.zeros((1,3))

        if not flat:
            self.unflat()


    def unflat(self):
        #UNDO the base pair step
        self.bases[0].xyz = (self.bases[0].xyz - self.rg) @ self.Tg 
        self.bases[1].xyz = (self.bases[1].xyz - self.rg) @ self.Tg 
        # UNFLAT the bases 
        self.bases[0].xyz = self.bases[0].xyz @ self.T1.T + self.dr/2.
        self.bases[1].xyz = self.bases[1].xyz @ self.T2.T - self.dr/2.
        #REDO the base pair step
        self.bases[0].xyz = self.bases[0].xyz @ self.Tg.T + self.rg
        self.bases[1].xyz = self.bases[1].xyz @ self.Tg.T + self.rg


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
        for i in range(1,len(self.bases[0].atoms)):
            line = pdb_line(atom, 1, self.bases[0], i, segname='DNA1')
            out.write(line)
            atom += 1
        #second base in pair
        for i in range(1,len(self.bases[1].atoms)):
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
    # 1) Select equiuvalent atoms
    equivalent_atoms_i = np.vstack((
        pair1.bases[0].xyz[0], #center
        pair1.bases[0].xyz[1], pair1.bases[1].xyz[1], # C1'/C1'
        pair1.bases[0].xyz[2], pair1.bases[1].xyz[2]  # RN9/YN1
    ))
    equivalent_atoms_ip1 = np.vstack((
        pair2.bases[0].xyz[0], #center
        pair2.bases[0].xyz[1], pair2.bases[1].xyz[1], #C1'/C1'
        pair2.bases[0].xyz[2], pair2.bases[1].xyz[2]  # RN9/YN1
    ))
    #calculate vectors from one set of atoms to the other
    Del_i_ip1 = (equivalent_atoms_ip1 - equivalent_atoms_i)
    #middle point of these vectors
    Avg_i_ip1 = (equivalent_atoms_ip1 + equivalent_atoms_i)/2.
    
    # 2) determine helix axis (normal) and translation along it (a)
    pinvD  = np.linalg.pinv(Del_i_ip1)
    normal =  np.sum(pinvD, axis=1)
    # translation distance along helix axis
    a = 1/np.linalg.norm(normal)
    #this is the RISE
    #print(a)

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

    return np.squeeze(origin), normal


def base_step(letter, pair, form='B'):
    na_type = form[:3]

    #first make a flat pair
    pair2 = Pair(letter, na_type, flat=True)

    if na_type == 'DNA':
        l2i = {'A': 0,'T': 1,'C': 2,'G': 3}
    elif na_type == 'RNA':
        l2i = {'A': 0,'U': 1,'C': 2,'G': 3}

    b1 = l2i[pair.bases[0].name[0]]
    b2 = l2i[pair2.bases[0].name[0]]

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

    T_ip1_i = Rz(Omega/2 - phi) @ Ry(Gamma) @ Rz(Omega/2 + phi) #FretMatrix
    #T_ip1_i = Rz(Omega/2 - phi) @ Ry(Gamma) @ Rz(Omega/2 - phi)    

    pair2.Tg = pair.Tg @ T_ip1_i  
    pair2.rg = pair.rg + r_ip1_i @ pair.Tg.T

    pair2.bases[0].xyz = pair2.bases[0].xyz @ pair2.Tg.T + pair2.rg
    pair2.bases[1].xyz = pair2.bases[1].xyz @ pair2.Tg.T + pair2.rg
    
    #########
    # identify helix origin 
    # and translation vector (also rotation axis)
    origin, normal = helix_axis(pair, pair2)
    
    return pair2, origin, normal

################
# Helix
#####

class Helix():
    def __init__(self, sequence, form='DNAavgB', flat=True):
        self.seq = sequence
        self.form = form
        self.length = len(sequence)
        self.listI = []
        self.listII = []
        self.origins = []
        self.normals = []

        na_type = form[:3]

        #first make a flat pair
        pair = Pair(sequence[0], na_type, flat=True)

        #self.listI.append(pair.bases[0])
        #self.listII.append(pair.bases[1])

        for letter in sequence[1:]:
            pair2, origin, normal = base_step(letter, pair, form=form)

            self.origins.append(origin)
            self.normals.append(normal)

            #unflat the previous pair and store coordinates
            if not flat:
                pair.unflat()
            #pair.stats()
            self.listI.append(pair.bases[0])
            self.listII.append(pair.bases[1])

            pair = pair2

        #unflat the last pair and store coordinates
        if not flat: 
            pair.unflat()
        #pair.stats()
        self.listI.append(pair.bases[0])
        self.listII.append(pair.bases[1])
     

    def write_pdb(self, file=None):
        if file == None:
            if self.length > 6:
                file_name = 'pdbs/hx' +self.form + '_' + self.seq[:2] + '-' + str(self.length-4) + '-' + self.seq[-2:] + '.pdb'
            else:
                file_name = 'pdbs/hx' +self.form + '_' + self.seq + '.pdb'
        else:
            file_name = 'pdbs/' + file# + '.pdb'
        out = open(file_name,'w')
        atom, resid = 1, 1
        #go over strand I
        for base in self.listI:
            for i in range(1,len(base.atoms)):
                line = pdb_line(atom, resid, base, i, segname='DNA1')
                out.write(line)
                atom += 1
            resid += 1
        resid = 1
        #go over strand II
        for base in reversed(self.listII):
            for i in range(1,len(base.atoms)):
                line = pdb_line(atom, resid, base, i, segname='DNA2')
                out.write(line)
                atom += 1
            resid += 1
        out.close()


    def write_axis_pdb(self, file='pdbs/axis.pdb'):
        file_name = file
        out = open(file_name,'w')
        atom, resid = 1, 1
        #go over strand I
        for origin in self.origins:
            line = f"ATOM{atom:>7} {'S':^4} {'GUA':>3}   {resid:>3} " \
    + f"{origin[0]:11.3f}{origin[1]:8.3f}{origin[2]:8.3f}" \
    + f"  1.00  1.00      {'DNAx':>4}\n"
            out.write(line)
            atom += 1
        for i in range(atom-2):
            line = f"CONECT{i+1:>5}{i+2:>5}\n"
            out.write(line)
        out.close()


    def write_center_pdb(self, file='pdbs/center.pdb'):
        file_name = file
        out = open(file_name,'w')
        atom, resid = 1, 1
        #go over strand I
        for base in self.listI:
            line = pdb_line(atom, resid, base, 0, segname='DNAc')
            out.write(line)
            atom += 1
        for i in range(atom-2):
            line = f"CONECT{i+1:>5}{i+2:>5}\n"
            out.write(line)
        out.close()


def get_nitroxide_CNCO(helix):
    # get N and O positions of label on first strand
    label_idx = []
    label_CNCO = []
    for b,base in enumerate(helix.listI):
        if base.name == 'C_':
            label_idx.append(b)
            xyz = []
            for i,atom in enumerate(base.atoms):
                if atom in ['C15','N16','C17','O21']:
                    xyz.append(base.xyz[i])
            label_CNCO.append(xyz)
                    #break
    #now go over second strand
    for b,base in enumerate(helix.listII):
        if base.name == 'C_':
            label_idx.append(b)
            xyz = []
            for i,atom in enumerate(base.atoms):
                if atom in ['C15','N16','C17','O21']:
                    xyz.append(base.xyz[i])
            label_CNCO.append(xyz)

    return label_idx, np.array(label_CNCO)


def labels_dist(helix):
    label_idx, label_CNCO = get_nitroxide_CNCO(helix)
    n_labels = len(label_idx)
    dist_NN = []
    dist_OO = []
    dist_idx = []
    for i in range(1,n_labels):
        dist_idx.append( label_idx[i]-label_idx[0])
        dist_NN.append( np.linalg.norm(label_CNCO[i,1]-label_CNCO[0,1]) )
        dist_OO.append( np.linalg.norm(label_CNCO[i,3]-label_CNCO[0,3]) )
    return dist_NN, dist_OO, dist_idx


def labels_orientation(helix):
    label_idx, label_CNCO = get_nitroxide_CNCO(helix)
    n_labels = len(label_idx)
    rotations = []
    dist_idx = []

    for i in range(1,n_labels):
        R, t = rigid_transform_3D( label_CNCO[i].T, label_CNCO[0].T )
        rotations.append( R )
        dist_idx.append( label_idx[i]-label_idx[0])
    return rotations, dist_idx




