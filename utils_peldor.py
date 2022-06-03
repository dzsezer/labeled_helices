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


def get_nitroxide_NO(helix):
    # get N and O positions of label on first strand
    label_idx = []
    label_NO = []
    for b,pair in enumerate(helix.pairs):
        for base in pair.bases:
            if base.name == 'C_':
                label_idx.append(b)
                xyz = []
                for i,atom in enumerate(base.atoms):
                    if atom in ['N16','O21']:
                        xyz.append(base.xyz[i])
                label_NO.append(xyz)

    return label_idx, np.array(label_NO)


def labels_dist(helix):
    label_idx, label_NO = get_nitroxide_NO(helix)
    n_labels = len(label_idx)
    dist_NN = []
    dist_OO = []
    dist_NO = []
    dist_ON = []
    dist_idx = []
    for i in range(1,n_labels):
        dist_idx.append( label_idx[i]-label_idx[0])
        dist_NN.append( np.linalg.norm(label_NO[i,0]-label_NO[0,0]) )
        dist_OO.append( np.linalg.norm(label_NO[i,1]-label_NO[0,1]) )
        dist_NO.append( np.linalg.norm(label_NO[i,0]-label_NO[0,1]) )
        dist_ON.append( np.linalg.norm(label_NO[i,1]-label_NO[0,0]) )
    return dist_NN, dist_OO, dist_NO, dist_ON, dist_idx


def get_nitroxide_CNCO(helix):
    # get N and O positions of label on first strand
    label_idx = []
    label_CNCO = []
    for b,pair in enumerate(helix.pairs):
        for base in pair.bases:
            if base.name == 'C_':
                label_idx.append(b)
                xyz = []
                for i,atom in enumerate(base.atoms):
                    if atom in ['C15','N16','C17','O21']:
                        xyz.append(base.xyz[i])
                label_CNCO.append(xyz)

    return label_idx, np.array(label_CNCO)

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




