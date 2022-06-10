import param_bases as parbase
import param_pairs as parpair

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



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


def distances_evn_odd(helix_evn, helix_odd):
    evn_NN, evn_OO, evn_NO, evn_ON, evn_id = labels_dist(helix_evn)
    odd_NN, odd_OO, odd_NO, odd_ON, odd_id = labels_dist(helix_odd)
    
    NN = np.concatenate([evn_NN,odd_NN],axis=-1)
    OO = np.concatenate([evn_OO,odd_OO],axis=-1)
    NO = np.concatenate([evn_NO,odd_NO],axis=-1)
    ON = np.concatenate([evn_ON,odd_ON],axis=-1)
    idx = np.concatenate([evn_id,odd_id],axis=-1)
    
    return NN,OO,NO,ON,idx


def plot_distances(helix_evn, helix_odd, exp_data, plot_title, error_bars=True):
    
    ax = plt.figure().gca()


    #the experimental data
    exp_id,exp_dist,exp_Dr = exp_data

    if error_bars:
        plt.errorbar(exp_id,exp_dist,exp_Dr,label="PELDOR",
            linestyle='',marker='*',ms=10,color='k', lw=1, capsize=2, capthick=2)
    else:
        plt.plot(exp_id,exp_dist,'--*k',markersize=10,label="PELDOR")
    
    #the model data
    NN,OO,NO,ON,idx = distances_evn_odd(helix_evn,helix_odd)

    plt.plot(idx,NN,'bo',ms=8,label='N-N')
    plt.plot(idx,OO,'ro',ms=8,label='O-O')
    plt.plot(idx,NO,'co',ms=8,label='N-O')
    plt.plot(idx,ON,'go',ms=8,label='O-N')

    #plt.plot(exp_id_dna_2,exp_dist_dna_2,'^k', label='C-dot')
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


    plt.xlabel("Base pair separation ($\Delta$)",fontsize=14)
    plt.ylabel("Distance [A]",fontsize=14)
    plt.title(plot_title)
    #plt.ylim(15,55)
    plt.legend(frameon=False,loc=2);

##################
#


"""

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

"""


