import numpy as np
from utils_peldor import labels_dist
from utils import pdb_line
import copy
#import os

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


def nodes2cos(pair1, point1, pair2, point2, xyz_nodes):
    dx, dy, dz = xyz_nodes[pair2][point2] - xyz_nodes[pair1][point1]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    norm = 1.0/dist
    return dx*norm, dy*norm, dz*norm


def make_links_helix(n_pairs, n_points):

    links = {} # [node1, node2, spring constant]

    #1) same pair
    for i in range(n_pairs):
        #backbone
        links[len(links)] = [(i,0), (i,1), 1.0]
        #center
        links[len(links)] = [(i,0), (i,2), 0.1]
        links[len(links)] = [(i,1), (i,2), 0.1]
        
    
    #2) next pair
    for i in range(n_pairs-1):
        j = i+1
        # backbone
        links[len(links)] = [(i,0), (j,0), 1.]
        links[len(links)] = [(i,1), (j,1), 1.]
        links[len(links)] = [(i,0), (j,1), 1.]
        links[len(links)] = [(i,1), (j,0), 1.]
        # center   
        #links[len(links)] = [(i,2), (j,2), 0.1]
        links[len(links)] = [(i,0), (j,2), 0.1]
        links[len(links)] = [(i,1), (j,2), 0.1]
        links[len(links)] = [(i,2), (j,0), 0.1]
        links[len(links)] = [(i,2), (j,1), 0.1]
        #3) two pairs down
    for i in range(n_pairs-2):
        j = i+2
        # backbone only
        # backbone
        links[len(links)] = [(i,0), (j,0), 1.]
        links[len(links)] = [(i,1), (j,1), 1.]
        #links[len(links)] = [(i,0), (j,1), 1.]
        #links[len(links)] = [(i,1), (j,0), 1.] 
    #4) three pairs down
    #for i in range(n_pairs-3):
    #    j = i+3
        # backbone
        #links[len(links)] = [(i,0), (j,0), 1.]
        #links[len(links)] = [(i,1), (j,1), 1.]  
    
    return links


def make_ANM_matrix(links, xyz_nodes):
    n_links = len(links)
    print("number of links:",n_links)

    n_pairs, n_points, _ = xyz_nodes.shape
    n_nodes = n_pairs*n_points
    print("number of nodes:",n_nodes)

    # construct direction cosine matrices
    Bx = np.zeros((n_nodes,n_links))
    By = np.zeros((n_nodes,n_links))
    Bz = np.zeros((n_nodes,n_links))
    # spring constants
    K = np.zeros(n_links)

    for lk in links:
        K[lk] = links[lk][2]
        pair1, point1 = links[lk][0]
        pair2, point2 = links[lk][1]
        cosX, cosY, cosZ = nodes2cos(pair1, point1, pair2, point2, xyz_nodes)
        i = n_points*pair1 + point1
        j = n_points*pair2 + point2
        Bx[i][lk] = cosX
        Bx[j][lk] = -cosX
        By[i][lk] = cosY
        By[j][lk] = -cosY
        Bz[i][lk] = cosZ
        Bz[j][lk] = -cosZ

    # calclate ANM matrix
    return np.vstack((
        np.hstack(((Bx*K)@Bx.T, (Bx*K)@By.T, (Bx*K)@Bz.T)),
        np.hstack(((By*K)@Bx.T, (By*K)@By.T, (By*K)@Bz.T)),
        np.hstack(((Bz*K)@Bx.T, (Bz*K)@By.T, (Bz*K)@Bz.T))
    ))


def ANM_analysis(helix, n_low=3, verbose=True):

    xyz_nodes = helix.nodes
    n_pairs, n_points, _ = xyz_nodes.shape

    links = make_links_helix(n_pairs, n_points)
    A     = make_ANM_matrix(links, xyz_nodes)

    rank = np.linalg.matrix_rank(A)

    eVals, eVecs = np.linalg.eig(A)

    idx = eVals.argsort()
    eVals = eVals[idx]
    eVecs = eVecs[:,idx]

    if verbose:
        print(f"rank of the {A.shape} matrix: {rank}")

        print("\nlowest 6 eigenvalues: (should be zeros)")
        print(eVals[:6])
        print(f"\nnext {n_low} eigenvalues: (should be real)")
        print(eVals[6:6+n_low])
    
    #restrict to lowest, non-zero frequency modes
    skip = 6 #these are the 6 translation and rotation modes
    eVals = eVals[skip:skip+n_low]
    eVecs = eVecs[:,skip:skip+n_low]

    # take real parts if all imaginary parts are zero
    if not np.any(np.iscomplex(eVals)):
        eVals = eVals.real
    if not np.any(np.iscomplex(eVecs)):
        eVecs = eVecs.real

    if verbose:
        np.set_printoptions(precision=6,suppress=True)

        print("  eVals       ratio    amplitude of oscillation")
        for ev in eVals:
            print(f"{ev:10.6f} {ev/eVals[0]:7.2f} {np.sqrt(eVals[0])/np.sqrt(ev):9.3f}")

        print("\ncheck orthogonality of the eigenvectors that belong to above eigenvalues:")
        print(eVecs.T @ eVecs)
    
    print(f"reshape lowest {n_low} eigenvectors from {eVecs.shape}")
    eVecs_tmp = eVecs.T.reshape(n_low,3,n_pairs,n_points)
    print(f"to {eVecs_tmp.shape}")

    eVecs = np.transpose(eVecs_tmp,(0,2,3,1))
    print(f"and to {eVecs.shape}")
    #print(xyz_nodes.shape)

    return eVals, eVecs, links


def write_pdb_nodes(helix, links, eVecs, eVals, pdb, n_models=11, scale = 0.01):

    n_modes, n_pairs, n_points, _ = eVecs.shape

    helix.write_nodes_pdb(pdb)
    # write links to the end of _nodes.pdb
    out = open('pdbs/' + pdb + '_nodes.pdb','a')
    for lk in links:
        #print(links[lk][0]+1)
        i = n_points*links[lk][0][0] + links[lk][0][1]
        j = n_points*links[lk][1][0] + links[lk][1][1]
        out.write('CONECT %4i %4i\n' %(i+1,j+1))
    out.close()
    
    s = np.linspace(-1,1,n_models)

    amplitudes = 1./np.sqrt(eVals)

    #go over modes
    for l in range(n_modes):
           
        s = np.linspace(-1,1,n_models)

        out = open('pdbs/' + pdb + '_nodes' + str(l+1)+'.pdb','w')

        # create models
        for md in range(n_models):
            
            out.write("MODEL" + f"{str(md+1):>9s}\n")

            helix_copy = copy.deepcopy(helix)
            #go over pairs in helix
            for i,pair in enumerate(helix.pairs):
                A = helix.nodes[i]
                B = A + amplitudes[l]*scale*s[md]*eVecs[l][i]
                helix_copy.nodes[i] = B
                R, t = rigid_transform_3D(A.T, B.T)
                #print(i,R,t)
                #transform coordinates of pair in the  copy
                helix_copy.pairs[i].bases[0].xyz = helix.pairs[i].bases[0].xyz@R.T + t.T
                helix_copy.pairs[i].bases[1].xyz = helix.pairs[i].bases[1].xyz@R.T + t.T
            
            #write to pdb file
            types = ['X','X','S']
            atom, resid = 1, 1
            #go over strand I
            for j in range(n_pairs):
                name = helix_copy.pairs[j].bases[0].name
                for i in range(n_points):
                    line = f"ATOM{atom:>7} {types[i]:^4} {name:>3}   {resid:>3} " \
                    + f"{helix_copy.nodes[j][i][0]:11.3f}{helix_copy.nodes[j][i][1]:8.3f}{helix_copy.nodes[j][i][2]:8.3f}" \
                    + f"  1.00  1.00      {'DNAx':>4}\n"
                    out.write(line)
                    atom += 1
                resid += 1
            for lk in links:
                #print(links[lk][0]+1)
                i = n_points*links[lk][0][0] + links[lk][0][1]
                j = n_points*links[lk][1][0] + links[lk][1][1]
                out.write('CONECT %4i %4i\n' %(i+1,j+1))
            out.write("END\n")



def write_pdb_modes(helix, eVecs, eVals, pdb, n_models=11, scale = 0.01):
    
    n_modes, n_pairs, n_points, _ = eVecs.shape
    s = np.linspace(-1,1,n_models)

    amplitudes = 1./np.sqrt(eVals)
    #go over modes
    for l in range(n_modes):
       
        out = open('pdbs/' + pdb + '_mode' + str(l+1)+'.pdb','w')

        # create models
        for md in range(n_models):
            
            if pdb:
                out.write("MODEL" + f"{str(md+1):>9s}\n")

            helix_copy = copy.deepcopy(helix)
            #go over pairs in helix
            for i,pair in enumerate(helix.pairs):
                A = helix.nodes[i]
                #print(A.shape)
                B = A + amplitudes[l]*scale*s[md]*eVecs[l][i]
                #print(B.shape)
                R, t = rigid_transform_3D(A.T, B.T)
                #print(i,R,t)
                #transform coordinates of pair in the  copy
                helix_copy.pairs[i].bases[0].xyz = helix.pairs[i].bases[0].xyz@R.T + t.T
                helix_copy.pairs[i].bases[1].xyz = helix.pairs[i].bases[1].xyz@R.T + t.T
            
            #NN, OO, NO, ON, idx = labels_dist(helix_copy)
            #NN_model.append(NN)
            #OO_model.append(OO)
            #NO_model.append(NO)
            #ON_model.append(ON)

            #write to pdb file
            if pdb:
                atom, resid = 1, 1
                #go over strand I
                for pair in helix_copy.pairs:
                    base = pair.bases[0]
                    for i in range(0,len(base.atoms)):
                        line = pdb_line(atom, resid, base, i, segname='DNA1')
                        out.write(line)
                        atom += 1
                    resid += 1
                resid = 1
                #go over strand II
                for pair in reversed(helix_copy.pairs):
                    base = pair.bases[1]
                    for i in range(0,len(base.atoms)):
                        line = pdb_line(atom, resid, base, i, segname='DNA2')
                        out.write(line)
                        atom += 1
                    resid += 1
                out.write("END\n")

        if pdb:
            out.close()

        #NN_all.append(NN_model)
        #OO_all.append(OO_model)
        #NO_all.append(NO_model)
        #ON_all.append(ON_model)

    #return np.array(NN_all), np.array(OO_all), np.array(NO_all), np.array(ON_all), np.array(idx)
    

def generate_ensemble(helix, eVecs, eVals, mask, n_samples=10, scale = 0.01):

    n_modes, n_pairs, n_points, _ = eVecs.shape

    amplitudes = scale/np.sqrt(eVals) * np.random.normal(size=(n_samples,n_modes))
    masked_ampl = np.einsum('ij,j->ij',amplitudes, np.array(mask))
    #print(masked_ampl)

    NN_all = []
    OO_all = []
    NO_all = []
    ON_all = []

    # create models
    for sample in range(n_samples):

        eVecs_scaled = np.einsum('i,ijkl->jkl',masked_ampl[sample],eVecs)
        #print(eVecs_scaled.shape)

        helix_copy = copy.deepcopy(helix)
        #go over pairs in helix
        for i,pair in enumerate(helix.pairs):
            A = helix.nodes[i]
            #print(A.shape)
            B = A + eVecs_scaled[i]
            #print(B.shape)
            R, t = rigid_transform_3D(A.T, B.T)
            #print(i,R,t)
            #transform coordinates of pair in the  copy
            helix_copy.pairs[i].bases[0].xyz = helix.pairs[i].bases[0].xyz@R.T + t.T
            helix_copy.pairs[i].bases[1].xyz = helix.pairs[i].bases[1].xyz@R.T + t.T

        
        NN, OO, NO, ON, idx = labels_dist(helix_copy)

        NN_all.append(NN)
        OO_all.append(OO)
        NO_all.append(NO)
        ON_all.append(ON)

    return np.array(NN_all), np.array(OO_all), np.array(NO_all), np.array(ON_all), np.array(idx)

##################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def plot_histograms(dist_all, idx_all, exp_data, xlim, ylim, show=False):

    #the experimental data
    exp_id,exp_dist,exp_Dr = exp_data

    df_dist = pd.DataFrame(
        data = dist_all,
        columns = idx_all
    )
    #sort columns
    cols = df_dist.columns.tolist()
    cols = [int(x) for x in cols]
    cols.sort()
    df_dist = df_dist[cols]

    #fit normal distribution to columns of the dataframe
    mu_all = []
    std_all = []

    for col in df_dist: 
        mu, std = norm.fit(df_dist[col])
        mu_all.append(mu)
        std_all.append(std)

    #plot
    sns.set(style="white")

    f, ax = plt.subplots(1,1,figsize=(12,8))

    b = sns.histplot(data=df_dist, element="poly", binwidth=.33, kde=True, stat='density')
    b.set_xlabel("Distance [A]",fontsize=16)
    b.set_ylabel("Density",fontsize=16)

    x_min = exp_dist[0] - exp_Dr[0]
    x_max = exp_dist[-1] + exp_Dr[-1]

    x_plot = np.linspace(x_min,x_max,300)

    plt.xlim(xlim)
    plt.ylim(ylim)


    for i,xd in enumerate(exp_dist):
        
        if show:
            # Plot the PDF.
            #p = norm.pdf(x_plot, mu_all[i], std_all[i])
            #plt.plot(x_plot, 0.1*p, 'k', linewidth=1)
            
            sig = exp_Dr[i]/2.355
            y_gauss = np.exp(-((x_plot-xd)/sig)**2/2)/(sig*np.sqrt(2*np.pi))
            if i%2 == 0:
                plt.plot(x_plot,(0.2)*y_gauss,color='red',linestyle=':')
            else:
                plt.plot(x_plot,(0.2)*y_gauss,color='blue',linestyle=':')
            
        
        plt.axvline(x=xd, color='gray', linestyle='--')
