import numpy as np
from utils_peldor import rigid_transform_3D, labels_dist
from utils import pdb_line
import copy
#import os


def nodes2cos(pair1, point1, pair2, point2, xyz_nodes):
    dx, dy, dz = xyz_nodes[pair2][point2] - xyz_nodes[pair1][point1]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    norm = 1.0/dist
    return dx*norm, dy*norm, dz*norm


def read_nodes(inp_file):

    """
    Inp: 
        atom:       name of atom to serve as a node
        inp_file:   name of pdb file containing the molecular structure
        out_file:   name of pdb file to store the nodes

    Out:
        xyz_nodes:  a dictionary of the xyz coordinates of the nodes
                    key: CA #, value: list of [x,y,z]
    """

    # go over the input pdb file and collect the node atoms

    inp = open(inp_file + '.pdb')
    xyz_nodes = []
    for line in inp:
        if (line[0:4] == 'ATOM'):
            xyz_nodes.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
    inp.close()
    return np.array(xyz_nodes)


def make_links_helix(n_pairs, n_points, pdb):

    links = {} # [node1, node2, spring constant]

    #1) same pair
    for i in range(n_pairs):
        links[len(links)] = [(i,0), (i,1), 0.1]
        links[len(links)] = [(i,0), (i,2), 0.1]
        links[len(links)] = [(i,1), (i,2), 1.0]
    
    #2) next pair
    for i in range(n_pairs-1):
        j = i+1
        # backbone
        links[len(links)] = [(i,1), (j,1), 1.]
        links[len(links)] = [(i,2), (j,2), 1.]
        links[len(links)] = [(i,1), (j,2), 1.]
        links[len(links)] = [(i,2), (j,1), 1.]
        # center   
        links[len(links)] = [(i,0), (j,1), 0.1]
        links[len(links)] = [(i,0), (j,2), 0.1]
        links[len(links)] = [(i,0), (j,0), 0.1]
        links[len(links)] = [(i,1), (j,0), 0.1]
        links[len(links)] = [(i,2), (j,0), 0.1]
        #3) two pairs down
    for i in range(n_pairs-2):
        j = i+2
        # backbone only
        #links[len(links)] = [n1+1,n1+7, 1.]
        #links[len(links)] = [n1+2,n1+8, 1.]
        #links[len(links)] = [n1+2,n1+7, 1.]
        #links[len(links)] = [n1+1,n1+8, 1.]    

    # write links to the end of nodes.pdb
    if pdb:
        out = open('pdbs/' + pdb + '_nodes.pdb','a')
        for lk in links:
            #print(links[lk][0]+1)
            i = n_points*links[lk][0][0] + links[lk][0][1]
            j = n_points*links[lk][1][0] + links[lk][1][1]
            out.write('CONECT %4i %4i\n' %(i+1,j+1))
        out.close()
    
    return links


def  make_ANM_matrix(links, xyz_nodes):
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


def ANM_analysis(helix, n_low=3, pdb=None, verbose=True):

    xyz_nodes = helix.nodes
    n_pairs, n_points, _ = xyz_nodes.shape

    if pdb:
        helix.write_nodes_pdb(pdb)
    links = make_links_helix(n_pairs, n_points, pdb)
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
        print(f"next {n_low} eigenvalues: (should be real)")
        print(eVals[6:6+n_low])
    
    skip = 6 #these are the 6 translation and rotation modes
    #skip = 0
    eVals_low = eVals[skip:skip+n_low]
    eVecs_low = eVecs[:,skip:skip+n_low]

    # take real parts if all imaginary parts are zero
    if not np.any(np.iscomplex(eVals_low)):
        eVals_low = eVals_low.real
    if not np.any(np.iscomplex(eVecs_low)):
        eVecs_low = eVecs_low.real

    if verbose:
        np.set_printoptions(precision=6,suppress=True)

        print("  eVals       ratio    amplitude of oscillation")
        for ev in eVals_low:
            print(f"{ev:10.6f} {ev/eVals_low[0]:7.2f} {np.sqrt(eVals_low[0])/np.sqrt(ev):9.3f}")

        print("\ncheck orthogonality of the eigenvectors that belong to above eigenvalues:")
        print(eVecs_low.T @ eVecs_low)
    #print(f"reshape lowest {n_low} eigenvectors from {eVecs_low.shape}")
    eVecs_tmp = eVecs_low.T.reshape(n_low,3,n_pairs,-1)
    eVecs_low = np.transpose(eVecs_tmp,(0,2,3,1))
    #print(f"to {eVecs_low.shape}")

    form = lambda x: "%8.3f" % x
    scale = 2.
   
    for l in range(n_low):

        """
        scale = 0.2
        if (abs(ev) > 1e-6):
            scale = 0.02/np.sqrt(ev)
        #find maximum
        id_max = np.argmax(np.abs(eVecs_low[:,l]))
        max_sign = np.sign(eVecs_low[:,l][id_max])

        vx = eVecs_low[0*n_nodes:1*n_nodes,l]*max_sign
        vy = eVecs_low[1*n_nodes:2*n_nodes,l]*max_sign
        vz = eVecs_low[2*n_nodes:3*n_nodes,l]*max_sign

        
        #jv = Bx.T@vx + By.T@vy  + Bz.T@vz
        #maxjv = max(abs(jv))
        ##jv = abs(jv)/maxjv

        #if l == 0:
        #for nl in range(n_links):
        #    #if (jv[nl] > 0.99):    print(l,":",links[nl],jv[nl])
        #    if (jv[nl] < 0.001):    print(l,":",links[nl],jv[nl])
        """
        
        '''
        # write to anm.pdb file
        if pdb:
            out = open('pdbs/anm'+str(l+1)+'.pdb','w')
            # create 21 models
            models = 11
            low = (models-1)//2
            for md in range(models):
                #if md > 0:  out.write("ENDMDL\n")
                out.write("MODEL" + f"{str(md+1):>9s}\n")

                #inp = open('gnm.pdb')
                inp = open('pdbs/nodes.pdb')
                for line in inp:
                    if (line[0:4] == 'ATOM'):
                        words = line.split()
                        pair = int(words[4])-1
                        num = int(words[1])-1
                        point = num - (n_points*pair)
                        x_ref,y_ref,z_ref = float(line[30:38]),float(line[38:46]),float(line[46:54])
                        x = x_ref + (md-low)*scale*eVecs_low[l][pair][point][0]
                        y = y_ref + (md-low)*scale*eVecs_low[l][pair][point][1]
                        z = z_ref + (md-low)*scale*eVecs_low[l][pair][point][2]
                        new = line[:30] + form(x) + form(y) + form(z) + line[54:]
                        out.write(new)
                    else:
                        out.write(line)
                out.write("END\n")
                inp.close()
            out.close()
            '''
        
    return eVals_low, eVecs_low


def write_pdb_modes(file, helix, eVecs, eVals, n_models=11, scale = 10.):
    
    n_modes, n_pairs, n_points, _ = eVecs.shape
    s = np.linspace(-1,1,n_models)

    NN_all = []
    OO_all = []
    NO_all = []
    ON_all = []

    #go over modes
    for l in range(n_modes):
        ampl = 1./np.sqrt(eVals[l])

        NN_model = []
        OO_model = []
        NO_model = []
        ON_model = []

        vec_mode = eVecs[l]
        #print(vec_mode.shape)
        out = open('pdbs/' + file + '_mode' + str(l+1)+'.pdb','w')

        # create models
        for md in range(n_models):


            out.write("MODEL" + f"{str(md+1):>9s}\n")

            helix_copy = copy.deepcopy(helix)
            #go over pairs in helix
            for i,pair in enumerate(helix.pairs):
                A = helix.nodes[i]
                #print(A.shape)
                B = A + ampl*scale*s[md]*vec_mode[i]
                #print(B.shape)
                R, t = rigid_transform_3D(A.T, B.T)
                #print(i,R,t)
                #transform coordinates of pair in the  copy
                helix_copy.pairs[i].bases[0].xyz = helix.pairs[i].bases[0].xyz@R.T + t.T
                helix_copy.pairs[i].bases[1].xyz = helix.pairs[i].bases[1].xyz@R.T + t.T
            
            NN, OO, NO, ON, idx = labels_dist(helix_copy)
            NN_model.append(NN)
            OO_model.append(OO)
            NO_model.append(NO)
            ON_model.append(ON)

            #write to pdb file
            atom, resid = 1, 1
            #go over strand I
            for pair in helix_copy.pairs:
                base = pair.bases[0]
                for i in range(2,len(base.atoms)):
                    line = pdb_line(atom, resid, base, i, segname='DNA1')
                    out.write(line)
                    atom += 1
                resid += 1
            resid = 1
            #go over strand II
            for pair in reversed(helix_copy.pairs):
                base = pair.bases[1]
                for i in range(2,len(base.atoms)):
                    line = pdb_line(atom, resid, base, i, segname='DNA2')
                    out.write(line)
                    atom += 1
                resid += 1
            out.write("END\n")

        NN_all.append(NN_model)
        OO_all.append(OO_model)
        NO_all.append(NO_model)
        ON_all.append(ON_model)

        out.close()

    return np.array(NN_all), np.array(OO_all), np.array(NO_all), np.array(ON_all), np.array(idx)
    


###################
# MAIN 
#######################
'''
pdb_nodes = 'pdbs/nodes'
pdb_gnm   = 'pdbs/gnm'
pdb_anm   = 'pdbs/anm'


xyz_nodes = read_nodes('pdbs/3points')

os.system("cp pdbs/3points.pdb pdbs/nodes.pdb");
#print("xyz:",xyz_nodes)

ANM_analysis(xyz_nodes, n_low=6, pdb=True)

'''