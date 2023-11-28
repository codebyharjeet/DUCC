import scipy
import vqe_methods
import pyscf_helper

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, molden, cc
from pyscf.cc import ccsd

import openfermion as of
from openfermion import *
from tVQE import *

import numpy as np


"""Provide the primary functions for DUCC."""

# Function to print the one-, two-, and three-body tensors
thresh = 1e-15
def tprint(tens):
	if(np.ndim(tens) == 0):
		print(tens) 
	elif(np.ndim(tens) == 1):
		for i in range(0,len(tens)):
			if(abs(tens[i]) > thresh):
				print("[%d] : %e"%(i,tens[i]))
	elif(np.ndim(tens) == 2):
		for i in range(0,tens.shape[0]):
			for j in range(0,tens.shape[1]):
				if(abs(tens[i,j]) > thresh):
					print("[%d,%d] : %e"%(i,j,tens[i,j]))
	elif(np.ndim(tens) == 4): 
		for i in range(0,tens.shape[0]):
			for j in range(0,tens.shape[1]):
				for k in range(0,tens.shape[2]):
					for l in range(0,tens.shape[3]):
						if(abs(tens[i,j,k,l]) > thresh):
							print("[%d,%d,%d,%d] : %e"%(i,j,k,l,tens[i,j,k,l]))
	elif(np.ndim(tens) == 6): 
		for i in range(0,tens.shape[0]):
			for j in range(0,tens.shape[1]):
				for k in range(0,tens.shape[2]):
					for l in range(0,tens.shape[3]):
						for m in range(0,tens.shape[4]):
							for n in range(0,tens.shape[5]):
								if(abs(tens[i,j,k,l,m,n]) > thresh):
									print("[%d,%d,%d,%d,%d,%d] : %e"%(i,j,k,l,m,n,tens[i,j,k,l,m,n]))
	else:
		print("TODO: implement a printing for a %dD tensor."%(np.ndim(tens)))
		exit()

comment = """
def make_f_1(one_elec, config_a, config_b):
    n_orb = self.n_orb
    f = cp.deepcopy(self.int_H)
    for p in range(0,n_orb):
        pa = 2*p
        pb = 2*p+1
        for q in range(0,n_orb):
            qa = 2*q
            qb = 2*q+1
            for i in config_a:
                f[pa,qa] += self.int_V[pa,qa,2*i,2*i] - self.int_V[pa,2*i,2*i,qa]
                f[pb,qb] += self.int_V[pb,qb,2*i,2*i]
            for j in config_b:
                f[pa,qa] += self.int_V[pa,qa,2*j+1,2*j+1]
                f[pb,qb] += self.int_V[pb,qb,2*j+1,2*j+1] - self.int_V[pb,2*j+1,2*j+1,qb]
    return f

def make_v(two_elec):
    #v^{pq}_{rs} = <pq||rs>
    n_orb = self.n_orb
    v = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
    for p in range(0,n_orb):
        pa = 2*p
        pb = 2*p+1
        for q in range(0,n_orb):
            qa = 2*q
            qb = 2*q+1
            for r in range(0,n_orb):
                ra = 2*r
                rb = 2*r+1
                for s in range(0,n_orb):
                    sa = 2*s
                    sb = 2*s+1
                    v[pa,qa,ra,sa] = self.int_V[pa,ra,qa,sa] - self.int_V[pa,sa,qa,ra]
                    v[pa,qb,ra,sb] = self.int_V[pa,ra,qb,sb]
                    v[pa,qb,rb,sa] = -self.int_V[pa,sa,qb,rb]
                    v[pb,qa,rb,sa] = self.int_V[pb,rb,qa,sa]
                    v[pb,qa,ra,sb] = -self.int_V[pb,sb,qa,ra]
                    v[pb,qb,rb,sb] = self.int_V[pb,rb,qb,sb] - self.int_V[pb,sb,qb,rb]
    return v
"""

def get_spin_to_spatial(integral,n_orb):
    if(np.ndim(integral) == 0):
        print("It is a constant term.")
        exit()
    elif(np.ndim(integral) == 2):
        integral_spatial = np.zeros((n_orb,n_orb))
        for p in range(0,n_orb):
            pa = 2*p
            for q in range(0,n_orb):
                qa = 2*q
                qb = 2*q + 1
                integral_spatial[p,q] = integral[pa,qa]
    elif(np.ndim(integral) == 4): 
        integral_spatial = np.zeros((n_orb,n_orb,n_orb,n_orb))
        for p in range(0,n_orb):
            pa = 2*p
            pb = 2*p+1
            for q in range(0,n_orb):
                qa = 2*q
                qb = 2*q+1
                for r in range(0,n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    for s in range(0,n_orb):
                        sa = 2*s
                        sb = 2*s+1
                        # vmat_spatial[p,q,r,s] = vmat[pa,qb,ra,sb]
                        integral_spatial[p,r,q,s] = integral[pa,qb,ra,sb]
                        #vmat_spatial[p,q,r,s] = vmat[pa, ra,qb,sb]
        
    return integral_spatial


def get_u(two_elec, config_a, config_b):
    """
    u_{pq} = \sum_{i} <pi||qi>
    """
    n_orb = int(two_elec.shape[0]/2)
    u = np.zeros((2*n_orb,2*n_orb))
    #two_elec = get_spin_to_spatial(two_elec,n_orb)
    print("Running get_u function and its shape = ",u.shape)
    for p in range(0,n_orb):
        pa = 2*p
        pb = 2*p+1
        for q in range(0,n_orb):
            qa = 2*q
            qb = 2*q+1
            for i in config_a:
                u[pa,qa] += two_elec[pa,2*i,qa,2*i]
                u[pb,qb] += two_elec[pb,2*i,qb,2*i]
            for j in config_b:
                u[pa,qa] += two_elec[pa,2*j+1,qa,2*j+1]
                u[pb,qb] += two_elec[pb,2*j+1,qb,2*j+1]

    return u


# Function to create one body matrix      
def make_full_one(big,small,n_a,n_b,n_orb):
    if n_a+n_b==small.shape[0]:
        ind_0 = slice(0,n_a+n_b)
    else:
        ind_0 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[1]:
        ind_1 = slice(0,n_a+n_b)
    else:
        ind_1 = slice(n_a+n_b,2*n_orb)
    big[ind_0,ind_1] = small
    return big


# Function to create two body matrix      
def make_full_two(big,small,n_a,n_b,n_orb):
    if n_a+n_b==small.shape[0]:
        ind_0 = slice(0,n_a+n_b)
    else:
        ind_0 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[1]:
        ind_1 = slice(0,n_a+n_b)
    else:
        ind_1 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[2]:
        ind_2 = slice(0,n_a+n_b)
    else:
        ind_2 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[3]:
        ind_3 = slice(0,n_a+n_b)
    else:
        ind_3 = slice(n_a+n_b,2*n_orb)
    big[ind_0,ind_1,ind_2,ind_3] = small
    return big


# Function to create three body matrix      
def make_full_three(big,small,n_a,n_b,n_orb):
    if n_a+n_b==small.shape[0]:
        ind_0 = slice(0,n_a+n_b)
    else:
        ind_0 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[1]:
        ind_1 = slice(0,n_a+n_b)
    else:
        ind_1 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[2]:
        ind_2 = slice(0,n_a+n_b)
    else:
        ind_2 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[3]:
        ind_3 = slice(0,n_a+n_b)
    else:
        ind_3 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[4]:
        ind_4 = slice(0,n_a+n_b)
    else:
        ind_4 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[5]:
        ind_5= slice(0,n_a+n_b)
    else:
        ind_5 = slice(n_a+n_b,2*n_orb)
    big[ind_0,ind_1,ind_2,ind_3,ind_4,ind_5] = small
    return big


# Function to get the amplitudes outside the active space
def get_t_ext(t1_amps,t2_amps,n_a,n_b,act_max):
    # getting t1_external_a               
    for i in range(0,n_a):
        for a in range(n_a,act_max):
            t1_amps[0][i,a-n_a]=0
    # getting t1_external_b               
    for i in range(0,n_b):
        for a in range(n_b,act_max):
            t1_amps[1][i,a-n_b]=0 
    #print(t1_amps)

    # getting t2_external_aa
    for i in range(0,n_a):
        for j in range(0,n_a):
            for a in range(n_a,act_max):
                for b in range(n_a,act_max):
                    t2_amps[0][i,j,a-n_a,b-n_a]=0

    # getting t2_external_ab
    for i in range(0,n_a):
        for j in range(0,n_b):
            for a in range(n_a,act_max):
                for b in range(n_b,act_max):
                    t2_amps[1][i,j,a-n_a,b-n_b]=0 
                    
    # getting t2_external_bb
    for i in range(0,n_b):
        for j in range(0,n_b):
            for a in range(n_b,act_max):
                for b in range(n_b,act_max):
                    t2_amps[2][i,j,a-n_b,b-n_b]=0 

    #print(t2_amps)
    return t1_amps, t2_amps


# Function to transform the amplitudes from spatial to spin space
def transform_t_spatial_to_spin(t1_amps, t2_amps, n_a, n_b, n_orb):
    # expanding the t1 amplitude into alpha and beta space
    t1 = np.zeros((n_a+n_b,2*n_orb-n_a-n_b))
    for i in range(0,n_a):
        for a in range(n_a,n_orb):
            ia = 2*i
            aa = 2*a
            t1[ia,aa-(n_a+n_b)] = t1_amps[0][i,a-n_a]
    for i in range(0,n_b):
        for a in range(n_b,n_orb):
            ib = 2*i+1
            ab = 2*a+1
            t1[ib,ab-(n_a+n_b)] = t1_amps[1][i,a-n_b]

    print(t1.shape)
    #print(t1)

    # expanding the t2 amplitude_aa
    t2 = np.zeros((n_a+n_b,n_a+n_b,2*n_orb-n_a-n_b,2*n_orb-n_a-n_b))
    for i in range(0,n_a):
        ia = 2*i
        for j in range(0,n_a):
            ja = 2*j
            for a in range(n_a,n_orb):
                aa = 2*a
                for b in range(n_a,n_orb):
                    ba = 2*b
                    t2[ia,ja,aa-(n_a+n_b),ba-(n_a+n_b)]=t2_amps[0][i,j,a-n_a,b-n_a]

                    
                    
    # expanding the t2 amplitude_ab
    for i in range(0,n_a):
        ia = 2*i
        for j in range(0,n_b):
            jb = 2*j+1
            for a in range(n_a,n_orb):
                aa = 2*a
                for b in range(n_b,n_orb):
                    bb = 2*b+1 
                    t2[ia,jb,aa-(n_a+n_b),bb-(n_a+n_b)]=t2_amps[1][i,j,a-n_a,b-n_b]
                    t2[jb,ia,bb-(n_a+n_b),aa-(n_a+n_b)]=t2_amps[1][i,j,a-n_a,b-n_b]
                    t2[ia,jb,bb-(n_a+n_b),aa-(n_a+n_b)]=-t2_amps[1][i,j,a-n_a,b-n_b]
                    t2[jb,ia,aa-(n_a+n_b),bb-(n_a+n_b)]=-t2_amps[1][i,j,a-n_a,b-n_b]
                    
    # expanding the t2 amplitude_bb
    for i in range(0,n_b):
        ib = 2*i+1
        for j in range(0,n_b):
            jb = 2*j+1
            for a in range(n_b,n_orb):
                ab = 2*a+1
                for b in range(n_b,n_orb):
                    bb = 2*b+1 
                    t2[ib,jb,ab-(n_a+n_b),bb-(n_a+n_b)]=t2_amps[2][i,j,a-n_b,b-n_b]

    print(t2.shape)
    return t1, t2


# Function to get the projected active space
def proj_tens_to_as(term,act_max):
    if(np.ndim(term) == 0):
        print("It is a constant term.")
        exit()
    elif(np.ndim(term) == 2):
        proj_term = term[0:2*act_max, 0:2*act_max]
    elif(np.ndim(term) == 4): 
        proj_term = term[0:2*act_max, 0:2*act_max, 0:2*act_max, 0:2*act_max]
    elif(np.ndim(term) == 6): 
        proj_term = term[0:2*act_max, 0:2*act_max, 0:2*act_max, 0:2*act_max, 0:2*act_max, 0:2*act_max]
    else:
	    print("TODO: implement a projection for a %dD tensor."%(np.ndim(term)))
	    exit()
    return proj_term


# Function to compute the contractions in the DUCC procedure
def compute_ducc(fmat,vmat,t1,t2,n_a,n_b,n_occ,n_orb,act_max,compute_three_body=False):
    """
    Parameters
    ----------
    fmat: One electron integral
    vmat: Two electron integral
    t1: Single excitation amplitude
    t2: Double excitation amplitude
    n_a: Number of alpha electrons
    n_b: Number of beta electrons
    n_occ: Number of alpha+beta electrons
    n_orb: Total number of spatial orbitals
    act_max: Total number of spatial orbitals in active space
    compute_three_body : bool, Optional, default: False

    Returns
    -------
    constant: Contant term
    one_body: Sum of all one body terms
    two_body: Sum of all two body terms
    three_body: Sum of all three body terms
    """

    # Initializing the constant, one-, two-, and three-body tensors
    constant = 0
    #one_body = np.zeros((2*act_max,2*act_max))
    #two_body = np.zeros((2*act_max,2*act_max,2*act_max,2*act_max))
    #three_body = np.zeros((2*act_max,2*act_max,2*act_max,2*act_max,2*act_max,2*act_max))
    one_body = np.zeros((2*n_orb,2*n_orb))
    two_body = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
    three_body = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb,2*n_orb,2*n_orb))

    # Extracting different blocks of F
    fmat_1 = np.array(fmat)

    fmat_oo = fmat_1[0:n_a+n_b , 0:n_a+n_b]
    #print(fmat_oo.shape)

    fmat_ov = fmat_1[0:n_a+n_b , n_a+n_b:2*n_orb]
    #print(fmat_ov.shape)

    fmat_vo = fmat_1[n_a+n_b:2*n_orb , 0:n_a+n_b]
    #print(fmat_vo.shape)

    fmat_vv = fmat_1[n_a+n_b:2*n_orb , n_a+n_b:2*n_orb]
    #print(fmat_vv.shape)

    #one_body += proj_tens_to_as(fmat_1,act_max)
    one_body += fmat_1

    # Extracting different blocks of V
    vmat_1 = np.array(vmat)
    #print(vmat_1.shape)

    vmat_oooo = vmat_1[0:n_a+n_b , 0:n_a+n_b , 0:n_a+n_b , 0:n_a+n_b]
    #print(vmat_oooo.shape)

    vmat_ooov = vmat_1[0:n_a+n_b , 0:n_a+n_b , 0:n_a+n_b , n_a+n_b:2*n_orb]
    #print(vmat_ooov.shape)

    vmat_ovoo = vmat_1[0:n_a+n_b , n_a+n_b:2*n_orb , 0:n_a+n_b , 0:n_a+n_b]
    #print(vmat_ovoo.shape)

    vmat_ovov = vmat_1[0:n_a+n_b , n_a+n_b:2*n_orb , 0:n_a+n_b , n_a+n_b:2*n_orb]
    #print(vmat_ovov.shape)

    vmat_oovv = vmat_1[0:n_a+n_b , 0:n_a+n_b , n_a+n_b:2*n_orb , n_a+n_b:2*n_orb]
    #print(vmat_oovv.shape)

    vmat_vvoo = vmat_1[n_a+n_b:2*n_orb , n_a+n_b:2*n_orb , 0:n_a+n_b , 0:n_a+n_b]
    #print(vmat_vvoo.shape)

    vmat_ovvv = vmat_1[0:n_a+n_b , n_a+n_b:2*n_orb , n_a+n_b:2*n_orb , n_a+n_b:2*n_orb]
    #print(vmat_ovvv.shape)

    vmat_vvov = vmat_1[n_a+n_b:2*n_orb , n_a+n_b:2*n_orb , 0:n_a+n_b , n_a+n_b:2*n_orb]
    #print(vmat_vvov.shape)

    vmat_vvvv = vmat_1[n_a+n_b:2*n_orb , n_a+n_b:2*n_orb , n_a+n_b:2*n_orb , n_a+n_b:2*n_orb]
    #print(vmat_vvvv.shape)

    #two_body += proj_tens_to_as(vmat_1,act_max)
    two_body += vmat_1

    # FT1
    # t_ov = t1 and t_vo = transpose(t1)

    # f_oo and t1
    ft1oo = 0
    ft1oo += 1.000000000 * np.einsum("ai,ja->ji",fmat_vo,t1,optimize="optimal")
    ft1oo += 1.000000000 * np.einsum("ia,aj->ij",fmat_ov,t1.transpose(),optimize="optimal")
    print("ft1oo = ",ft1oo.shape)

    # f_ov and t1
    ft1ov = 0
    ft1ov += -1.000000000 * np.einsum("ij,ja->ia",fmat_oo,t1,optimize="optimal")
    ft1ov += 1.000000000 * np.einsum("ba,ib->ia",fmat_vv,t1,optimize="optimal")
    print("ft1ov = ",ft1ov.shape)

    # f_vo and t1
    ft1vo = 0
    ft1vo += -1.000000000 * np.einsum("ji,aj->ai",fmat_oo,t1.transpose(),optimize="optimal")
    ft1vo += 1.000000000 * np.einsum("ab,bi->ai",fmat_vv,t1.transpose(),optimize="optimal")
    print("ft1vo = ",ft1vo.shape)

    # f_vv and t1
    ft1vv = 0
    ft1vv += -1.000000000 * np.einsum("ai,ib->ab",fmat_vo,t1,optimize="optimal")
    ft1vv += -1.000000000 * np.einsum("ia,bi->ba",fmat_ov,t1.transpose(),optimize="optimal")
    print("ft1vv = ",ft1vv.shape)

    # f and t1
    ft1 = 0
    ft1 += 1.000000000 * np.einsum("ai,ia->",fmat_vo,t1,optimize="optimal")
    ft1 += 1.000000000 * np.einsum("ia,ai->",fmat_ov,t1.transpose(),optimize="optimal")
    print("ft1 = ",ft1)

    ft1_mat = np.zeros((2*n_orb,2*n_orb))
    ft1_mat = make_full_one(ft1_mat,ft1oo,n_a,n_b,n_orb)
    ft1_mat = make_full_one(ft1_mat,ft1ov,n_a,n_b,n_orb)
    ft1_mat = make_full_one(ft1_mat,ft1vo,n_a,n_b,n_orb)
    ft1_mat = make_full_one(ft1_mat,ft1vv,n_a,n_b,n_orb)

    # Projecting into the active space
    #one_body += proj_tens_to_as(ft1_mat,act_max)
    one_body += ft1_mat
    constant += ft1

    # FT2
    # t_oovv = t2 and t_vvoo = transpose(t2)

    # Initializing the ft2_mat_1 one body
    ft2_mat_1 = np.zeros((2*n_orb,2*n_orb))
    print(ft2_mat_1.shape)

    # Initializing the ft2_mat_2 two body
    ft2_mat_2 = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
    print(ft2_mat_2.shape)

    ft2ooov = 0
    ft2ooov += -0.500000000 * np.einsum("bi,jkab->jkia",fmat_vo,t2,optimize="optimal")
    #print(ft2ooov.shape)
    ft2_mat_2 = make_full_two(ft2_mat_2,ft2ooov,n_a,n_b,n_orb)

    ft2oovv = 0
    ft2oovv += 0.500000000 * np.einsum("ik,jkab->ijab",fmat_oo,t2,optimize="optimal")
    ft2oovv += -0.500000000 * np.einsum("ca,ijbc->ijab",fmat_vv,t2,optimize="optimal")
    #print(ft2oovv.shape)
    ft2_mat_2 = make_full_two(ft2_mat_2,ft2oovv,n_a,n_b,n_orb)

    ft2ovoo = 0
    ft2ovoo += -0.500000000 * np.einsum("ib,abjk->iajk",fmat_ov,t2.transpose(),optimize="optimal")
    #print(ft2ovoo.shape)
    ft2_mat_2 = make_full_two(ft2_mat_2,ft2ovoo,n_a,n_b,n_orb)

    ft2ovvv = 0
    ft2ovvv += -0.500000000 * np.einsum("aj,ijbc->iabc",fmat_vo,t2,optimize="optimal")
    #print(ft2ovvv.shape)
    ft2_mat_2 = make_full_two(ft2_mat_2,ft2ovvv,n_a,n_b,n_orb)
           
    ft2vvoo = 0
    ft2vvoo += 0.500000000 * np.einsum("ki,abjk->abij",fmat_oo,t2.transpose(),optimize="optimal")
    ft2vvoo += -0.500000000 * np.einsum("ac,bcij->abij",fmat_vv,t2.transpose(),optimize="optimal")
    #print(ft2vvoo.shape)
    ft2_mat_2 = make_full_two(ft2_mat_2,ft2vvoo,n_a,n_b,n_orb)

    ft2vvov = 0
    ft2vvov += -0.500000000 * np.einsum("ja,bcij->bcia",fmat_ov,t2.transpose(),optimize="optimal")
    #print(ft2vvov.shape)
    ft2_mat_2 = make_full_two(ft2_mat_2,ft2vvov,n_a,n_b,n_orb)

    ft2ov = 0
    ft2ov += 1.000000000 * np.einsum("bj,ijab->ia",fmat_vo,t2,optimize="optimal")
    #print(ft2ov.shape)
    ft2_mat_1 = make_full_one(ft2_mat_1,ft2ov,n_a,n_b,n_orb)

    ft2vo = 0
    ft2vo += 1.000000000 * np.einsum("jb,abij->ai",fmat_ov,t2.transpose(),optimize="optimal")
    #print(ft2vo.shape)
    ft2_mat_1 = make_full_one(ft2_mat_1,ft2vo,n_a,n_b,n_orb)

    # Projecting into the active space
    #two_body += proj_tens_to_as(ft2_mat_2,act_max)
    #one_body += proj_tens_to_as(ft2_mat_1,act_max)
    two_body += ft2_mat_2
    one_body += ft2_mat_1


    # WT1
    # t_ov = t1 and t_vo = transpose(t1)

    # Initializing the wt1_mat_1 one body
    wt1_mat_1 = np.zeros((2*n_orb,2*n_orb))
    print("wt1_mat_1 = ",wt1_mat_1.shape)

    # Initializing the wt1_mat_2 two body
    wt1_mat_2 = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
    print("wt1_mat_2 = ",wt1_mat_2.shape)


    wt1oooo = 0
    wt1oooo += 0.500000000 * np.einsum("kaij,la->klij",vmat_ovoo,t1,optimize="optimal")
    wt1oooo += 0.500000000 * np.einsum("jkia,al->jkil",vmat_ooov,t1.transpose(),optimize="optimal")
    print("wt1oooo = ",wt1oooo.shape)
    wt1_mat_2 = make_full_two(wt1_mat_2,wt1oooo,n_a,n_b,n_orb)

    wt1ooov = 0
    wt1ooov += -0.500000000 * np.einsum("jkil,la->jkia",vmat_oooo,t1,optimize="optimal")
    wt1ooov += 1.000000000 * np.einsum("jbia,kb->jkia",vmat_ovov,t1,optimize="optimal")
    wt1ooov += -0.500000000 * np.einsum("ijab,bk->ijka",vmat_oovv,t1.transpose(),optimize="optimal")
    print("wt1ooov = ",wt1ooov.shape)
    wt1_mat_2 = make_full_two(wt1_mat_2,wt1ooov,n_a,n_b,n_orb)

    wt1oovv = 0
    wt1oovv += 0.500000000 * np.einsum("ijka,kb->ijab",vmat_ooov,t1,optimize="optimal")
    wt1oovv += 0.500000000 * np.einsum("icab,jc->ijab",vmat_ovvv,t1,optimize="optimal")
    print("wt1oovv = ",wt1oovv.shape)
    wt1_mat_2 = make_full_two(wt1_mat_2,wt1oovv,n_a,n_b,n_orb)

    wt1ovoo = 0
    wt1ovoo += -0.500000000 * np.einsum("klij,al->kaij",vmat_oooo,t1.transpose(),optimize="optimal")
    wt1ovoo += -0.500000000 * np.einsum("abij,kb->kaij",vmat_vvoo,t1,optimize="optimal")
    wt1ovoo += 1.000000000 * np.einsum("jaib,bk->jaik",vmat_ovov,t1.transpose(),optimize="optimal")
    print("wt1ovoo = ",wt1ovoo.shape)
    wt1_mat_2 = make_full_two(wt1_mat_2,wt1ovoo,n_a,n_b,n_orb)

    wt1ovov = 0
    wt1ovov += -1.000000000 * np.einsum("jaik,kb->jaib",vmat_ovoo,t1,optimize="optimal")
    wt1ovov += -1.000000000 * np.einsum("jkia,bk->jbia",vmat_ooov,t1.transpose(),optimize="optimal")
    wt1ovov += -1.000000000 * np.einsum("bcia,jc->jbia",vmat_vvov,t1,optimize="optimal")
    wt1ovov += -1.000000000 * np.einsum("ibac,cj->ibja",vmat_ovvv,t1.transpose(),optimize="optimal")
    print("wt1ovov = ",wt1ovov.shape)
    wt1_mat_2 = make_full_two(wt1_mat_2,wt1ovov,n_a,n_b,n_orb)

    wt1ovvv = 0
    wt1ovvv += 1.000000000 * np.einsum("ibja,jc->ibac",vmat_ovov,t1,optimize="optimal")
    wt1ovvv += -0.500000000 * np.einsum("ijab,cj->icab",vmat_oovv,t1.transpose(),optimize="optimal")
    wt1ovvv += -0.500000000 * np.einsum("cdab,id->icab",vmat_vvvv,t1,optimize="optimal")
    print("wt1ovvv = ",wt1ovvv.shape)
    wt1_mat_2 = make_full_two(wt1_mat_2,wt1ovvv,n_a,n_b,n_orb)

    wt1vvoo = 0
    wt1vvoo += 0.500000000 * np.einsum("kaij,bk->abij",vmat_ovoo,t1.transpose(),optimize="optimal")
    wt1vvoo += 0.500000000 * np.einsum("abic,cj->abij",vmat_vvov,t1.transpose(),optimize="optimal")
    print("wt1vvoo = ",wt1vvoo.shape)
    wt1_mat_2 = make_full_two(wt1_mat_2,wt1vvoo,n_a,n_b,n_orb)

    wt1vvov = 0
    wt1vvov += -0.500000000 * np.einsum("abij,jc->abic",vmat_vvoo,t1,optimize="optimal")
    wt1vvov += 1.000000000 * np.einsum("jbia,cj->bcia",vmat_ovov,t1.transpose(),optimize="optimal")
    wt1vvov += -0.500000000 * np.einsum("bcad,di->bcia",vmat_vvvv,t1.transpose(),optimize="optimal")
    print("wt1vvov = ",wt1vvov.shape)
    wt1_mat_2 = make_full_two(wt1_mat_2,wt1vvov,n_a,n_b,n_orb)

    wt1vvvv = 0
    wt1vvvv += 0.500000000 * np.einsum("bcia,id->bcad",vmat_vvov,t1,optimize="optimal")
    wt1vvvv += 0.500000000 * np.einsum("icab,di->cdab",vmat_ovvv,t1.transpose(),optimize="optimal")
    print("wt1vvvv = ",wt1vvvv.shape)
    wt1_mat_2 = make_full_two(wt1_mat_2,wt1vvvv,n_a,n_b,n_orb)

    wt1oo = 0
    wt1oo += 1.000000000 * np.einsum("jaik,ka->ji",vmat_ovoo,t1,optimize="optimal")
    wt1oo += 1.000000000 * np.einsum("jkia,ak->ji",vmat_ooov,t1.transpose(),optimize="optimal")
    print("wt1oo = ",wt1oo.shape)
    wt1_mat_1 = make_full_one(wt1_mat_1,wt1oo,n_a,n_b,n_orb)

    wt1ov = 0
    wt1ov += -1.000000000 * np.einsum("ibja,jb->ia",vmat_ovov,t1,optimize="optimal")
    wt1ov += 1.000000000 * np.einsum("ijab,bj->ia",vmat_oovv,t1.transpose(),optimize="optimal")
    print("wt1ov = ",wt1ov.shape)
    wt1_mat_1 = make_full_one(wt1_mat_1,wt1ov,n_a,n_b,n_orb)

    wt1vo = 0
    wt1vo += 1.000000000 * np.einsum("abij,jb->ai",vmat_vvoo,t1,optimize="optimal")
    wt1vo += -1.000000000 * np.einsum("jaib,bj->ai",vmat_ovov,t1.transpose(),optimize="optimal")
    print("wt1vo = ",wt1vo.shape)
    wt1_mat_1 = make_full_one(wt1_mat_1,wt1vo,n_a,n_b,n_orb)

    wt1vv = 0
    wt1vv += -1.000000000 * np.einsum("bcia,ic->ba",vmat_vvov,t1,optimize="optimal")
    wt1vv += -1.000000000 * np.einsum("ibac,ci->ba",vmat_ovvv,t1.transpose(),optimize="optimal")
    print("wt1vv = ",wt1vv.shape)
    wt1_mat_1 = make_full_one(wt1_mat_1,wt1vv,n_a,n_b,n_orb)

    # Projecting into the active space
    #two_body += proj_tens_to_as(wt1_mat_2,act_max)
    #one_body += proj_tens_to_as(wt1_mat_1,act_max)
    two_body += wt1_mat_2
    one_body += wt1_mat_1

    # WT2
    # t_oovv = t2 and t_vvoo = transpose(t2)

    # Initializing the wt2_mat_1 one body
    wt2_mat_1 = np.zeros((2*n_orb,2*n_orb))
    print("wt2_mat_1 = ",wt2_mat_1.shape)

    # Initializing the wt2_mat_2 two body
    wt2_mat_2 = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
    print("wt2_mat_2 = ",wt2_mat_2.shape)
    
    # Initializing the wt2_mat_3 three body
    wt2_mat_3 = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb,2*n_orb,2*n_orb))
    print("wt2_mat_3 = ",wt2_mat_3.shape)

    if compute_three_body==True:
        wt2ooooov = 0
        wt2ooooov += -0.250000000 * np.einsum("kbij,lmab->klmija",vmat_ovoo,t2,optimize="optimal")
        print("wt2ooooov = ",wt2ooooov.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2ooooov,n_a,n_b,n_orb)

        wt2oooovv = 0
        wt2oooovv += 0.250000000 * np.einsum("jkim,lmab->jkliab",vmat_oooo,t2,optimize="optimal")
        wt2oooovv += -0.500000000 * np.einsum("jcia,klbc->jkliab",vmat_ovov,t2,optimize="optimal")
        print("wt2oooovv = ",wt2oooovv.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2oooovv,n_a,n_b,n_orb)

        wt2ooovvv = 0
        wt2ooovvv += -0.250000000 * np.einsum("ijla,klbc->ijkabc",vmat_ooov,t2,optimize="optimal")
        wt2ooovvv += -0.250000000 * np.einsum("idab,jkcd->ijkabc",vmat_ovvv,t2,optimize="optimal")
        print("wt2ooovvv = ",wt2ooovvv.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2ooovvv,n_a,n_b,n_orb)

        wt2oovooo = 0
        wt2oovooo += -0.250000000 * np.einsum("jkib,ablm->jkailm",vmat_ooov,t2.transpose(),optimize="optimal")
        print("wt2oovooo = ",wt2oovooo.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2oovooo,n_a,n_b,n_orb)

        wt2oovoov = 0
        wt2oovoov += -0.250000000 * np.einsum("acij,klbc->klaijb",vmat_vvoo,t2,optimize="optimal")
        wt2oovoov += -0.250000000 * np.einsum("ijac,bckl->ijbkla",vmat_oovv,t2.transpose(),optimize="optimal")
        print("wt2oovoov = ",wt2oovoov.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2oovoov,n_a,n_b,n_orb)

        wt2oovovv = 0
        wt2oovovv += -0.500000000 * np.einsum("jail,klbc->jkaibc",vmat_ovoo,t2,optimize="optimal")
        wt2oovovv += -0.500000000 * np.einsum("bdia,jkcd->jkbiac",vmat_vvov,t2,optimize="optimal")
        print("wt2oovovv = ",wt2oovovv.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2oovovv,n_a,n_b,n_orb)

        wt2oovvvv = 0
        wt2oovvvv += 0.500000000 * np.einsum("ibka,jkcd->ijbacd",vmat_ovov,t2,optimize="optimal")
        wt2oovvvv += -0.250000000 * np.einsum("ceab,ijde->ijcabd",vmat_vvvv,t2,optimize="optimal")
        print("wt2oovvvv = ",wt2oovvvv.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2oovvvv,n_a,n_b,n_orb)

        wt2ovvooo = 0
        wt2ovvooo += 0.250000000 * np.einsum("kmij,ablm->kabijl",vmat_oooo,t2.transpose(),optimize="optimal")
        wt2ovvooo += -0.500000000 * np.einsum("jaic,bckl->jabikl",vmat_ovov,t2.transpose(),optimize="optimal")
        print("wt2ovvooo = ",wt2ovvooo.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2ovvooo,n_a,n_b,n_orb)

        wt2ovvoov = 0
        wt2ovvoov += -0.500000000 * np.einsum("jlia,bckl->jbcika",vmat_ooov,t2.transpose(),optimize="optimal")
        wt2ovvoov += -0.500000000 * np.einsum("ibad,cdjk->ibcjka",vmat_ovvv,t2.transpose(),optimize="optimal")
        print("wt2ovvoov = ",wt2ovvoov.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2ovvoov,n_a,n_b,n_orb)

        wt2ovvovv = 0
        wt2ovvovv += 0.250000000 * np.einsum("abik,jkcd->jabicd",vmat_vvoo,t2,optimize="optimal")
        wt2ovvovv += 0.250000000 * np.einsum("ikab,cdjk->icdjab",vmat_oovv,t2.transpose(),optimize="optimal")
        print("wt2ovvovv = ",wt2ovvovv.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2ovvovv,n_a,n_b,n_orb)

        wt2ovvvvv = 0
        wt2ovvvvv += -0.250000000 * np.einsum("bcja,ijde->ibcade",vmat_vvov,t2,optimize="optimal")
        print("wt2ovvvvv = ",wt2ovvvvv.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2ovvvvv,n_a,n_b,n_orb)

        wt2vvvooo = 0
        wt2vvvooo += -0.250000000 * np.einsum("laij,bckl->abcijk",vmat_ovoo,t2.transpose(),optimize="optimal")
        wt2vvvooo += -0.250000000 * np.einsum("abid,cdjk->abcijk",vmat_vvov,t2.transpose(),optimize="optimal")
        print("wt2vvvooo = ",wt2vvvooo.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2vvvooo,n_a,n_b,n_orb)

        wt2vvvoov = 0
        wt2vvvoov += 0.500000000 * np.einsum("kbia,cdjk->bcdija",vmat_ovov,t2.transpose(),optimize="optimal")
        wt2vvvoov += -0.250000000 * np.einsum("bcae,deij->bcdija",vmat_vvvv,t2.transpose(),optimize="optimal")
        print("wt2vvvoov = ",wt2vvvoov.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2vvvoov,n_a,n_b,n_orb)

        wt2vvvovv = 0
        wt2vvvovv += -0.250000000 * np.einsum("jcab,deij->cdeiab",vmat_ovvv,t2.transpose(),optimize="optimal")
        print("wt2vvvovv = ",wt2vvvovv.shape)
        wt2_mat_3 = make_full_three(wt2_mat_3,wt2vvvovv,n_a,n_b,n_orb)

    wt2oooo = 0
    wt2oooo += 0.125000000 * np.einsum("abij,klab->klij",vmat_vvoo,t2,optimize="optimal")
    wt2oooo += 0.125000000 * np.einsum("ijab,abkl->ijkl",vmat_oovv,t2.transpose(),optimize="optimal")
    print("wt2oooo = ",wt2oooo.shape)
    wt2_mat_2 = make_full_two(wt2_mat_2,wt2oooo,n_a,n_b,n_orb)

    wt2ooov = 0
    wt2ooov += 1.000000000 * np.einsum("jbil,klab->jkia",vmat_ovoo,t2,optimize="optimal")
    wt2ooov += 0.250000000 * np.einsum("bcia,jkbc->jkia",vmat_vvov,t2,optimize="optimal")
    print("wt2ooov = ",wt2ooov.shape)
    wt2_mat_2 = make_full_two(wt2_mat_2,wt2ooov,n_a,n_b,n_orb)

    wt2oovv = 0
    wt2oovv += 0.125000000 * np.einsum("ijkl,klab->ijab",vmat_oooo,t2,optimize="optimal")
    wt2oovv += -1.000000000 * np.einsum("icka,jkbc->ijab",vmat_ovov,t2,optimize="optimal")
    wt2oovv += 0.125000000 * np.einsum("cdab,ijcd->ijab",vmat_vvvv,t2,optimize="optimal")
    print("wt2oovv = ",wt2oovv.shape)
    wt2_mat_2 = make_full_two(wt2_mat_2,wt2oovv,n_a,n_b,n_orb)

    wt2ovoo = 0
    wt2ovoo += 1.000000000 * np.einsum("jlib,abkl->jaik",vmat_ooov,t2.transpose(),optimize="optimal")
    wt2ovoo += 0.250000000 * np.einsum("iabc,bcjk->iajk",vmat_ovvv,t2.transpose(),optimize="optimal")
    print("wt2ovoo = ",wt2ovoo.shape)
    wt2_mat_2 = make_full_two(wt2_mat_2,wt2ovoo,n_a,n_b,n_orb)

    wt2ovov = 0
    wt2ovov += -1.000000000 * np.einsum("acik,jkbc->jaib",vmat_vvoo,t2,optimize="optimal")
    wt2ovov += -1.000000000 * np.einsum("ikac,bcjk->ibja",vmat_oovv,t2.transpose(),optimize="optimal")
    print("wt2ovov = ",wt2ovov.shape)
    wt2_mat_2 = make_full_two(wt2_mat_2,wt2ovov,n_a,n_b,n_orb)

    wt2ovvv = 0
    wt2ovvv += 0.250000000 * np.einsum("iajk,jkbc->iabc",vmat_ovoo,t2,optimize="optimal")
    wt2ovvv += 1.000000000 * np.einsum("bdja,ijcd->ibac",vmat_vvov,t2,optimize="optimal")
    print("wt2ovvv = ",wt2ovvv.shape)
    wt2_mat_2 = make_full_two(wt2_mat_2,wt2ovvv,n_a,n_b,n_orb)

    wt2vvoo = 0
    wt2vvoo += 0.125000000 * np.einsum("klij,abkl->abij",vmat_oooo,t2.transpose(),optimize="optimal")
    wt2vvoo += -1.000000000 * np.einsum("kaic,bcjk->abij",vmat_ovov,t2.transpose(),optimize="optimal")
    wt2vvoo += 0.125000000 * np.einsum("abcd,cdij->abij",vmat_vvvv,t2.transpose(),optimize="optimal")
    print("wt2vvoo = ",wt2vvoo.shape)
    wt2_mat_2 = make_full_two(wt2_mat_2,wt2vvoo,n_a,n_b,n_orb)

    wt2vvov = 0
    wt2vvov += 0.250000000 * np.einsum("jkia,bcjk->bcia",vmat_ooov,t2.transpose(),optimize="optimal")
    wt2vvov += 1.000000000 * np.einsum("jbad,cdij->bcia",vmat_ovvv,t2.transpose(),optimize="optimal")
    print("wt2vvov = ",wt2vvov.shape)
    wt2_mat_2 = make_full_two(wt2_mat_2,wt2vvov,n_a,n_b,n_orb)

    wt2vvvv = 0
    wt2vvvv += 0.125000000 * np.einsum("abij,ijcd->abcd",vmat_vvoo,t2,optimize="optimal")
    wt2vvvv += 0.125000000 * np.einsum("ijab,cdij->cdab",vmat_oovv,t2.transpose(),optimize="optimal")
    print("wt2vvvv = ",wt2vvvv.shape)
    wt2_mat_2 = make_full_two(wt2_mat_2,wt2vvvv,n_a,n_b,n_orb)

    wt2oo = 0
    wt2oo += 0.500000000 * np.einsum("abik,jkab->ji",vmat_vvoo,t2,optimize="optimal")
    wt2oo += 0.500000000 * np.einsum("ikab,abjk->ij",vmat_oovv,t2.transpose(),optimize="optimal")
    print("wt2oo = ",wt2oo.shape)
    wt2_mat_1 = make_full_one(wt2_mat_1,wt2oo,n_a,n_b,n_orb)

    wt2ov = 0
    wt2ov += -0.500000000 * np.einsum("ibjk,jkab->ia",vmat_ovoo,t2,optimize="optimal")
    wt2ov += -0.500000000 * np.einsum("bcja,ijbc->ia",vmat_vvov,t2,optimize="optimal")
    print("wt2ov = ",wt2ov.shape)
    wt2_mat_1 = make_full_one(wt2_mat_1,wt2ov,n_a,n_b,n_orb)

    wt2vo = 0
    wt2vo += -0.500000000 * np.einsum("jkib,abjk->ai",vmat_ooov,t2.transpose(),optimize="optimal")
    wt2vo += -0.500000000 * np.einsum("jabc,bcij->ai",vmat_ovvv,t2.transpose(),optimize="optimal")
    print("wt2vo = ",wt2vo.shape)
    wt2_mat_1 = make_full_one(wt2_mat_1,wt2vo,n_a,n_b,n_orb)

    wt2vv = 0
    wt2vv += -0.500000000 * np.einsum("acij,ijbc->ab",vmat_vvoo,t2,optimize="optimal")
    wt2vv += -0.500000000 * np.einsum("ijac,bcij->ba",vmat_oovv,t2.transpose(),optimize="optimal")
    print("wt2vv = ",wt2vv.shape)
    wt2_mat_1 = make_full_one(wt2_mat_1,wt2vv,n_a,n_b,n_orb)

    wt2 = 0 
    wt2 += 0.250000000 * np.einsum("abij,ijab->",vmat_vvoo,t2,optimize="optimal")
    wt2 += 0.250000000 * np.einsum("ijab,abij->",vmat_oovv,t2.transpose(),optimize="optimal")
    print("wt2 = ",wt2)

    # Projecting into the active space
    #three_body += proj_tens_to_as(wt2_mat_3,act_max)
    #two_body += proj_tens_to_as(wt2_mat_2,act_max)
    #one_body += proj_tens_to_as(wt2_mat_1,act_max)
    two_body += wt2_mat_2
    one_body += wt2_mat_1
    constant += wt2

    # [[F,T1]T1]
    # t_ov = t1 and t_vo = transpose(t1)

    # Initializing the ft1t1_mat_1 one body
    ft1t1_mat_1 = np.zeros((2*n_orb,2*n_orb))
    print("ft1t1_mat_1 = ",ft1t1_mat_1.shape)


    ft1t1oo = 0
    ft1t1oo += -1.000000000 * np.einsum("ki,ja,ak->ji",fmat_oo,t1,t1.transpose(),optimize="optimal")
    ft1t1oo += -1.000000000 * np.einsum("ik,ka,aj->ij",fmat_oo,t1,t1.transpose(),optimize="optimal")
    ft1t1oo += 2.000000000 * np.einsum("ba,ib,aj->ij",fmat_vv,t1,t1.transpose(),optimize="optimal")
    print("ft1t1oo = ",ft1t1oo.shape)
    ft1t1_mat_1 = make_full_one(ft1t1_mat_1,ft1t1oo,n_a,n_b,n_orb)

    ft1t1ov = 0
    ft1t1ov += -2.000000000 * np.einsum("bj,ja,ib->ia",fmat_vo,t1,t1,optimize="optimal")
    ft1t1ov += -1.000000000 * np.einsum("ja,ib,bj->ia",fmat_ov,t1,t1.transpose(),optimize="optimal")
    ft1t1ov += -1.000000000 * np.einsum("ib,ja,bj->ia",fmat_ov,t1,t1.transpose(),optimize="optimal")
    print("ft1t1ov = ",ft1t1ov.shape)
    ft1t1_mat_1 = make_full_one(ft1t1_mat_1,ft1t1ov,n_a,n_b,n_orb)

    ft1t1vo = 0
    ft1t1vo += -1.000000000 * np.einsum("bi,jb,aj->ai",fmat_vo,t1,t1.transpose(),optimize="optimal")
    ft1t1vo += -1.000000000 * np.einsum("aj,jb,bi->ai",fmat_vo,t1,t1.transpose(),optimize="optimal")
    ft1t1vo += -2.000000000 * np.einsum("jb,bi,aj->ai",fmat_ov,t1.transpose(),t1.transpose(),optimize="optimal")
    print("ft1t1vo = ",ft1t1vo.shape)
    ft1t1_mat_1 = make_full_one(ft1t1_mat_1,ft1t1vo,n_a,n_b,n_orb)

    ft1t1vv = 0
    ft1t1vv += 2.000000000 * np.einsum("ji,ia,bj->ba",fmat_oo,t1,t1.transpose(),optimize="optimal")
    ft1t1vv += -1.000000000 * np.einsum("ca,ic,bi->ba",fmat_vv,t1,t1.transpose(),optimize="optimal")
    ft1t1vv += -1.000000000 * np.einsum("ac,ib,ci->ab",fmat_vv,t1,t1.transpose(),optimize="optimal")
    print("ft1t1vv = ",ft1t1vv.shape)
    ft1t1_mat_1 = make_full_one(ft1t1_mat_1,ft1t1vv,n_a,n_b,n_orb)

    ft1t1 = 0
    ft1t1 += -2.000000000 * np.einsum("ji,ia,aj->",fmat_oo,t1,t1.transpose(),optimize="optimal")
    ft1t1 += 2.000000000 * np.einsum("ba,ib,ai->",fmat_vv,t1,t1.transpose(),optimize="optimal")
    print("ft1t1 = ",ft1t1)

    # Projecting into the active space
    #one_body += proj_tens_to_as(ft1t1_mat_1,act_max)
    one_body += ft1t1_mat_1
    constant += ft1t1

    # [[F,T2]T1]
    # t_ov = t1 and t_vo = transpose(t1)
    # t_oovv = t2 and t_vvoo = transpose(t2)

    # Initializing the ft2t1_mat_1 one body
    ft2t1_mat_1 = np.zeros((2*n_orb,2*n_orb))
    print("ft2t1_mat_1 = ",ft2t1_mat_1.shape)

    # Initializing the ft2t1_mat_2 two body
    ft2t1_mat_2 = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
    print("ft2t1_mat_2 = ",ft2t1_mat_2.shape)


    ft2t1oooo = 0
    ft2t1oooo += 0.500000000 * np.einsum("ai,bj,klab->klij",fmat_vo,t1.transpose(),t2,optimize="optimal")
    ft2t1oooo += 0.500000000 * np.einsum("ia,jb,abkl->ijkl",fmat_ov,t1,t2.transpose(),optimize="optimal")
    print("ft2t1oooo = ",ft2t1oooo.shape)
    ft2t1_mat_2 = make_full_two(ft2t1_mat_2,ft2t1oooo,n_a,n_b,n_orb)

    ft2t1ooov = 0
    ft2t1ooov += -1.000000000 * np.einsum("il,bj,klab->ikja",fmat_oo,t1.transpose(),t2,optimize="optimal")
    ft2t1ooov += -0.500000000 * np.einsum("ba,ci,jkbc->jkia",fmat_vv,t1.transpose(),t2,optimize="optimal")
    ft2t1ooov += -0.500000000 * np.einsum("cb,bi,jkac->jkia",fmat_vv,t1.transpose(),t2,optimize="optimal")
    print("ft2t1ooov = ",ft2t1ooov.shape)
    ft2t1_mat_2 = make_full_two(ft2t1_mat_2,ft2t1ooov,n_a,n_b,n_orb)

    ft2t1oovv = 0
    ft2t1oovv += 0.500000000 * np.einsum("ck,ka,ijbc->ijab",fmat_vo,t1,t2,optimize="optimal")
    ft2t1oovv += 0.500000000 * np.einsum("ck,ic,jkab->ijab",fmat_vo,t1,t2,optimize="optimal")
    print("ft2t1oovv = ",ft2t1oovv.shape)
    ft2t1_mat_2 = make_full_two(ft2t1_mat_2,ft2t1oovv,n_a,n_b,n_orb)

    ft2t1ovoo = 0
    ft2t1ovoo += -1.000000000 * np.einsum("li,jb,abkl->jaik",fmat_oo,t1,t2.transpose(),optimize="optimal")
    ft2t1ovoo += -0.500000000 * np.einsum("ab,ic,bcjk->iajk",fmat_vv,t1,t2.transpose(),optimize="optimal")
    ft2t1ovoo += -0.500000000 * np.einsum("cb,ic,abjk->iajk",fmat_vv,t1,t2.transpose(),optimize="optimal")
    print("ft2t1ovoo = ",ft2t1ovoo.shape)
    ft2t1_mat_2 = make_full_two(ft2t1_mat_2,ft2t1ovoo,n_a,n_b,n_orb)

    ft2t1ovov = 0
    ft2t1ovov += 1.000000000 * np.einsum("ci,ak,jkbc->jaib",fmat_vo,t1.transpose(),t2,optimize="optimal")
    ft2t1ovov += 1.000000000 * np.einsum("ak,ci,jkbc->jaib",fmat_vo,t1.transpose(),t2,optimize="optimal")
    ft2t1ovov += 1.000000000 * np.einsum("ka,ic,bcjk->ibja",fmat_ov,t1,t2.transpose(),optimize="optimal")
    ft2t1ovov += 1.000000000 * np.einsum("ic,ka,bcjk->ibja",fmat_ov,t1,t2.transpose(),optimize="optimal")
    print("ft2t1ovov = ",ft2t1ovov.shape)
    ft2t1_mat_2 = make_full_two(ft2t1_mat_2,ft2t1ovov,n_a,n_b,n_orb)

    ft2t1ovvv = 0
    ft2t1ovvv += 0.500000000 * np.einsum("ij,ak,jkbc->iabc",fmat_oo,t1.transpose(),t2,optimize="optimal")
    ft2t1ovvv += 0.500000000 * np.einsum("kj,ak,ijbc->iabc",fmat_oo,t1.transpose(),t2,optimize="optimal")
    ft2t1ovvv += 1.000000000 * np.einsum("da,bj,ijcd->ibac",fmat_vv,t1.transpose(),t2,optimize="optimal")
    print("ft2t1ovvv = ",ft2t1ovvv.shape)
    ft2t1_mat_2 = make_full_two(ft2t1_mat_2,ft2t1ovvv,n_a,n_b,n_orb)

    ft2t1vvoo = 0
    ft2t1vvoo += 0.500000000 * np.einsum("kc,ci,abjk->abij",fmat_ov,t1.transpose(),t2.transpose(),optimize="optimal")
    ft2t1vvoo += 0.500000000 * np.einsum("kc,ak,bcij->abij",fmat_ov,t1.transpose(),t2.transpose(),optimize="optimal")
    print("ft2t1vvoo = ",ft2t1vvoo.shape)
    ft2t1_mat_2 = make_full_two(ft2t1_mat_2,ft2t1vvoo,n_a,n_b,n_orb)

    ft2t1vvov = 0
    ft2t1vvov += 0.500000000 * np.einsum("ji,ka,bcjk->bcia",fmat_oo,t1,t2.transpose(),optimize="optimal")
    ft2t1vvov += 0.500000000 * np.einsum("kj,ja,bcik->bcia",fmat_oo,t1,t2.transpose(),optimize="optimal")
    ft2t1vvov += 1.000000000 * np.einsum("ad,jb,cdij->acib",fmat_vv,t1,t2.transpose(),optimize="optimal")
    print("ft2t1vvov = ",ft2t1vvov.shape)
    ft2t1_mat_2 = make_full_two(ft2t1_mat_2,ft2t1vvov,n_a,n_b,n_orb)

    ft2t1vvvv = 0
    ft2t1vvvv += 0.500000000 * np.einsum("ai,bj,ijcd->abcd",fmat_vo,t1.transpose(),t2,optimize="optimal")
    ft2t1vvvv += 0.500000000 * np.einsum("ia,jb,cdij->cdab",fmat_ov,t1,t2.transpose(),optimize="optimal")
    print("ft2t1vvvv = ",ft2t1vvvv.shape)
    ft2t1_mat_2 = make_full_two(ft2t1_mat_2,ft2t1vvvv,n_a,n_b,n_orb)

    ft2t1oo = 0
    ft2t1oo += 1.000000000 * np.einsum("ai,bk,jkab->ji",fmat_vo,t1.transpose(),t2,optimize="optimal")
    ft2t1oo += -1.000000000 * np.einsum("ak,bi,jkab->ji",fmat_vo,t1.transpose(),t2,optimize="optimal")
    ft2t1oo += 1.000000000 * np.einsum("ia,kb,abjk->ij",fmat_ov,t1,t2.transpose(),optimize="optimal")
    ft2t1oo += -1.000000000 * np.einsum("ka,ib,abjk->ij",fmat_ov,t1,t2.transpose(),optimize="optimal")
    print("ft2t1oo = ",ft2t1oo.shape)
    ft2t1_mat_1 = make_full_one(ft2t1_mat_1,ft2t1oo,n_a,n_b,n_orb)

    ft2t1ov = 0
    ft2t1ov += -1.000000000 * np.einsum("ij,bk,jkab->ia",fmat_oo,t1.transpose(),t2,optimize="optimal")
    ft2t1ov += -1.000000000 * np.einsum("kj,bk,ijab->ia",fmat_oo,t1.transpose(),t2,optimize="optimal")
    ft2t1ov += 1.000000000 * np.einsum("ba,cj,ijbc->ia",fmat_vv,t1.transpose(),t2,optimize="optimal")
    ft2t1ov += 1.000000000 * np.einsum("cb,bj,ijac->ia",fmat_vv,t1.transpose(),t2,optimize="optimal")
    print("ft2t1ov = ",ft2t1ov.shape)
    ft2t1_mat_1 = make_full_one(ft2t1_mat_1,ft2t1ov,n_a,n_b,n_orb)

    ft2t1vo = 0
    ft2t1vo += -1.000000000 * np.einsum("ji,kb,abjk->ai",fmat_oo,t1,t2.transpose(),optimize="optimal")
    ft2t1vo += -1.000000000 * np.einsum("kj,jb,abik->ai",fmat_oo,t1,t2.transpose(),optimize="optimal")
    ft2t1vo += 1.000000000 * np.einsum("ab,jc,bcij->ai",fmat_vv,t1,t2.transpose(),optimize="optimal")
    ft2t1vo += 1.000000000 * np.einsum("cb,jc,abij->ai",fmat_vv,t1,t2.transpose(),optimize="optimal")
    print("ft2t1vo = ",ft2t1vo.shape)
    ft2t1_mat_1 = make_full_one(ft2t1_mat_1,ft2t1vo,n_a,n_b,n_orb)

    ft2t1vv = 0
    ft2t1vv += -1.000000000 * np.einsum("ai,cj,ijbc->ab",fmat_vo,t1.transpose(),t2,optimize="optimal")
    ft2t1vv += 1.000000000 * np.einsum("ci,aj,ijbc->ab",fmat_vo,t1.transpose(),t2,optimize="optimal")
    ft2t1vv += -1.000000000 * np.einsum("ia,jc,bcij->ba",fmat_ov,t1,t2.transpose(),optimize="optimal")
    ft2t1vv += 1.000000000 * np.einsum("ic,ja,bcij->ba",fmat_ov,t1,t2.transpose(),optimize="optimal")
    print("ft2t1vv = ",ft2t1vv.shape)
    ft2t1_mat_1 = make_full_one(ft2t1_mat_1,ft2t1vv,n_a,n_b,n_orb)

    ft2t1 = 0
    ft2t1 += 1.000000000 * np.einsum("ai,bj,ijab->",fmat_vo,t1.transpose(),t2,optimize="optimal")
    ft2t1 += 1.000000000 * np.einsum("ia,jb,abij->",fmat_ov,t1,t2.transpose(),optimize="optimal")
    print("ft2t1 = ",ft2t1)

    # Projecting into the active space
    #two_body += proj_tens_to_as(ft2t1_mat_2,act_max)
    #one_body += proj_tens_to_as(ft2t1_mat_1,act_max)
    two_body += ft2t1_mat_2
    one_body += ft2t1_mat_1
    constant += ft2t1

    # [[F,T1]T2]
    # t_ov = t1 and t_vo = transpose(t1)
    # t_oovv = t2 and t_vvoo = transpose(t2)

    # Initializing the ft1t2_mat_1 one body
    ft1t2_mat_1 = np.zeros((2*n_orb,2*n_orb))
    print("ft1t2_mat_1 = ",ft1t2_mat_1.shape)

    # Initializing the ft1t2_mat_2 two body
    ft1t2_mat_2 = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
    print("ft1t2_mat_2 = ",ft1t2_mat_2.shape)

    ft1t2ooov = 0
    ft1t2ooov += 0.500000000 * np.einsum("li,bl,jkab->jkia",fmat_oo,t1.transpose(),t2,optimize="optimal")
    ft1t2ooov += -0.500000000 * np.einsum("cb,bi,jkac->jkia",fmat_vv,t1.transpose(),t2,optimize="optimal")
    print("ft1t2ooov = ",ft1t2ooov.shape)
    ft1t2_mat_2 = make_full_two(ft1t2_mat_2,ft1t2ooov,n_a,n_b,n_orb)

    ft1t2oovv = 0
    ft1t2oovv += 0.500000000 * np.einsum("ck,ka,ijbc->ijab",fmat_vo,t1,t2,optimize="optimal")
    ft1t2oovv += 0.500000000 * np.einsum("ck,ic,jkab->ijab",fmat_vo,t1,t2,optimize="optimal")
    ft1t2oovv += 0.500000000 * np.einsum("ka,ck,ijbc->ijab",fmat_ov,t1.transpose(),t2,optimize="optimal")
    ft1t2oovv += 0.500000000 * np.einsum("ic,ck,jkab->ijab",fmat_ov,t1.transpose(),t2,optimize="optimal")
    print("ft1t2oovv = ",ft1t2oovv.shape)
    ft1t2_mat_2 = make_full_two(ft1t2_mat_2,ft1t2oovv,n_a,n_b,n_orb)

    ft1t2ovoo = 0
    ft1t2ovoo += 0.500000000 * np.einsum("il,lb,abjk->iajk",fmat_oo,t1,t2.transpose(),optimize="optimal")
    ft1t2ovoo += -0.500000000 * np.einsum("cb,ic,abjk->iajk",fmat_vv,t1,t2.transpose(),optimize="optimal")
    print("ft1t2ovoo = ",ft1t2ovoo.shape)
    ft1t2_mat_2 = make_full_two(ft1t2_mat_2,ft1t2ovoo,n_a,n_b,n_orb)

    ft1t2ovvv = 0
    ft1t2ovvv += 0.500000000 * np.einsum("kj,ak,ijbc->iabc",fmat_oo,t1.transpose(),t2,optimize="optimal")
    ft1t2ovvv += -0.500000000 * np.einsum("ad,dj,ijbc->iabc",fmat_vv,t1.transpose(),t2,optimize="optimal")
    print("ft1t2ovvv = ",ft1t2ovvv.shape)
    ft1t2_mat_2 = make_full_two(ft1t2_mat_2,ft1t2ovvv,n_a,n_b,n_orb)

    ft1t2vvoo = 0
    ft1t2vvoo += 0.500000000 * np.einsum("ci,kc,abjk->abij",fmat_vo,t1,t2.transpose(),optimize="optimal")
    ft1t2vvoo += 0.500000000 * np.einsum("ak,kc,bcij->abij",fmat_vo,t1,t2.transpose(),optimize="optimal")
    ft1t2vvoo += 0.500000000 * np.einsum("kc,ci,abjk->abij",fmat_ov,t1.transpose(),t2.transpose(),optimize="optimal")
    ft1t2vvoo += 0.500000000 * np.einsum("kc,ak,bcij->abij",fmat_ov,t1.transpose(),t2.transpose(),optimize="optimal")
    print("ft1t2vvoo = ",ft1t2vvoo.shape)
    ft1t2_mat_2 = make_full_two(ft1t2_mat_2,ft1t2vvoo,n_a,n_b,n_orb)

    ft1t2vvov = 0
    ft1t2vvov += 0.500000000 * np.einsum("kj,ja,bcik->bcia",fmat_oo,t1,t2.transpose(),optimize="optimal")
    ft1t2vvov += -0.500000000 * np.einsum("da,jd,bcij->bcia",fmat_vv,t1,t2.transpose(),optimize="optimal")
    print("ft1t2vvov = ",ft1t2vvov.shape)
    ft1t2_mat_2 = make_full_two(ft1t2_mat_2,ft1t2vvov,n_a,n_b,n_orb)

    ft1t2ov = 0
    ft1t2ov += -1.000000000 * np.einsum("kj,bk,ijab->ia",fmat_oo,t1.transpose(),t2,optimize="optimal")
    ft1t2ov += 1.000000000 * np.einsum("cb,bj,ijac->ia",fmat_vv,t1.transpose(),t2,optimize="optimal")
    print("ft1t2ov = ",ft1t2ov.shape)
    ft1t2_mat_1 = make_full_one(ft1t2_mat_1,ft1t2ov,n_a,n_b,n_orb)

    ft1t2vo = 0
    ft1t2vo += -1.000000000 * np.einsum("kj,jb,abik->ai",fmat_oo,t1,t2.transpose(),optimize="optimal")
    ft1t2vo += 1.000000000 * np.einsum("cb,jc,abij->ai",fmat_vv,t1,t2.transpose(),optimize="optimal")
    print("ft1t2vo = ",ft1t2vo.shape)
    ft1t2_mat_1 = make_full_one(ft1t2_mat_1,ft1t2vo,n_a,n_b,n_orb)

    # Projecting into the active space
    #two_body += proj_tens_to_as(ft1t2_mat_2,act_max)
    #one_body += proj_tens_to_as(ft1t2_mat_1,act_max)
    two_body += ft1t2_mat_2
    one_body += ft1t2_mat_1

    # [[F,T2]T2]
    # t_oovv = t2 and t_vvoo = transpose(t2)

    # Initializing the ft2t2_mat_1 one body
    ft2t2_mat_1 = np.zeros((2*n_orb,2*n_orb))
    print("ft2t2_mat_1 = ",ft2t2_mat_1.shape)

    # Initializing the ft2t2_mat_2 two body
    ft2t2_mat_2 = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
    print("ft2t2_mat_2 = ",ft2t2_mat_2.shape)

    # Initializing the ft2t2_mat_3 three body
    ft2t2_mat_3 = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb,2*n_orb,2*n_orb))
    print("ft2t2_mat_3 = ",ft2t2_mat_3.shape)

    if compute_three_body==True:
        ft2t2ooooov = 0
        ft2t2ooooov += -0.250000000 * np.einsum("ib,jkac,bclm->ijklma",fmat_ov,t2,t2.transpose(),optimize="optimal")
        print("ft2t2ooooov = ",ft2t2ooooov.shape)
        ft2t2_mat_3 = make_full_three(ft2t2_mat_3,ft2t2ooooov,n_a,n_b,n_orb)

        ft2t2ooovvv = 0
        ft2t2ooovvv += 0.500000000 * np.einsum("dl,ilab,jkcd->ijkabc",fmat_vo,t2,t2,optimize="optimal")
        print("ft2t2ooovvv = ",ft2t2ooovvv.shape)
        ft2t2_mat_3 = make_full_three(ft2t2_mat_3,ft2t2ooovvv,n_a,n_b,n_orb)

        ft2t2oovooo = 0
        ft2t2oovooo += -0.250000000 * np.einsum("bi,jkbc,aclm->jkailm",fmat_vo,t2,t2.transpose(),optimize="optimal")
        print("ft2t2oovooo = ",ft2t2oovooo.shape)
        ft2t2_mat_3 = make_full_three(ft2t2_mat_3,ft2t2oovooo,n_a,n_b,n_orb)

        ft2t2oovoov = 0
        ft2t2oovoov += -0.500000000 * np.einsum("mi,jkac,bclm->jkbila",fmat_oo,t2,t2.transpose(),optimize="optimal")
        ft2t2oovoov += -0.500000000 * np.einsum("im,jmac,bckl->ijbkla",fmat_oo,t2,t2.transpose(),optimize="optimal")
        ft2t2oovoov += -0.250000000 * np.einsum("ca,ijcd,bdkl->ijbkla",fmat_vv,t2,t2.transpose(),optimize="optimal")
        ft2t2oovoov += -0.250000000 * np.einsum("ac,ijbd,cdkl->ijaklb",fmat_vv,t2,t2.transpose(),optimize="optimal")
        ft2t2oovoov += -0.500000000 * np.einsum("dc,ijad,bckl->ijbkla",fmat_vv,t2,t2.transpose(),optimize="optimal")
        print("ft2t2oovoov = ",ft2t2oovoov.shape)
        ft2t2_mat_3 = make_full_three(ft2t2_mat_3,ft2t2oovoov,n_a,n_b,n_orb)

        ft2t2oovovv = 0
        ft2t2oovovv += 0.500000000 * np.einsum("la,ijbd,cdkl->ijckab",fmat_ov,t2,t2.transpose(),optimize="optimal")
        ft2t2oovovv += 0.500000000 * np.einsum("id,jlab,cdkl->ijckab",fmat_ov,t2,t2.transpose(),optimize="optimal")
        print("ft2t2oovovv = ",ft2t2oovovv.shape)
        ft2t2_mat_3 = make_full_three(ft2t2_mat_3,ft2t2oovovv,n_a,n_b,n_orb)

        ft2t2ovvoov = 0
        ft2t2ovvoov += 0.500000000 * np.einsum("di,jlad,bckl->jbcika",fmat_vo,t2,t2.transpose(),optimize="optimal")
        ft2t2ovvoov += 0.500000000 * np.einsum("al,ilbd,cdjk->iacjkb",fmat_vo,t2,t2.transpose(),optimize="optimal")
        print("ft2t2ovvoov = ",ft2t2ovvoov.shape)
        ft2t2_mat_3 = make_full_three(ft2t2_mat_3,ft2t2ovvoov,n_a,n_b,n_orb)

        ft2t2ovvovv = 0
        ft2t2ovvovv += -0.250000000 * np.einsum("ki,jlab,cdkl->jcdiab",fmat_oo,t2,t2.transpose(),optimize="optimal")
        ft2t2ovvovv += -0.250000000 * np.einsum("ik,klab,cdjl->icdjab",fmat_oo,t2,t2.transpose(),optimize="optimal")
        ft2t2ovvovv += -0.500000000 * np.einsum("lk,ikab,cdjl->icdjab",fmat_oo,t2,t2.transpose(),optimize="optimal")
        ft2t2ovvovv += -0.500000000 * np.einsum("ea,ikbe,cdjk->icdjab",fmat_vv,t2,t2.transpose(),optimize="optimal")
        ft2t2ovvovv += -0.500000000 * np.einsum("ae,ikbc,dejk->iadjbc",fmat_vv,t2,t2.transpose(),optimize="optimal")
        print("ft2t2ovvovv = ",ft2t2ovvovv.shape)
        ft2t2_mat_3 = make_full_three(ft2t2_mat_3,ft2t2ovvovv,n_a,n_b,n_orb)

        ft2t2ovvvvv = 0
        ft2t2ovvvvv += -0.250000000 * np.einsum("ja,ikbc,dejk->ideabc",fmat_ov,t2,t2.transpose(),optimize="optimal")
        print("ft2t2ovvvvv = ",ft2t2ovvvvv.shape)
        ft2t2_mat_3 = make_full_three(ft2t2_mat_3,ft2t2ovvvvv,n_a,n_b,n_orb)

        ft2t2vvvooo = 0
        ft2t2vvvooo += 0.500000000 * np.einsum("ld,adij,bckl->abcijk",fmat_ov,t2.transpose(),t2.transpose(),optimize="optimal")
        print("ft2t2vvvooo = ",ft2t2vvvooo.shape)
        ft2t2_mat_3 = make_full_three(ft2t2_mat_3,ft2t2vvvooo,n_a,n_b,n_orb)

        ft2t2vvvovv = 0
        ft2t2vvvovv += -0.250000000 * np.einsum("aj,jkbc,deik->adeibc",fmat_vo,t2,t2.transpose(),optimize="optimal")
        print("ft2t2vvvovv = ",ft2t2vvvovv.shape)
        ft2t2_mat_3 = make_full_three(ft2t2_mat_3,ft2t2vvvovv,n_a,n_b,n_orb)

    ft2t2oooo = 0
    ft2t2oooo += 0.250000000 * np.einsum("mi,jkab,ablm->jkil",fmat_oo,t2,t2.transpose(),optimize="optimal")
    ft2t2oooo += 0.250000000 * np.einsum("im,jmab,abkl->ijkl",fmat_oo,t2,t2.transpose(),optimize="optimal")
    ft2t2oooo += 0.500000000 * np.einsum("ba,ijbc,ackl->ijkl",fmat_vv,t2,t2.transpose(),optimize="optimal")
    print("ft2t2oooo = ",ft2t2oooo.shape)
    ft2t2_mat_2 = make_full_two(ft2t2_mat_2,ft2t2oooo,n_a,n_b,n_orb)

    ft2t2ooov = 0
    ft2t2ooov += -0.250000000 * np.einsum("la,ijbc,bckl->ijka",fmat_ov,t2,t2.transpose(),optimize="optimal")
    ft2t2ooov += 1.000000000 * np.einsum("ib,jlac,bckl->ijka",fmat_ov,t2,t2.transpose(),optimize="optimal")
    ft2t2ooov += 0.500000000 * np.einsum("lb,ijac,bckl->ijka",fmat_ov,t2,t2.transpose(),optimize="optimal")
    print("ft2t2ooov = ",ft2t2ooov.shape)
    ft2t2_mat_2 = make_full_two(ft2t2_mat_2,ft2t2ooov,n_a,n_b,n_orb)

    ft2t2ovov = 0
    ft2t2ovov += 1.000000000 * np.einsum("ki,jlac,bckl->jbia",fmat_oo,t2,t2.transpose(),optimize="optimal")
    ft2t2ovov += 1.000000000 * np.einsum("ik,klac,bcjl->ibja",fmat_oo,t2,t2.transpose(),optimize="optimal")
    ft2t2ovov += 2.000000000 * np.einsum("lk,ikac,bcjl->ibja",fmat_oo,t2,t2.transpose(),optimize="optimal")
    ft2t2ovov += -1.000000000 * np.einsum("ca,ikcd,bdjk->ibja",fmat_vv,t2,t2.transpose(),optimize="optimal")
    ft2t2ovov += -1.000000000 * np.einsum("ac,ikbd,cdjk->iajb",fmat_vv,t2,t2.transpose(),optimize="optimal")
    ft2t2ovov += -2.000000000 * np.einsum("dc,ikad,bcjk->ibja",fmat_vv,t2,t2.transpose(),optimize="optimal")
    print("ft2t2ovov = ",ft2t2ovov.shape)
    ft2t2_mat_2 = make_full_two(ft2t2_mat_2,ft2t2ovov,n_a,n_b,n_orb)

    ft2t2ovoo = 0
    ft2t2ovoo += 1.000000000 * np.einsum("bi,jlbc,ackl->jaik",fmat_vo,t2,t2.transpose(),optimize="optimal")
    ft2t2ovoo += -0.250000000 * np.einsum("al,ilbc,bcjk->iajk",fmat_vo,t2,t2.transpose(),optimize="optimal")
    ft2t2ovoo += 0.500000000 * np.einsum("bl,ilbc,acjk->iajk",fmat_vo,t2,t2.transpose(),optimize="optimal")
    print("ft2t2ovoo = ",ft2t2ovoo.shape)
    ft2t2_mat_2 = make_full_two(ft2t2_mat_2,ft2t2ovoo,n_a,n_b,n_orb)

    ft2t2ovvv = 0
    ft2t2ovvv += 1.000000000 * np.einsum("ja,ikbd,cdjk->icab",fmat_ov,t2,t2.transpose(),optimize="optimal")
    ft2t2ovvv += -0.250000000 * np.einsum("id,jkab,cdjk->icab",fmat_ov,t2,t2.transpose(),optimize="optimal")
    ft2t2ovvv += 0.500000000 * np.einsum("jd,ikab,cdjk->icab",fmat_ov,t2,t2.transpose(),optimize="optimal")
    print("ft2t2ovvv = ",ft2t2ovvv.shape)
    ft2t2_mat_2 = make_full_two(ft2t2_mat_2,ft2t2ovvv,n_a,n_b,n_orb)

    ft2t2vvov = 0
    ft2t2vvov += -0.250000000 * np.einsum("di,jkad,bcjk->bcia",fmat_vo,t2,t2.transpose(),optimize="optimal")
    ft2t2vvov += 1.000000000 * np.einsum("aj,jkbd,cdik->acib",fmat_vo,t2,t2.transpose(),optimize="optimal")
    ft2t2vvov += 0.500000000 * np.einsum("dj,jkad,bcik->bcia",fmat_vo,t2,t2.transpose(),optimize="optimal")
    print("ft2t2vvov = ",ft2t2vvov.shape)
    ft2t2_mat_2 = make_full_two(ft2t2_mat_2,ft2t2vvov,n_a,n_b,n_orb)

    ft2t2vvvv = 0
    ft2t2vvvv += -0.500000000 * np.einsum("ji,ikab,cdjk->cdab",fmat_oo,t2,t2.transpose(),optimize="optimal")
    ft2t2vvvv += -0.250000000 * np.einsum("ea,ijbe,cdij->cdab",fmat_vv,t2,t2.transpose(),optimize="optimal")
    ft2t2vvvv += -0.250000000 * np.einsum("ae,ijbc,deij->adbc",fmat_vv,t2,t2.transpose(),optimize="optimal")
    print("ft2t2vvvv = ",ft2t2vvvv.shape)
    ft2t2_mat_2 = make_full_two(ft2t2_mat_2,ft2t2vvvv,n_a,n_b,n_orb)

    ft2t2oo = 0
    ft2t2oo += -0.500000000 * np.einsum("ki,jlab,abkl->ji",fmat_oo,t2,t2.transpose(),optimize="optimal")
    ft2t2oo += -0.500000000 * np.einsum("ik,klab,abjl->ij",fmat_oo,t2,t2.transpose(),optimize="optimal")
    ft2t2oo += -1.000000000 * np.einsum("lk,ikab,abjl->ij",fmat_oo,t2,t2.transpose(),optimize="optimal")
    ft2t2oo += 2.000000000 * np.einsum("ba,ikbc,acjk->ij",fmat_vv,t2,t2.transpose(),optimize="optimal")
    print("ft2t2oo = ",ft2t2oo.shape)
    ft2t2_mat_1 = make_full_one(ft2t2_mat_1,ft2t2oo,n_a,n_b,n_orb)

    ft2t2ov = 0
    ft2t2ov += -0.500000000 * np.einsum("ja,ikbc,bcjk->ia",fmat_ov,t2,t2.transpose(),optimize="optimal")
    ft2t2ov += -0.500000000 * np.einsum("ib,jkac,bcjk->ia",fmat_ov,t2,t2.transpose(),optimize="optimal")
    ft2t2ov += 1.000000000 * np.einsum("jb,ikac,bcjk->ia",fmat_ov,t2,t2.transpose(),optimize="optimal")
    print("ft2t2ov = ",ft2t2ov.shape)
    ft2t2_mat_1 = make_full_one(ft2t2_mat_1,ft2t2ov,n_a,n_b,n_orb)

    ft2t2vo = 0
    ft2t2vo += -0.500000000 * np.einsum("bi,jkbc,acjk->ai",fmat_vo,t2,t2.transpose(),optimize="optimal")
    ft2t2vo += -0.500000000 * np.einsum("aj,jkbc,bcik->ai",fmat_vo,t2,t2.transpose(),optimize="optimal")
    ft2t2vo += 1.000000000 * np.einsum("bj,jkbc,acik->ai",fmat_vo,t2,t2.transpose(),optimize="optimal")
    print("ft2t2vo = ",ft2t2vo.shape)
    ft2t2_mat_1 = make_full_one(ft2t2_mat_1,ft2t2vo,n_a,n_b,n_orb)

    ft2t2vv = 0
    ft2t2vv += 2.000000000 * np.einsum("ji,ikac,bcjk->ba",fmat_oo,t2,t2.transpose(),optimize="optimal")
    ft2t2vv += -0.500000000 * np.einsum("ca,ijcd,bdij->ba",fmat_vv,t2,t2.transpose(),optimize="optimal")
    ft2t2vv += -0.500000000 * np.einsum("ac,ijbd,cdij->ab",fmat_vv,t2,t2.transpose(),optimize="optimal")
    ft2t2vv += -1.000000000 * np.einsum("dc,ijad,bcij->ba",fmat_vv,t2,t2.transpose(),optimize="optimal")
    print("ft2t2vv = ",ft2t2vv.shape)
    ft2t2_mat_1 = make_full_one(ft2t2_mat_1,ft2t2vv,n_a,n_b,n_orb)

    ft2t2 = 0
    ft2t2 += -1.000000000 * np.einsum("ji,ikab,abjk->",fmat_oo,t2,t2.transpose(),optimize="optimal")
    ft2t2 += 1.000000000 * np.einsum("ba,ijbc,acij->",fmat_vv,t2,t2.transpose(),optimize="optimal")
    print("ft2t2 = ",ft2t2)

    # Projecting into the active space
    #three_body += proj_tens_to_as(ft2t2_mat_3,act_max)
    #two_body += proj_tens_to_as(ft2t2_mat_2,act_max)
    #one_body += proj_tens_to_as(ft2t2_mat_1,act_max)
    two_body += ft2t2_mat_2
    one_body += ft2t2_mat_1
    constant += ft2t2

    print("DUCC contractions computed successfully.")
    return constant, one_body, two_body, three_body

if __name__ == "__main__":
    geometry = [('Be', (0,0,0))]
    charge = 0
    spin = 0
    basis = 'cc-pvdz'

    [n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis,reference='rhf')
    sq_ham = pyscf_helper.SQ_Hamiltonian()
    sq_ham.init(h, g, C, S)
    print(" HF Energy: %21.15f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))

    n_occ = n_a+n_b
    n_act = 5
    n_frz = 0
    act_max = n_frz+n_act
    shift = 2*n_frz

    mf = scf.RHF(mol)
    mf.conv_tol_grad = 1e-14
    mf.max_cycle = 1000
    mf.verbose = 0
    mf.init_guess = 'atom'
    mf.kernel()
    print("MO energies:")
    print(mf.mo_energy)

    mccsd = cc.UCCSD(mf)
    mccsd.conv_tol = 1e-12
    mccsd.conv_tol_normt = 1e-10
    #myccsd.diis_start_cycle=10000
    mccsd.max_cycle = 1000
    mccsd.kernel()
    t1_amps = mccsd.t1
    t2_amps = mccsd.t2
    fmat = sq_ham.make_f(range(n_a),range(n_b))
    vmat = sq_ham.make_v()
    
    
    #fmat -= umat


    t1_amps, t2_amps = get_t_ext(t1_amps,t2_amps,n_a,n_b,act_max)

    t1_amps, t2_amps = transform_t_spatial_to_spin(t1_amps, t2_amps, n_a, n_b, n_orb)

    constant, one_body, two_body, three_body = compute_ducc(fmat,vmat,t1_amps,t2_amps,n_a,n_b,n_occ,n_orb,act_max,compute_three_body=False)

    C = mf.mo_coeff

    h  = pyscf.scf.hf.get_hcore(mf.mol)
    j,k = mf.get_jk()
    print("j in AO")
    tprint(k)
    h = C.T @ h @ C;
    j = C.T @ j @ C;
    k = C.T @ k @ C;
    print("j in MO")
    tprint(k)
    #h = h +j-0.5*k
    #j = j[0:act_max,0:act_max]
    #k = k[0:act_max,0:act_max]
    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    h2 = pyscf.ao2mo.kernel(mol, C, aosym="s4", compact=False)
    h2.shape = (n_orb, n_orb, n_orb, n_orb)

    print("shape of h1 and h2",h1.shape,h2.shape)
    #Fao = mf.get_fock()
    #Fmo = C.T @ Fao @ C
    #print("Test 0 = ",np.linalg.norm(Fmo-h))

    #u_tilde = get_u(two_body,range(n_a),range(n_b))
    #one_body -= u_tilde
    
    #umat = sq_ham.make_u(range(n_a),range(n_b))
    
    umat_1 = get_u(two_body,range(n_a),range(n_b))
    
    one_body -= umat_1

    one_elec_test = get_spin_to_spatial(one_body,n_orb)
    two_elec_test = get_spin_to_spatial(two_body,n_orb)


    #print("Test u = ",np.linalg.norm(umat-umat_1))
    
    #one_body += u_piqi


    #fmat_spatial_act = fmat_spatial[0:act_max,0:act_max]
    #vmat_spatial_act = vmat_spatial[0:act_max,0:act_max,0:act_max,0:act_max]


    #print("Test 1 (get_u) = ",np.linalg.norm(fmat_spatial-h1))
    #print("Test 2 (make_u) = ",np.linalg.norm(fmat_spatial-h1))

    h1 = one_elec_test[0:act_max,0:act_max]
    h2 = two_elec_test[0:act_max,0:act_max,0:act_max,0:act_max]  

    cisolver = fci.direct_spin1.FCI()
    ecore = constant
    nroots = 1
    nelec = n_occ

    efci_orgnl,ci_orgnl = cisolver.kernel(h1, h2, h1.shape[1], nelec, ecore=ecore,nroots =nroots,verbose=100)
    fci_dim_orgnl = ci_orgnl.shape[0]*ci_orgnl.shape[1]
    print(" FCI:        %12.8f Dim:%6d"%(efci_orgnl,fci_dim_orgnl))


    #efci, ci = cisolver.kernel(fmat_spatial_act, vmat_spatial_act, fmat_spatial_act.shape[1], nelec, ecore=constant,nroots =nroots,verbose=100)
    #fci_dim = ci.shape[0]*ci.shape[1]
    #print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))

    print(" SCF:        %12.8f"%(E_scf))

    print("Correlation energy = ",efci_orgnl-E_scf)
    #print("Correlation energy 2= ",efci-E_scf)

    

    save_data = False
    if save_data == True:
        np.savez('ducc_data_constant',constant)
        np.savez('ducc_data_one_body',one_body)
        np.savez('ducc_data_two_body',two_body)
        np.savez('ducc_data_three_body',three_body)
        print("DUCC data saved successfully.")






