from __future__ import print_function, division
import matplotlib.pyplot as plt  # plotting library
import numpy as np
from quspin.tools.evolution import expm_multiply_parallel
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general
#
import sys
import os
# uncomment this line if omp error occurs on OSX for python 3
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# set number of OpenMP threads to run in parallel
os.environ['OMP_NUM_THREADS'] = '1'
# set number of MKL threads to run in parallel
os.environ['MKL_NUM_THREADS'] = '1'
#
quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)
########################################################################
#                            example 19                                #
# This exampled shows how to use the Op_shift_sector() method of the   #
# general basis classes to compute autocorrelation functions in the    #
# Heisenberg model.                                                    #
########################################################################
#
# Heisenberg model parameers
#
L = 10
Jxy = 1.0  # xy interaction
Jzz_0 = 1.0  # zz interaction
#
# hamiltonian parameters
# J_list = [[1.0, i, (i+1) % L] for i in range(L)]
# J_list = [[1.0/4.0, i, (i+1)] for i in range(L-1)]
J_zz = [[Jzz_0, i, (i+1) % L] for i in range(L)]  # PBC
J_xy = [[Jxy/2.0, i, (i+1) % L] for i in range(L)]  # PBC
static = [["+-", J_xy], ["-+", J_xy], ["zz", J_zz]]
# static = [[op, J_list] for op in ["-+", "+-", "zz"]]
# time vector
times = np.linspace(0.0, 25.0, 101)
#
# straightforward autocorrelation function without using symmetries
#


def auto_correlator(L, times, S="1/2"):
    # construct basis in zero magnetization sector: no lattice symmetries
    #     basis = spin_basis_general(L, S=S, m=0, pauli=False)
    # use spin operators instead of Pauli
    basis = spin_basis_general(L, S=S, pauli=False)
    # define Heisenberg Hamiltonian
    no_checks = dict(check_symm=False, check_herm=False, check_pcon=False)
    H = hamiltonian(static, [], basis=basis, dtype=np.float64, **no_checks)
    # compute GS
    E, V = H.eigsh(k=1, which="SA")
    print(E[0])
    psi_GS = V[:, 0]
    # evolve GS under H (gives a trivial phase factor)
    psi_GS_t = H.evolve(psi_GS, 0.0, times)
    #
    # define operator O to compute the autocorrelation function of
    #
    op_list = [["x", [0], np.sqrt(2.0)]]
    # use inplace_Op to apply operator O on psi_GS
#     Opsi_GS = basis.inplace_Op(psi_GS, op_list, np.float64)
    Opsi_GS = basis.inplace_Op(psi_GS, op_list, np.complex64)
    # time evolve Opsi_GS under H
    Opsi_GS_t = H.evolve(Opsi_GS, 0.0, times)
    # apply operator O on time-evolved psi_t
#     op_list2 = [["-", [0], np.sqrt(2.0)]]
#     O_psi_GS_t = basis.inplace_Op(psi_GS_t, op_list, np.float64)
    O_psi_GS_t = basis.inplace_Op(psi_GS_t, op_list, np.complex64)
    # compute autocorrelator
    C_t = np.einsum("ij,ij->j", O_psi_GS_t.conj(), Opsi_GS_t)
    #
    return C_t
#


def auto_correlator_symm(L, times, S="1/2"):
    # define momentum p sector of the GS of the Heisenberg Hamiltonian
    if (L//2) % 2:
        p = L//2  # corresponds to momentum pi
        dtype = np.complex128
    else:
        p = 0
        dtype = np.float64
    #
    # define translation operator
    T = (np.arange(L)+1) % L
    # compute the basis in the momentum sector of the GS of the Heisenberg model
    basis_p = spin_basis_general(L, S=S, m=0, kblock=(T, p), pauli=False)
    # define Heisenberg Hamiltonian
    no_checks = dict(check_symm=False, check_herm=False, check_pcon=False)
    H = hamiltonian(static, [], basis=basis_p, dtype=dtype, **no_checks)
    # compute GS
    E, V = H.eigsh(k=1, which="SA")
    psi_GS = V[:, 0]
    # evolve GS under symmetry-reduced H (gives a trivial phase factor)
    psi_GS_t = H.evolve(psi_GS, 0, times)
    #
    # compute autocorrelation function foe every momentum sector
    Cq_t = np.zeros((times.shape[0], L), dtype=np.complex128)
    #
    for q in range(L):  # sum over symmetry sectors
        #
        # define operator O_q, sum over lattice sites
        op_list = [
            ["z", [j], (np.sqrt(2.0)/L)*np.exp(-1j*2.0*np.pi*q*j/L)] for j in range(L)]
        # compute basis in the (q+p)-momentum sector (the total momentum of O_q|psi_GS> is q+p)
        basis_q = spin_basis_general(L, S=S, m=0, kblock=(T, p+q), pauli=False)
        # define Hamiltonian in the q-momentum sector
        Hq = hamiltonian(static, [], basis=basis_q,
                         dtype=np.complex128, **no_checks)
        # use Op_shift_sector apply operator O_q to GS; the momentum of the new state is p+q
        Opsi_GS = basis_q.Op_shift_sector(basis_p, op_list, psi_GS)
        # time evolve Opsi_GS under H_q
        Opsi_GS_t = Hq.evolve(Opsi_GS, 0.0, times)
        # apply operator O on time-evolved psi_t
        O_psi_GS_t = basis_q.Op_shift_sector(basis_p, op_list, psi_GS_t)
        # compute autocorrelator for every momentum sector
        Cq_t[..., q] = np.einsum("ij,ij->j", O_psi_GS_t.conj(), Opsi_GS_t)
    #
    return np.sum(Cq_t, axis=1)  # sum over momentum sectors


#
# compute autocorrelation function
C_t = auto_correlator(L, times)
C_t_symm = auto_correlator_symm(L, times)
#
# plot result
#
plt.plot(times, C_t.real, '-b', label='no symm.: $\\mathrm{Re}\\;C(t)$')
plt.plot(times, C_t.imag, '-r', label='no symm.: $\\mathrm{Im}\\;C(t)$')
#
plt.plot(times, C_t_symm.real, 'ob', label='symm.: $\\mathrm{Re}\\;C(t)$')
plt.plot(times, C_t_symm.imag, 'or', label='symm.: $\\mathrm{Im}\\;C(t)$')
#
plt.xlabel('time $t$')
plt.legend()
#
plt.show()
