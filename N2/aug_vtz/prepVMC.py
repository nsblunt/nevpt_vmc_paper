import numpy as np
import math
from pyscf import gto, scf, ao2mo, mcscf, tools
from pyscf.shciscf import shci
import os

# make your molecule here
r = 2.5 * 0.529177
atomstring = "N 0 0 0; N 0 0 %g"%(r)
mol = gto.M(
    atom = atomstring,
    basis = 'aug-cc-pvtz',
    verbose=4,
    symmetry=0,
    spin = 0)

mf = scf.RHF(mol)
mf.chkfile = 'N2_HF.chk'
mf.kernel()

norbAct = 8
nelecAct = 10
norbFrozen = 0

mc = shci.SHCISCF(mf, norbAct, nelecAct)
mc.chkfile = 'N2_SHCISCF.chk'
mc.fcisolver.sweep_iter = [0]
mc.fcisolver.sweep_epsilon = [0]
mc.fcisolver.nPTiter = 0
mc.max_cycle_macro = 30
mc.fcisolver.nPTiter = 0  # Turns off PT calculation, i.e. no PTRDM.
mc.kernel()

fileh = open("moEne.txt", 'w')
for i in range(mol.nao - norbFrozen):
  fileh.write('%.12e\n'%(mc.mo_energy[i + norbFrozen]))
fileh.close()

tools.fcidump.from_mc(mc, 'FCIDUMP.h5', nFrozen = norbFrozen)

# active space integrals for dice calculation
mo_core = mc.mo_coeff[:,:norbCore]
mo_rest = mc.mo_coeff[:,norbCore:norbCore + norbAct]
core_dm = 2 * mo_core.dot(mo_core.T)
corevhf = mc.get_veff(mol, core_dm)
energy_core = mol.energy_nuc()
energy_core += np.einsum('ij,ji', core_dm, mc.get_hcore())
energy_core += np.einsum('ij,ji', core_dm, corevhf) * .5
h1eff = mo_rest.T.dot(mc.get_hcore() + corevhf).dot(mo_rest)
eri = ao2mo.kernel(mol, mo_rest)
tools.fcidump.from_integrals('FCIDUMP', h1eff, eri, norbAct, nelecAct, energy_core, tol=1e-12)
