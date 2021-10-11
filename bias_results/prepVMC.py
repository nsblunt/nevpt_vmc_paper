import numpy as np
import math
from pyscf import gto, scf, ao2mo, mcscf, tools, fci, mp
import sys
import os
import NEVPT2Helper as nev

def doRHF(mol):
  mf = scf.RHF(mol)
  print (mf.kernel())
  return mf


# make your molecule here
r = 2.5 * 0.529177
atomstring = "N 0 0 0; N 0 0 %g"%(r)
mol = gto.M(
    atom = atomstring,
    basis = 'cc-pvdz',
    verbose=4,
    symmetry=0,
    spin = 0)

mf = doRHF(mol)
mc = mcscf.CASSCF(mf, 8, 10)
mc.kernel()

print ("moEne")
for i in range(28):
  print (mc.mo_energy[i])

tools.fcidump.from_mo(mol, 'FCIDUMP_all_orbs', mc.mo_coeff)

#this is the scratch file for writing potentially larger integral files
intfolder = "int/"
os.system("mkdir -p "+intfolder)

#this code will be used for generating the files that are needed for deterministic PT
from pyscf.shciscf import shci
mc2 = shci.SHCISCF(mf, 8, 10)
mc2.max_cycle_macro=0
mc2.mo_coeff = 1.*mc.mo_coeff

mc2.fcisolver.sweep_iter = [0]
mc2.fcisolver.sweep_epsilon = [0.0]

ecas=mc2.kernel(mc.mo_coeff)[0]
dm1, dm2a = mc2.fcisolver.make_rdm12(0, mc.ncas, mc.nelecas)
dm2 = np.einsum('ijkl->ikjl', dm2a)

np.save(intfolder+"E2.npy", np.asfortranarray(dm2))
np.save(intfolder+"E1.npy", np.asfortranarray(dm1))

print ("trace of 2rdm", np.einsum('ijij',dm2))
print ("trace of 1rdm", np.einsum('ii',dm1))

nfro = 0
E1eff = dm1 #for state average
nev.writeNEVPTIntegrals(mc2, dm1, dm2, E1eff, nfro, intfolder)
nev.write_ic_inputs(mc.nelecas[0]+mc.nelecas[1], mc.ncore, mc.ncas, nfro, mc.nelecas[0]-mc.nelecas[1],
                   'NEVPT2')
