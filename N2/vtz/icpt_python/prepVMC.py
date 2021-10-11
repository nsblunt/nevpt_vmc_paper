import numpy as np
import math
from pyscf import gto, scf, ao2mo, mcscf, tools, lib
from pyscf.shciscf import shci
import NEVPT2Helper as nev
import os, sys

# make your molecule here
r = 2.5 * 0.529177
atomstring = "N 0 0 0; N 0 0 %g"%(r)
mol = gto.M(
    atom = atomstring,
    basis = 'cc-pvtz',
    verbose=4,
    symmetry=0,
    spin = 0)

mf = scf.RHF(mol)
chkfile = 'N2_HF.chk'
mf.__dict__.update(scf.chkfile.load(chkfile, 'scf'))

norbAct = 8
nelecAct = 10
norbFrozen = 0

mc = shci.SHCISCF(mf, norbAct, nelecAct)
chkfile = 'N2_SHCISCF.chk'
mc.__dict__.update( lib.chkfile.load(chkfile, 'mcscf') )

#this is the scratch file for writing potentially larger integral files
intfolder = "int/"
os.system("mkdir -p "+intfolder)

dm2a = np.zeros((norbAct, norbAct, norbAct, norbAct))
file2pdm = "spatialRDM.0.0.txt"
file2pdm = file2pdm.encode()  # .encode for python3 compatibility
shci.r2RDM(dm2a, norbAct, file2pdm)
dm1 = np.einsum('ikjj->ki', dm2a)
dm1 /= (nelecAct - 1)
#dm1, dm2a = mc.fcisolver.make_rdm12(0, mc.ncas, mc.nelecas)
dm2 = np.einsum('ijkl->ikjl', dm2a)

np.save(intfolder+"E2.npy", np.asfortranarray(dm2))
np.save(intfolder+"E1.npy", np.asfortranarray(dm1))

print ("trace of 2rdm", np.einsum('ijij',dm2))
print ("trace of 1rdm", np.einsum('ii',dm1))

nfro = 0
E1eff = dm1 # for state average
nev.writeNEVPTIntegrals(mc, dm1, dm2, E1eff, nfro, intfolder)
nev.write_ic_inputs(mc.nelecas[0]+mc.nelecas[1], mc.ncore, mc.ncas, nfro, mc.nelecas[0]-mc.nelecas[1],
                   'NEVPT2')
