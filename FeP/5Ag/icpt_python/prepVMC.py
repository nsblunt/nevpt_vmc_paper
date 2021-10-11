import numpy as np
import math
from pyscf import gto, scf, ao2mo, mcscf, tools, molden, lib, symm
import os, sys
import NEVPT2Helper as nev

print('About to read mol from chk file...')
sys.stdout.flush()

# make your molecule here
chkName = 'feP_HF.chk'
mol = lib.chkfile.load_mol(chkName)
mol.spin = 4
mol.basis = 'ccpvdz'
mol.verbose = 4

print('Done.')
sys.stdout.flush()

print('About to initialize mf object...')
sys.stdout.flush()

mf = scf.RHF(mol)
mf.irrep_nelec = {'Ag': (25,24), 'B3u': (19,19), 'B2u': (19,19), 'B1g': (15,14), 'B1u': (7,7), 'B2g': (4,3), 'B3g': (4,3),'Au': (2,2)}

print('Done.')
sys.stdout.flush()

print('About to read molecular orbitals...')
sys.stdout.flush()

#mf.kernel()
mf.__dict__.update(scf.chkfile.load(chkName, 'scf'))

print('Done.')
sys.stdout.flush()

print('About to initialize mc object...')
sys.stdout.flush()

from pyscf.shciscf import shci
norb = 29
nelec = 32
mc = shci.SHCISCF(mf, norb, nelec)

print('Done.')
sys.stdout.flush()

print('About to read mc object from chkfile...')
sys.stdout.flush()

chkfile = "feP_5AgOrbs.chk"
mc.__dict__.update( lib.chkfile.load(chkfile, 'mcscf') )

print('Done.')
sys.stdout.flush()

nFrozen = 33

#this is the scratch file for writing potentially larger integral files
#intfolder = "int/"
#os.system("mkdir -p "+intfolder)
#
#dm2a = np.zeros((norb, norb, norb, norb))
#file2pdm = "spatialRDM.0.0.txt"
#file2pdm = file2pdm.encode()  # .encode for python3 compatibility
#shci.r2RDM(dm2a, norb, file2pdm)
#dm1 = np.einsum('ikjj->ki', dm2a)
#dm1 /= (nelec - 1)
##dm1, dm2a = mc.fcisolver.make_rdm12(0, mc.ncas, mc.nelecas)
#dm2 = np.einsum('ijkl->ikjl', dm2a)
#
#np.save(intfolder+"E2.npy", np.asfortranarray(dm2))
#np.save(intfolder+"E1.npy", np.asfortranarray(dm1))
#
#print ("trace of 2rdm", np.einsum('ijij',dm2))
#print ("trace of 1rdm", np.einsum('ii',dm1))
#
#nfro = 33
#E1eff = dm1 # for state average
#nev.writeNEVPTIntegrals(mc, dm1, dm2, E1eff, nfro, intfolder)
#nev.write_ic_inputs(mc.nelecas[0]+mc.nelecas[1], mc.ncore, mc.ncas, nfro, mc.nelecas[0]-mc.nelecas[1],
#                   'NEVPT2')
