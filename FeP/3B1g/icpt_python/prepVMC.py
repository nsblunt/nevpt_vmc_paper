import numpy as np
import math
from pyscf import gto, scf, ao2mo, mcscf, tools, molden, lib, symm
import os, sys
import NEVPT2Helper as nev

print('About to read mol from chk file...')
sys.stdout.flush()

# make your molecule here
chkName = "fe_dz_5Ag_hf.chk"
mol = lib.chkfile.load_mol(chkName)

print('Done.')
sys.stdout.flush()

mol.spin = 2
mol.verbose = 4
mf = scf.RHF(mol)

print('About to read molecular orbitals...')
sys.stdout.flush()

mf.__dict__.update( lib.chkfile.load( chkName, 'scf') )
mf.irrep_nelec = {
    "Ag": (25, 24),
    "B3u": (19, 19),
    "B2u": (19, 19),
    "B1g": (15, 14),
    "B1u": (7, 7),
    "B2g": (4, 4),
    "B3g": (3, 3),
    "Au": (2, 2),
}

print('Done.')
sys.stdout.flush()
#mf.kernel()

print('About to initialize mc object...')
sys.stdout.flush()

from pyscf.shciscf import shci
norb = 29
nelec = 32
mc = shci.SHCISCF(mf, norb, nelec)
chkfile = "fe_dz_3B1g_SHCISCF.chk"
mc.__dict__.update( lib.chkfile.load(chkfile, 'mcscf') )

print('Done.')
sys.stdout.flush()

nFrozen = 33

#fileh = open("moEne.txt", 'w')
#for i in range(mol.nao - nFrozen):
#  fileh.write('%.12e\n'%(mc.mo_energy[i + nFrozen]))
#fileh.close()
#
#print('About to print FCIDUMP for all orbitals...')
#sys.stdout.flush()
#
#tools.fcidump.from_mc(mc, 'FCIDUMP_all_orbs', nFrozen=33)
#
#print('Done.')
#sys.stdout.flush()
#
#print('About to print FCIDUMP in CAS...')
#sys.stdout.flush()
#
## active space integrals for dice calculation
#mo_core = mc.mo_coeff[:,:77]
#mo_rest = mc.mo_coeff[:,77:77+29]
#core_dm = 2 * mo_core.dot(mo_core.T)
#corevhf = mc.get_veff(mol, core_dm)
#energy_core = mol.energy_nuc()
#energy_core += np.einsum('ij,ji', core_dm, mc.get_hcore())
#energy_core += np.einsum('ij,ji', core_dm, corevhf) * .5
#h1eff = mo_rest.T.dot(mc.get_hcore() + corevhf).dot(mo_rest)
#eri = ao2mo.kernel(mol, mo_rest)
#tools.fcidump.from_integrals('FCIDUMP', h1eff, eri, 29, 32, energy_core)
#
#print('Done.')
#sys.stdout.flush()


#this is the scratch file for writing potentially larger integral files
intfolder = "int/"
os.system("mkdir -p "+intfolder)

dm2a = np.zeros((norb, norb, norb, norb))
file2pdm = "spatialRDM.0.0.txt"
file2pdm = file2pdm.encode()  # .encode for python3 compatibility
shci.r2RDM(dm2a, norb, file2pdm)
dm1 = np.einsum('ikjj->ki', dm2a)
dm1 /= (nelec - 1)
#dm1, dm2a = mc.fcisolver.make_rdm12(0, mc.ncas, mc.nelecas)
dm2 = np.einsum('ijkl->ikjl', dm2a)

np.save(intfolder+"E2.npy", np.asfortranarray(dm2))
np.save(intfolder+"E1.npy", np.asfortranarray(dm1))

print ("trace of 2rdm", np.einsum('ijij',dm2))
print ("trace of 1rdm", np.einsum('ii',dm1))

nfro = 33
E1eff = dm1 # for state average
nev.writeNEVPTIntegrals(mc, dm1, dm2, E1eff, nfro, intfolder)
nev.write_ic_inputs(mc.nelecas[0]+mc.nelecas[1], mc.ncore, mc.ncas, nfro, mc.nelecas[0]-mc.nelecas[1],
                   'NEVPT2')
