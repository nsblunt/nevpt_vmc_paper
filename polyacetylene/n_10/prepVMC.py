import numpy as np
from pyscf import gto, scf, ao2mo, tools, mp, symm
from pyscf.shciscf import shci
import sys

def myocc(mf):
   mol = mf.mol
   irrep_id = mol.irrep_id
   so = mol.symm_orb
   orbsym = symm.label_orb_symm(mol, irrep_id, so, mf.mo_coeff)
   doccsym = np.array(orbsym)[mf.mo_occ==2]
   soccsym = np.array(orbsym)[mf.mo_occ==1]
   for ir, irname in enumerate(mol.irrep_name):
     print('%s, double-occ = %d, single-occ = %d' %
         (irname, sum(doccsym==ir), sum(soccsym==ir)))

# trans-polyacetylene: C_{2n}H_{2n+2}
if len(sys.argv) == 2:
    n = int(sys.argv[1])
else:
    n = 2

print("n:", n)

# bond lengths and angles (assuming uniform c=c, c-c and c-h bond lengths, and all angles to be 120 degrees)
b1 = 1.45
b2 = 1.34
bh = 1.08
t = 2*np.pi/3

if n%2 == 0:
  # vector displacements on one side
  c0 = np.array([b1/2, 0., 0.])
  cs = np.array([b1, 0., 0.])
  cd = np.array([b2/2, b2*3**0.5/2, 0.])
  ch = np.array([bh/2, -bh*3**0.5/2, 0.])
  cht = np.array([bh, 0., 0.])

  atomString = 'C {} {} {};\nC {} {} {};\n'.format(c0[0], c0[1], c0[2], -c0[0], -c0[1], -c0[2])
  currentC = c0
  for i in range(n-1):
      if i%2 == 0:
          newC = currentC + cd
          newH = currentC + ch
      else:
          newC = currentC + cs
          newH = currentC - ch
      atomString += 'C {} {} {};\nC {} {} {};\n'.format(newC[0], newC[1], newC[2], -newC[0], -newC[1], -newC[2])
      atomString += 'H {} {} {};\nH {} {} {};\n'.format(newH[0], newH[1], newH[2], -newH[0], -newH[1], -newH[2])
      currentC = newC

  # terminal h's
  th1 = currentC + cht
  th2 = currentC - ch
  atomString += 'H {} {} {};\nH {} {} {};\n'.format(th1[0], th1[1], th1[2], -th1[0], -th1[1], -th1[2])
  atomString += 'H {} {} {};\nH {} {} {};\n'.format(th2[0], th2[1], th2[2], -th2[0], -th2[1], -th2[2])

else:
  # vector displacements on one side
  c0 = np.array([b2/2, 0., 0.])
  cd = np.array([b2, 0., 0.])
  cs = np.array([b1/2, b1*3**0.5/2, 0.])
  ch = np.array([bh/2, -bh*3**0.5/2, 0.])
  cht = np.array([bh/2, bh*3**0.5/2, 0.])

  atomString = 'C {} {} {};\nC {} {} {};\n'.format(c0[0], c0[1], c0[2], -c0[0], -c0[1], -c0[2])
  currentC = c0
  for i in range(n-1):
      if i%2 == 0:
          newC = currentC + cs
          newH = currentC + ch
      else:
          newC = currentC + cd
          newH = currentC - ch
      atomString += 'C {} {} {};\nC {} {} {};\n'.format(newC[0], newC[1], newC[2], -newC[0], -newC[1], -newC[2])
      atomString += 'H {} {} {};\nH {} {} {};\n'.format(newH[0], newH[1], newH[2], -newH[0], -newH[1], -newH[2])
      currentC = newC

  # terminal h's
  th1 = currentC + cht
  th2 = currentC + ch
  atomString += 'H {} {} {};\nH {} {} {};\n'.format(th1[0], th1[1], th1[2], -th1[0], -th1[1], -th1[2])
  atomString += 'H {} {} {};\nH {} {} {};\n'.format(th2[0], th2[1], th2[2], -th2[0], -th2[1], -th2[2])

print("atomString:")
print(atomString)

mol = gto.M(atom = atomString, basis = '6-31g', verbose = 4, unit = 'angstrom', symmetry = 1, spin = 0)
mf = scf.RHF(mol)
mf.max_cycle = 500
mf.kernel()

# shciscf
norbAct = 2*n
nelecAct = 2*n
mc = shci.SHCISCF(mf, norbAct, nelecAct)
mc.chkfile = '{}_SHCISCF.chk'.format(n)

# pi system active space
mo = mc.sort_mo_by_irrep({'Ag': 0,'Bu': 0,'Au': n,'Bg': n}, {'Ag': 3*n+1,'Bu': 3*n,'Au': 0,'Bg': 0})

# these need to be adjusted with system size
mc.fcisolver.sweep_iter = [0, 2, 4]
mc.fcisolver.sweep_epsilon = [1e-3, 1e-4, 5e-5]
mc.fcisolver.nPTiter = 0
mc.max_cycle_macro = 20
#mc.internal_rotation = True
mc.fcisolver.nPTiter = 0  # Turns off PT calculation, i.e. no PTRDM.
mc.kernel(mo)

# prep nevpt
norbFrozen = 2*n
norbCore = 6*n + 1

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
tools.fcidump.from_integrals('FCIDUMP', h1eff, eri, norbAct, nelecAct, energy_core)
