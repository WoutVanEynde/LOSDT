"""Bond orders of PDB heteroatomic residues from the wwPDB Chemical Component Dictionary (CCD)."""
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional
from rdkit import Chem

CCD_PATH = Path(__file__).resolve().parent.parent / 'static' / 'ccd' / 'components.pkl'
BOND_TYPES = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}

@lru_cache(maxsize=1)
def _ccd() -> dict:
    # {residue name: (atom names, elements, formal charges, ((i, j, order), ...), smiles)}, heavy atoms only
    with open(CCD_PATH, 'rb') as f:
        return pickle.load(f)

def _charge_from_valence(atom: Chem.Atom) -> int:
    """Formal charge implied by the explicit valence of an atom whose hydrogens are all present."""
    valence = int(sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds()))
    allowed = Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum())
    if atom.GetAtomicNum() == 6 or allowed[0] < 0 or valence in allowed:
        return 0 # carbon rarely has +/-1 charge, so force 0 charge
    return valence - allowed[0]

def ccd_bond_orders(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Rebuild the heavy atom bonds of a single PDB residue (parsed with sanitize=False) from its CCD entry, matched on atom names. Formal charges come from the CCD when the residue has no hydrogens, otherwise from the valence, so a different protonation state is respected. Missing hydrogens are added with coordinates. Returns None when the residue is not in the CCD or its heavy atoms do not match the entry.
    """
    info = mol.GetAtomWithIdx(0).GetPDBResidueInfo()
    entry = _ccd().get(info.GetResidueName().strip()) if info else None
    if entry is None:
        return None
    names, elements, charges, bonds, _ = entry

    # Match heavy atoms by name using dictionary
    heavy = {atom.GetPDBResidueInfo().GetName().strip(): atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}
    if set(heavy) != set(names) or any(heavy[n].GetSymbol().upper() != e.upper() for n, e in zip(names, elements)):
        return None

    # Clear existing bonds from the PDB, with exception of heavy atom-hydrogen bond
    rwmol = Chem.RWMol(mol)
    for bond in list(rwmol.GetBonds()):
        if bond.GetBeginAtom().GetAtomicNum() > 1 and bond.GetEndAtom().GetAtomicNum() > 1:
            rwmol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        else:
            bond.SetBondType(Chem.BondType.SINGLE)

    # Add in CCD bonds
    index = [heavy[n].GetIdx() for n in names]
    for i, j, order in bonds:
        rwmol.AddBond(index[i], index[j], BOND_TYPES[order])

    # Handling hydrogens, assign False/True on atoms with hydrogens. In case is True, these atoms dont get implicit hydrogens
    has_hydrogens = any(atom.GetAtomicNum() == 1 for atom in rwmol.GetAtoms())
    for atom in rwmol.GetAtoms():
        atom.SetNoImplicit(has_hydrogens)

    # Assign formal charges. Uses False/True from handling hydrogens, if no hydrogens use charge from CCD, otherwise valency helper.
    for name, charge in zip(names, charges):
        atom = rwmol.GetAtomWithIdx(heavy[name].GetIdx())
        atom.SetFormalCharge(_charge_from_valence(atom) if has_hydrogens else charge)

    try:
        Chem.SanitizeMol(rwmol)
    except Exception:
        return None
    mol = Chem.AddHs(rwmol.GetMol(), addCoords=True)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol
