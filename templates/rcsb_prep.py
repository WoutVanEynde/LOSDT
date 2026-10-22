"""Fetch a protein-ligand complex from the RCSB PDB and prepare it for the LOSDT pipeline."""
from io import StringIO
from pathlib import Path
from typing import List, Tuple
from string import ascii_uppercase, ascii_lowercase, digits
import numpy as np
from pdbfixer import PDBFixer
from pdbfixer.pdbfixer import proteinResidues, dnaResidues, rnaResidues
from openmm.app import PDBFile, Modeller, Topology, ForceField, Simulation
from openmm import unit, CustomExternalForce, LangevinMiddleIntegrator, Platform
from rdkit import Chem
from ccd_bonds import ccd_bond_orders
from constrained_em import (FORCEFIELDS, WATER_NAMES, forcefield_parametrized, amber_forcefield, add_hydrogens, DEFAULT_MINIMIZATION_STEPS,
                            DEFAULT_TEMPERATURE, DEFAULT_TIMESTEP)

POLYMER_RESIDUES = set(proteinResidues + dnaResidues + rnaResidues)

# Biochemical cofactors (CCD codes) that stay in the receptor from: https://www.ebi.ac.uk/pdbe/api/pdb/compound/cofactors
COFACTORS = {
    # Coenzyme A
    '01A', '01K', '0ET', '1C4', '1CV', '1CZ', '1HA', '1VU', '1XE', '2CP', '2NE', '3CP', '3H9', '3HC', '4CA',
    '4CO', '8JD', '8Z2', 'ACO', 'AMX', 'BCA', 'BCO', 'BSJ', 'BYC', 'CA3', 'CA5', 'CA6', 'CA8', 'CAA', 'CAJ',
    'CAO', 'CIC', 'CMC', 'CMX', 'CO6', 'CO8', 'COA', 'COD', 'COF', 'COO', 'COT', 'COW', 'COZ', 'DCA', 'DCC',
    'FAM', 'FCX', 'FRE', 'FYN', 'GRA', 'HAX', 'HMG', 'HSC', 'HXC', 'MCA', 'MCD', 'MDE', 'MLC', 'MYA', 'NHM',
    'NHQ', 'NHW', 'NMX', 'OXK', 'S0N', 'SCA', 'SCD', 'SCO', 'SDX', 'SOP', 'T1G', 'TC6', 'WCA', 'YNC', 'ZOZ',
    # Orthoquinone residues (LTQ, TTQ, CTQ)
    '0AF', 'TOQ', 'TQQ', 'TRQ',
    # Glutathione
    '0HG', '0HH', '1JO', '1JP', '1R4', '3GC', '48T', '5AU', 'ABY', 'AHE', 'ATA', 'BOB', 'BYG', 'EPY', 'ESG',
    'GBI', 'GBP', 'GBX', 'GDN', 'GDS', 'GF5', 'GGC', 'GIP', 'GNB', 'GPR', 'GPS', 'GS8', 'GSB', 'GSF', 'GSH',
    'GSM', 'GSN', 'GSO', 'GTB', 'GTD', 'GTS', 'GTX', 'GTY', 'GVX', 'HAG', 'IBG', 'ICY', 'JM2', 'JM5', 'JM7',
    'L9X', 'LEE', 'LZ6', 'RGE', 'TGG', 'TS5', 'VWW', 'ZBF',
    # S-adenosylmethionine
    '0UM', '0XU', '0Y0', '0Y1', '0Y2', '36A', '37H', '4IK', '62X', '6NR', '76H', '76J', '76K', '76L', '76M',
    'EEM', 'K15', 'SA8', 'SAH', 'SAM', 'SFG', 'SMM', 'SX0', 'TT8',
    # Nicotinamide-adenine dinucleotide
    '0WD', '1DG', '3AA', '3CD', '6V0', '8ID', 'A3D', 'AP0', 'CND', 'DG1', 'DN4', 'EAD', 'ENA', 'LNC', 'N01',
    'NA0', 'NAD', 'NAE', 'NAI', 'NAJ', 'NAP', 'NAQ', 'NAX', 'NBD', 'NBP', 'NDC', 'NDE', 'NDO', 'NDP', 'NHD',
    'NPW', 'ODP', 'P1H', 'PAD', 'SAD', 'SAE', 'SND', 'TAD', 'TAP', 'TDT', 'TXD', 'TXE', 'TXP', 'ZID',
    # Dipyrromethane
    '18W', '29P', 'DPM',
    # Thiamine diphosphate
    '1TP', '1U0', '2TP', '5GY', '8EF', '8EL', '8EO', '8FL', '8PA', 'D7K', 'EN0', 'HTL', 'M6T', 'N1T', 'N3T',
    'R1T', 'S1T', 'T5X', 'T6F', 'TD6', 'TD7', 'TD8', 'TD9', 'TDK', 'TDL', 'TDM', 'TDP', 'TDW', 'THD', 'THV',
    'THW', 'THY', 'TP8', 'TPP', 'TPU', 'TPW', 'TZD', 'WWF',
    # Topaquinone
    '1TY', '2TY', 'AGQ', 'G27', 'HCC', 'P2Q', 'P3Q', 'TPQ', 'TYQ', 'TYY',
    # Tetrahydrofolic acid
    '1YJ', 'C2F', 'FFO', 'FON', 'FOZ', 'THF', 'THG', 'THH',
    # Molybdopterin
    '2MD', 'MCN', 'MGD', 'MSS', 'MTE', 'MTQ', 'MTV', 'PCD', 'XAX',
    # Biopterin
    '4AB', '7AP', 'BHS', 'BIO', 'H2B', 'H4B', 'HBI', 'WSD',
    # Flavin Mononucleotide
    '4LS', '4LU', 'FMN', 'FNR', 'FNS', 'IRF', 'RBF',
    # Ubiquinone
    '4YP', 'AT5', 'UQ1', 'UQ2', 'UQ5', 'UQ6',
    # Flavin adenine dinucleotide
    '6FA', 'FA8', 'FAA', 'FAB', 'FAD', 'FAE', 'FAO', 'FAS', 'FCG', 'FDA', 'FED', 'FSH', 'P5F', 'RFL', 'SFD',
    # Heme
    '6HE', '7HE', 'CCH', 'COH', 'DDH', 'DHE', 'FDE', 'FMI', 'HAS', 'HDD', 'HDE', 'HEA', 'HEB', 'HEC', 'HEM',
    'HIF', 'ISW', 'MH0', 'MNH', 'MNR', 'PP9', 'SH0', 'SRM', 'ZEM', 'ZNH',
    # Ascorbic acid
    'ASC',
    # Adenosylcobalamin
    'B12', 'CNC', 'COB', 'COY',
    # Biotin
    'BTI', 'BTN', 'BYT', 'DTB', 'Y7Y',
    # Coenzyme M
    'COM',
    # Factor F430
    'F43', 'M43',
    # Lipoic acid
    'LPA', 'LPB',
    # MIO
    'MDO',
    # Pyridoxal 5'-phosphate
    'MPL', 'NOP', 'NPL', 'PDP', 'PLP', 'PLR', 'PMP', 'PXP', 'PZP', 'UAH',
    # Menaquinone
    'MQ7',
    # Phosphopantetheine
    'PNS',
    # Pyrroloquinoline Quinone
    'PQQ',
    # Coenzyme B
    'SHT', 'TP7', 'TPZ', 'TXZ', 'XP8', 'XP9',
}

CPU = Platform.getPlatformByName('CPU')  # Same platform as constrained_em
RESTRAINT_STRENGTH = 5000.0  # kJ/mol/nm^2 on heavy atoms during the preparation EM

ACIDIC_OH = Chem.MolFromSmarts('[OX2H1]-[$([#15]=O),$([#6]=O),$([#16]=O)]')  # Phosphates, carboxylic, sulfonic acids: deprotonated at pH 7

MAX_PEPTIDE_BOND = 2.0  # Angstrom, longer C-N distances between consecutive residues are chain breaks. Slightly looser than Biopython's PPBuilder's 1.8

def split_chain_breaks(topology: Topology, positions) -> Modeller:
    """Start a new chain at every backbone break, so the pipeline caps both ends instead of bonding across the gap."""
    # Change to angstrom
    xyz = positions.value_in_unit(unit.angstrom)
    # Check free chain IDs available
    free_ids = iter(c for c in ascii_uppercase + ascii_lowercase + digits if c not in {ch.id for ch in topology.chains()})
    # Empty topology for rebuilding
    new_topology, new_atoms = Topology(), {}
    # Original chains get their old IDs
    for chain in topology.chains():
        new_chain, previous = new_topology.addChain(chain.id), None
        for residue in chain.residues():
            # Mapping atom names to atoms
            atoms = {atom.name: atom for atom in residue.atoms()}
            # Break check: is it a break or terminus, how far from each other?
            if previous and 'C' in previous and 'N' in atoms and \
                    np.linalg.norm(np.subtract(xyz[previous['C'].index], xyz[atoms['N'].index])) > MAX_PEPTIDE_BOND:
                new_chain = new_topology.addChain(next(free_ids))
            # Rechain the residues after break to correct chain ID
            new_residue = new_topology.addResidue(residue.name, new_chain, residue.id, residue.insertionCode)
            # Copy atoms into new chain ID
            for atom in residue.atoms():
                new_atoms[atom] = new_topology.addAtom(atom.name, atom.element, new_residue, atom.id)
            previous = atoms
    # Rebuild bonds
    for atom1, atom2 in topology.bonds():
        a1, a2 = new_atoms[atom1], new_atoms[atom2]
        if a1.residue.chain is a2.residue.chain or {a1.name, a2.name} != {'C', 'N'}:
            new_topology.addBond(a1, a2)
    return Modeller(new_topology, positions)

def protonate_and_relax(modeller: Modeller) -> None:
    """
    Add hydrogens, then minimize as in constrained_em with all heavy atoms restrained to their crystal positions.
    """
    add_hydrogens(modeller)

    system = amber_forcefield().createSystem(modeller.topology)
    restraint = CustomExternalForce("k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    system.addForce(restraint)
    restraint.addGlobalParameter('k', RESTRAINT_STRENGTH * unit.kilojoules_per_mole / unit.nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    # Hydrogens not added to restraint to relax
    for atom in modeller.topology.atoms():
        if atom.element.symbol != 'H':
            pos = modeller.positions[atom.index]
            restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])

    integrator = LangevinMiddleIntegrator(DEFAULT_TEMPERATURE * unit.kelvin, 1.0 / unit.picosecond, DEFAULT_TIMESTEP * unit.picoseconds)
    simulation = Simulation(modeller.topology, system, integrator, CPU)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=DEFAULT_MINIMIZATION_STEPS)
    modeller.positions = simulation.context.getState(getPositions=True).getPositions()

def hetero_records(hetero: Modeller, first_serial: int) -> Tuple[List[str], List[str]]:
    """
    HETATM and CONECT records of the hetero residues with CCD bond orders and hydrogens at pH 7 (acids deprotonated). Written by RDKit, so bond orders (repeated CONECT entries) and formal charges are kept, as in MOE prepared files. Residues not in the CCD are written as they are.
    """
    atoms, conects = [], []
    for residue in hetero.topology.residues():
        single = Modeller(hetero.topology, hetero.positions)
        single.delete([r for r in single.topology.residues() if r.index != residue.index])
        block = StringIO()
        PDBFile.writeFile(single.topology, single.positions, block, keepIds=True)
        raw = Chem.MolFromPDBBlock(block.getvalue(), removeHs=False, sanitize=False)
        mol = ccd_bond_orders(raw)
        if mol is not None:
            mol = Chem.RemoveHs(mol)
            for (oxygen, _) in mol.GetSubstructMatches(ACIDIC_OH):
                mol.GetAtomWithIdx(oxygen).SetFormalCharge(-1)
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol, addCoords=True)
            info = mol.GetAtomWithIdx(0).GetPDBResidueInfo()
            added = [atom for atom in mol.GetAtoms() if atom.GetPDBResidueInfo() is None]
            for i, atom in enumerate(added, 1):
                atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(f'H{i}', residueName=info.GetResidueName(), residueNumber=info.GetResidueNumber(), chainId=info.GetChainId(), isHeteroAtom=True))
        else:
            mol = raw
        offset = first_serial - 1
        for line in Chem.MolToPDBBlock(mol).splitlines():
            if line.startswith('HETATM'):
                atoms.append(f"{line[:6]}{int(line[6:11]) + offset:5d}{line[11:]}")
            elif line.startswith('CONECT'):
                conects.append('CONECT' + ''.join(f"{int(line[i:i + 5]) + offset:5d}" for i in range(6, len(line), 5)))
        first_serial += mol.GetNumAtoms()
    return atoms, conects

def prepare_rcsb_complex(pdb_id: str, ligand_name: str, output_path: Path) -> None:
    """
    Download pdb_id and write a cleaned complex to output_path: the first instance of ligand_name, cofactors, force field parametrized residues (ions, lipids) and crystallographic waters are kept, all other heteroatomic residues (crystallization/purification artifacts) are removed. Non-canonical residues are reverted to canonical ones, internal gaps in SEQRES are modelled and remaining chain breaks are split. Missing heavy atoms and hydrogens are added, followed by a restrained energy minimization. The ligand and cofactors get CCD bond orders and hydrogens.
    """
    fixer = PDBFixer(pdbid=pdb_id, platform=CPU)
    # Replace non standard residues by standard
    fixer.findNonstandardResidues()
    replacements = {residue.name: replacement for residue, replacement in fixer.nonstandardResidues}
    for sequence in fixer.sequences:  # So SEQRES still matches the reverted residues
        sequence.residues = [replacements.get(name, name) for name in sequence.residues]
    fixer.replaceNonstandardResidues()

    # Keep first ligand copy and cofactors (HETATMS)
    hetero = [r for r in fixer.topology.residues() if r.name not in POLYMER_RESIDUES | WATER_NAMES]
    ligands = [r for r in hetero if r.name == ligand_name]
    if not ligands:
        raise ValueError(f"Ligand {ligand_name} not found in {pdb_id}, present: {', '.join(sorted({r.name for r in hetero}))}")
    forcefield_residues = forcefield_parametrized(fixer.topology)
    keep = {ligands[0].index} | {r.index for r in hetero if r.name != ligand_name and r.name in COFACTORS}

    # Ligand and cofactors have no parameters in forcefields, they are set aside and added back at the end
    kept_hetero = Modeller(fixer.topology, fixer.positions)
    kept_hetero.delete([r for r in kept_hetero.topology.residues() if r.index not in keep])
    polymer = Modeller(fixer.topology, fixer.positions)
    polymer.delete([r for r in polymer.topology.residues()
                    if r.name not in POLYMER_RESIDUES | WATER_NAMES and r.index not in forcefield_residues])
    fixer.topology, fixer.positions = polymer.topology, polymer.positions

    # Model internal gaps known from SEQRES, not the unresolved termini. Then add missing atoms
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    def internal(chain_index, index):
        residues = list(chains[chain_index].residues())
        return 0 < index < len(residues) and {residues[index - 1].name, residues[index].name} <= POLYMER_RESIDUES
    fixer.missingResidues = {key: names for key, names in fixer.missingResidues.items() if internal(*key)}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Split the breaks SEQRES does not describe
    polymer = split_chain_breaks(fixer.topology, fixer.positions)
    fixer.topology, fixer.positions = polymer.topology, polymer.positions
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    complex_ = Modeller(fixer.topology, fixer.positions)
    protonate_and_relax(complex_)
    block = StringIO()
    PDBFile.writeFile(complex_.topology, complex_.positions, block, keepIds=True)
    lines = [line for line in block.getvalue().splitlines() if line != 'END']
    records = [line for line in lines if not line.startswith('CONECT')]
    last_serial = max(int(line[6:11]) for line in records if line.startswith(('ATOM', 'HETATM', 'TER')))
    hetero_atoms, hetero_conects = hetero_records(kept_hetero, last_serial + 1)
    with open(output_path, 'w') as f:
        f.write('\n'.join(records + hetero_atoms + [line for line in lines if line.startswith('CONECT')] + hetero_conects + ['END']) + '\n')
