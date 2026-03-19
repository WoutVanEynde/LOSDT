import os
import sys
import warnings

# Force OpenMM to use single-threaded CPU platform to prevent thread conflicts with multiprocessing https://github.com/openmm/openmm/issues/4424
os.environ["OPENMM_CPU_THREADS"] = "1"
os.environ['OPENMM_DEFAULT_PLATFORM'] = 'CPU'

warnings.simplefilter("ignore", category=FutureWarning)

# PyInstaller imports
if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    os.environ['PATH'] = bundle_dir + os.pathsep + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = bundle_dir + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

import tempfile
import traceback
import logging
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Set, Dict
from multiprocessing import Pool, cpu_count
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds, rdFMCS
from openff.toolkit.topology import Molecule
from pdbfixer import PDBFixer
from openmmforcefields.generators import SystemGenerator
from openmm.app import PDBFile, Modeller, ForceField, Simulation
from openff.toolkit import Molecule
from openff.units import unit as openff_unit
from openmm import CustomExternalForce, LangevinMiddleIntegrator, unit#, Platform
from pdbtools import pdb_selresname, pdb_tocif

# Add dimorphite_dl to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static', 'third_party_software', 'dimorphite_dl')))

from dimorphite_dl.protonate.run import protonate_smiles

# Configure logging for multiprocessing
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================
DEFAULT_RESTRAINT_RADIUS = 0.6  # Distance in nm (6 Å = 0.6 nm)
DEFAULT_RESTRAINT_STRENGTH = 1000000.0  # kJ/mol/nm² - very high to effectively freeze atoms
DEFAULT_MINIMIZATION_STEPS = 5000
DEFAULT_ENERGY_REPORT_INTERVAL = 50
DEFAULT_TEMPERATURE = 300  # Kelvin
DEFAULT_TIMESTEP = 0.002  # picoseconds

STANDARD_AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'CYM', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 
    'TRP', 'TYR', 'VAL', 'HIE', 'HIP', 'HID', 'HSD', 'HSE', 'HSP',
    'ACE', 'NME'
}

WATER_NAMES = {'HOH', 'WAT', 'TIP3', 'SOL', 'OPC'}

def fix_oxygen_formal_charges(mol):
    # FIX OXYGEN FORMAL CHARGES, used to be problem in NAD+ that was deprotonated.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 8:         # skip non-O
            continue
        # count DOUBLE bonds; Double-bonded O should have formal charge = 0
        n_double = sum(1 for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        if n_double:
            atom.SetFormalCharge(0)
            continue

        # Single-bonded O should have formal charge = -1
        for neighbor in atom.GetNeighbors():
            amount_of_neighbors = sum(1 for neighbor in atom.GetNeighbors())
            if amount_of_neighbors == 1:
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                if (bond.GetBondType() == Chem.BondType.SINGLE and
                    atom.GetFormalCharge() == 0):
                    atom.SetFormalCharge(-1)
                    break

def extract_ligands_from_pdb(pdb_path: Path, protonation: bool, ligand_residue_names: Optional[List[str]] = None) -> Dict[str, List[Molecule]]:
    """
    Extract all ligands from PDB file and assign charges to each instance.
    """
    # Load PDB file to identify ligand residues and count instances
    pdb = PDBFile(str(pdb_path))
    
    # Identify ligand residues and their instances
    ligand_residues_map = {}  # {ligand_name: [(residue, chain_id, res_id), ...]}
    
    if ligand_residue_names:
        logger.info(f"Looking for specified ligands: {', '.join(ligand_residue_names)}")
        target_names = set(ligand_residue_names)
    else:
        logger.info("Auto-detecting ligands (non-protein, non-water residues)...")
        target_names = None
    
    for residue in pdb.topology.residues():
        if (residue.name not in WATER_NAMES and residue.name not in STANDARD_AMINO_ACIDS):
            if target_names is None or residue.name in target_names:
                if residue.name not in ligand_residues_map:
                    ligand_residues_map[residue.name] = []
                ligand_residues_map[residue.name].append((residue, residue.chain.id, residue.id))
                logger.info(f"Found ligand residue: {residue.name} (Chain {residue.chain.id}, ID: {residue.id})")
    
    if not ligand_residues_map:
        raise ValueError("No ligand residues found in PDB file!")
    
    # Extract each ligand instance separately and assign charges
    ligand_molecules = []
    
    for ligand_name, residue_list in sorted(ligand_residues_map.items()):
        logger.info(f"Processing {len(residue_list)} instance(s) of ligand: {ligand_name}")
        
        for idx, (residue, chain_id, res_id) in enumerate(residue_list):
            temp_ligand_path = pdb_path.parent / f"{pdb_path.stem}_temp_{ligand_name}_{chain_id}_{res_id}.pdb"
            temp_ligand_sdf_path = pdb_path.parent / f"{pdb_path.stem}_temp_{ligand_name}_{chain_id}_{res_id}.sdf"
            
            try:
                # Extract ligand (select residues with specified names)
                with open(pdb_path, "r") as in_f, open(temp_ligand_path, "w") as out_f:
                    for line in pdb_selresname.run(in_f, ligand_name):
                        out_f.write(line)
                
                if not temp_ligand_path.exists() or temp_ligand_path.stat().st_size == 0:
                    raise ValueError(f"Could not extract ligand {ligand_name} instance {idx+1}")
                
                # Convert ligand PDB to RDKit molecule with proper bonds
                mol = Chem.MolFromPDBFile(str(temp_ligand_path), removeHs=False, sanitize=False)
                if mol is None:
                    raise ValueError(f"Could not parse ligand {ligand_name} instance {idx+1}")
                
                 # FIX OXYGEN FORMAL CHARGES, used to be problem in NAD+ that was deprotonated.
                fix_oxygen_formal_charges(mol)
                
                formal_charge = Chem.GetFormalCharge(mol)
                rdDetermineBonds.DetermineBonds(mol, charge=formal_charge, covFactor=1.15, useVdw=True) #covFactor was set to 1.15 as BCP would throw errors due to carbons being too close.
                
                if protonation is True:
                    mol_noh = Chem.RemoveAllHs(mol)
                    mol_smiles = Chem.MolToSmiles(mol_noh)
                    protonated_smiles = protonate_smiles(mol_smiles, ph_min=7.4, ph_max=7.4, precision=0.0, max_variants=1) #Changing precision from 1.0 to 0.0 would crash consistently for multiple test complexes; not sure why
                    protonated = Chem.MolFromSmiles(protonated_smiles[0])
                    fix_oxygen_formal_charges(protonated)
                    # This worked best, assignbondsfromtemplate made explicit carbons resulting in radicals instead of hydrogens in some cases
                    from align_molecules import align_and_optimize
                    align_and_optimize(protonated, mol)
                    protonated = Chem.AddHs(protonated, addCoords=True)
                    logger.info(f"Ligand {ligand_name} protonated with SMILES: {protonated_smiles}!")
                    mol = protonated
                                                    
                # Compute MMFF partial charges; not needed anymore since Sage 2.3.0
                #mmff = AllChem.MMFFGetMoleculeProperties(mol)
                
                # Extract charges (one per atom, in atom order); not needed anymore since Sage 2.3.0
                #charges = [mmff.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]

                # Format as space-separated string; not needed anymore since Sage 2.3.0
                #charges_str = " ".join(f"{charge:.6f}" for charge in charges)
                
                # Set charge and name as molecular property
                #mol.SetProp("atom.dprop.PartialCharge", charges_str); not needed anymore since Sage 2.3.0
                mol.SetProp("_Name", ligand_name)
                
                # Convert to temp sdf file and then to openff molecule
                with Chem.SDWriter(str(temp_ligand_sdf_path)) as writer:
                    writer.write(mol)
                    
                openff_mol = Molecule.from_file(str(temp_ligand_sdf_path))
                openff_mol._residue_name = ligand_name  # Preserve for OpenMM topology; this is only for molstar recognization as ligand; otherwise UNK, which is seen as polymer
                
                ligand_molecules.append(openff_mol)
                
                #logger.info(f"Instance {idx+1}: {mol.GetNumAtoms()} atoms, total charge: {sum(charges):.3f}"); not needed anymore since Sage 2.3.0
                
            except Exception as e:
                logger.error(f"Failed to extract ligand {ligand_name} instance {idx+1}: {str(e)}")
                raise
            
            finally:
                # Clean up temporary files
                if temp_ligand_path.exists():
                    os.unlink(temp_ligand_path)
                if temp_ligand_sdf_path.exists():
                    os.unlink(temp_ligand_sdf_path)
                    
    return ligand_molecules

def add_seqres_with_caps(input_pdb: str, output_pdb: str):
    """
    Add SEQRES records with ACE/NME caps to a PDB file.
    PDBFixer will then detect these as 'missing' and build them.
    """
    # First, read the structure to get chain sequences
    pdb = PDBFile(input_pdb)
    
    # Build sequence for each chain
    chain_sequences = {}
    for chain in pdb.topology.chains():
        residues = list(chain.residues())
        protein_residues = [r for r in residues if r.name in STANDARD_AMINO_ACIDS]
        
        if protein_residues:
            # Add ACE at start, NME at end
            seq = ['ACE'] + [r.name for r in protein_residues] + ['NME']
            chain_sequences[chain.id] = seq
    
    # Now write modified PDB with SEQRES records
    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        # First write SEQRES records for each chain
        for chain_id, seq in chain_sequences.items():
            # SEQRES records: max 13 residues per line
            for i in range(0, len(seq), 13):
                chunk = seq[i:i+13]
                line_num = (i // 13) + 1
                seqres_line = f"SEQRES {line_num:>3} {chain_id} {len(seq):>4}  "
                seqres_line += " ".join(f"{res:>3}" for res in chunk)
                f_out.write(seqres_line + '\n')
        
        # Then copy the rest of the file (skip existing SEQRES lines)
        for line in f_in:
            if not line.startswith('SEQRES'):
                f_out.write(line)
    
    return chain_sequences

def _write_topology_file(filepath: Path, modeller, system, mobile_atoms: Set[int], 
                         mobile_residues: Set, unit) -> None:
    """Write detailed topology information."""
    total_atoms = modeller.topology.getNumAtoms()
    
    # Collect bonds
    bonds_written = set()
    
    with open(filepath, 'w') as f:
        f.write(f"; Topology for constrained energy minimization\n")
        f.write(f"; Total atoms: {total_atoms}\n")
        f.write(f"; Mobile atoms: {len(mobile_atoms)}\n")
        f.write(f"; Restrained atoms: {total_atoms - len(mobile_atoms)}\n\n")
        
        f.write("; Mobile Residues\n")
        for residue in sorted(mobile_residues, key=lambda r: (r.chain.id, r.id)):
            f.write(f"; Chain {residue.chain.id}, Residue {residue.id} ({residue.name})\n")
        
        f.write("\n; Atoms\n")
        f.write("; Index  Name  Residue  Chain  Status\n")
        for atom in modeller.topology.atoms():
            status = "MOBILE" if atom.index in mobile_atoms else "RESTRAINED"
            f.write(f"{atom.index:>6}  {atom.name:<4}  {atom.residue.name:>3}  "
                   f"{atom.residue.chain.id}      {status}\n")
        
        f.write("\n; Bonds\n")
        f.write("; Atom1  Atom2\n")
        for bond in modeller.topology.bonds():
            a1, a2 = bond.atom1.index, bond.atom2.index
            if (a1, a2) not in bonds_written and (a2, a1) not in bonds_written:
                f.write(f"{a1:>6}  {a2:>6}\n")
                bonds_written.add((a1, a2))
        
        f.write("\n; Summary\n")
        f.write(f"; Total atoms: {total_atoms}\n")
        f.write(f"; Mobile atoms: {len(mobile_atoms)}\n")
        f.write(f"; Restrained atoms: {total_atoms - len(mobile_atoms)}\n")
        f.write(f"; Mobile residues: {len(mobile_residues)}\n")
        f.write(f"; Total bonds: {len(bonds_written)}\n")


def _write_region_pdbs(output_dir: Path, output_prefix: str, modeller, positions, 
                       mobile_atoms: Set[int], unit) -> None:
    """Write separate PDB files for mobile and restrained regions."""
    # Mobile region
    with open(output_dir / f'{output_prefix}_mobile.pdb', 'w') as f:
        atom_num = 1
        for atom in modeller.topology.atoms():
            if atom.index in mobile_atoms:
                pos = positions[atom.index]
                x = pos[0] / unit.angstrom
                y = pos[1] / unit.angstrom
                z = pos[2] / unit.angstrom
                f.write(f"ATOM  {atom_num:>5} {atom.name:<4} {atom.residue.name:>3} "
                       f"{atom.residue.chain.id}{atom.residue.id:>4}    "
                       f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          "
                       f"{atom.element.symbol:>2}\n")
                atom_num += 1
        f.write("END\n")
    
    # Restrained region
    with open(output_dir / f'{output_prefix}_restrained.pdb', 'w') as f:
        atom_num = 1
        for atom in modeller.topology.atoms():
            if atom.index not in mobile_atoms:
                pos = positions[atom.index]
                x = pos[0] / unit.angstrom
                y = pos[1] / unit.angstrom
                z = pos[2] / unit.angstrom
                f.write(f"ATOM  {atom_num:>5} {atom.name:<4} {atom.residue.name:>3} "
                       f"{atom.residue.chain.id}{atom.residue.id:>4}    "
                       f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          "
                       f"{atom.element.symbol:>2}\n")
                atom_num += 1
        f.write("END\n")

def run_constrained_em_single(
    pdb_path: Path,
    output_dir: Path,
    protonation: bool,
    ligand_residue_names: Optional[List[str]] = None,
    restraint_radius: float = DEFAULT_RESTRAINT_RADIUS,
    restraint_strength: float = DEFAULT_RESTRAINT_STRENGTH,
    minimization_steps: int = DEFAULT_MINIMIZATION_STEPS,
    temperature: float = DEFAULT_TEMPERATURE,
    timestep: float = DEFAULT_TIMESTEP,
    verbose: bool = True
) -> dict:
    """
    SIMPLIFIED WORKFLOW:
    1. Extract ligands and parameterize (adds H atoms with 3D coords via RDKit)
    2. Prepare protein-only structure with PDBFixer
    3. Add ligands back using coordinates from parameterized molecules
    4. Run constrained EM
    
    Note: Ligand positions come from the parameterized molecules (post-H addition)
    rather than the original PDB, since AddHs generates proper 3D coordinates.
    """
    output_prefix = pdb_path.stem
    result = {
        'input_file': str(pdb_path),
        'output_prefix': output_prefix,
        'success': False,
        'error': None,
        'ligands_processed': [],
        'initial_energy': None,
        'final_energy': None,
        'mobile_atoms': 0,
        'restrained_atoms': 0
    }
    
    try:
        # ====================================================================
        # STEP 1: Extract ligands with proper parameterization
        # ====================================================================
        if verbose:
            logger.info("STEP 1: Extracting and parameterizing ligands...")
        
        # Extract and parameterize ligands (includes H addition with coordinates)
        ligand_molecules = extract_ligands_from_pdb(pdb_path, protonation, ligand_residue_names)
        
        # ====================================================================
        # STEP 2: Create protein-only structure and run PDBFixer
        # ====================================================================
        if verbose:
            logger.info("STEP 2: Preparing the termini and hydrogens of the protein...")
        
        # Load PDB
        pdb = PDBFile(str(pdb_path))

        # Remove protein hydrogens using Modeller
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            PDBFile.writeFile(pdb.topology, pdb.positions, tmp)
            tmp_path = tmp.name

        # Load with Modeller to remove protein hydrogens
        modeller = Modeller(pdb.topology, pdb.positions)
        toDelete = []
        for atom in modeller.topology.atoms():
            if atom.element.symbol == 'H' and atom.residue.name in STANDARD_AMINO_ACIDS:
                toDelete.append(atom)
            elif atom.residue.name not in STANDARD_AMINO_ACIDS and atom.residue.name not in WATER_NAMES:
                toDelete.append(atom)
        if toDelete:
            modeller.delete(toDelete)

        # Save deprotonated amino-acids PDB
        protein_only_path = tmp_path.replace('.pdb', '_no_protein_H.pdb')
        with open(protein_only_path, 'w') as f:
            PDBFile.writeFile(modeller.topology, modeller.positions, f)

        # Add SEQRES with caps
        tmp_with_seqres = protein_only_path.replace('.pdb', '_seqres.pdb')
        chain_seqs = add_seqres_with_caps(protein_only_path, tmp_with_seqres)
        if verbose:
            for chain_id, seq in chain_seqs.items():
                logger.info(f"Chain {chain_id}: SEQRES with {len(seq)} residues")

        # Run PDBFixer on protein-only structure
        fixer = PDBFixer(filename=tmp_with_seqres)
        fixer.findMissingResidues()
        if verbose:
            logger.info(f"Missing residues: {len(fixer.missingResidues)}")
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        
        # Create modeller from fixed protein
        logging.getLogger("openff").setLevel(logging.ERROR) #Otherwise a lot of partial charge assigned notifications
        modeller = Modeller(fixer.topology, fixer.positions)
        
        # ====================================================================
        # STEP 3: Add ligands back with proper parameters
        # ====================================================================
        if verbose:
            logger.info("STEP 3: Adding ligands with OpenFF parameters...")
        
        logger.info(f"Ligand molecules: {ligand_molecules}")
        
        # Add each ligand back to the modeller
        for ligand_mol in ligand_molecules:
            # Get ligand topology and positions from OpenFF molecule
            ligand_topology = ligand_mol.to_topology().to_openmm()
            
            # Fix atom names: OpenFF's to_openmm() appends an 'x' suffix to atom names (e.g. C1 -> C1x) for uniqueness. Strip it so the original atom names are preserved in the output PDB/CIF. Otherwise visiblity problems in molstar later on.
            original_name = getattr(ligand_mol, '_residue_name', None)
            for atom in ligand_topology.atoms():
                if atom.name.endswith('x'):
                    atom.name = atom.name[:-1]  
                if original_name:
                    atom.residue.name = original_name          
            
            # Convert OpenFF positions to OpenMM unit system
            ligand_positions = ligand_mol.conformers[0].to_openmm()
            
            # Add ligand to modeller
            modeller.add(ligand_topology, ligand_positions)   

        system_generator = SystemGenerator(
            forcefields=['amber19/protein.ff19SB.xml', 'amber19/DNA.OL21.xml', 'amber19/lipid21.xml', 'amber19/opc3.xml'], # IMPLICIT WATER MODEL ADDED https://github.com/openmm/openmm/issues/3364
            small_molecule_forcefield='openff-2.3.0',
            molecules=ligand_molecules,
            cache=str(output_dir / f'{output_prefix}_db.json')
        )
        
        system = system_generator.create_system(modeller.topology)
        
        # ====================================================================
        # STEP 4: Identify mobile region around ligands
        # ====================================================================
        if verbose:
            logger.info("STEP 4: Identifying mobile region...")
        
        # Find all ligand atoms
        ligand_atom_indices = set()
        for residue in modeller.topology.residues():
            if (residue.name not in WATER_NAMES and residue.name not in STANDARD_AMINO_ACIDS):
                for atom in residue.atoms():
                    ligand_atom_indices.add(atom.index)
        
        positions = modeller.positions
        mobile_residues: Set = set()
        mobile_atoms: Set[int] = set(ligand_atom_indices)
        
        for residue in modeller.topology.residues():
            if any(atom.index in ligand_atom_indices for atom in residue.atoms()):
                mobile_residues.add(residue)
                continue
            
            residue_is_mobile = False
            for atom in residue.atoms():
                atom_pos = np.array(positions[atom.index].value_in_unit(unit.nanometer))
                for lig_idx in ligand_atom_indices:
                    lig_pos = np.array(positions[lig_idx].value_in_unit(unit.nanometer))
                    distance = np.linalg.norm(atom_pos - lig_pos)
                    if distance <= restraint_radius:
                        residue_is_mobile = True
                        break
                if residue_is_mobile:
                    mobile_residues.add(residue)
                    break
                    
        for residue in mobile_residues:
            for atom in residue.atoms():
                mobile_atoms.add(atom.index)                    
        
        result['mobile_atoms'] = len(mobile_atoms)
        result['restrained_atoms'] = modeller.topology.getNumAtoms() - len(mobile_atoms)
        
        total_atoms = modeller.topology.getNumAtoms()
        restrained_atoms = total_atoms - len(mobile_atoms)      
        restrained_residues = sum(1 for r in modeller.topology.residues() if r not in mobile_residues)          
        
        if verbose:
            logger.info(f"Mobile atoms: {len(mobile_atoms)}")
            logger.info(f"Restrained atoms: {result['restrained_atoms']}")
        
        # ====================================================================
        # STEP 5: Add restraints to non-mobile atoms
        # ====================================================================
        if verbose:
            logger.info("STEP 5: Adding positional restraints...")
        
        restraint = CustomExternalForce("k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        system.addForce(restraint)        
        restraint.addGlobalParameter('k', restraint_strength * unit.kilojoules_per_mole / unit.nanometer**2)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        
        for atom in modeller.topology.atoms():
            if atom.index not in mobile_atoms:
                pos = positions[atom.index]
                restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
                
        # Create topology file
        _write_topology_file(
            output_dir / f'{output_prefix}_topology.txt',
            modeller, system, mobile_atoms, mobile_residues, unit
        )
        
        if verbose:
            # Create region PDB files
            _write_region_pdbs(
                output_dir, output_prefix, modeller, positions, mobile_atoms, unit
            )        
                
        # ====================================================================
        # STEP 6: Run energy minimization
        # ====================================================================
        if verbose:
            logger.info("STEP 6: Running constrained energy minimization...")
        
        integrator = LangevinMiddleIntegrator(
            temperature * unit.kelvin,
            1.0 / unit.picosecond, # friction coefficient
            timestep * unit.picoseconds
        )
        
        simulation = Simulation(modeller.topology, system, integrator)
        simulation.context.setPositions(modeller.positions)
        
        # Get initial energy
        state = simulation.context.getState(getEnergy=True)
        result['initial_energy'] = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        
        if verbose:
            logger.info(f"Initial energy: {result['initial_energy']:.2f} kJ/mol")
        
        # Minimize
        simulation.minimizeEnergy(maxIterations=minimization_steps)
        
        # Get final energy
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        result['final_energy'] = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        
        if verbose:
            logger.info(f"Final energy: {result['final_energy']:.2f} kJ/mol")
            logger.info(f"Energy change: {result['final_energy'] - result['initial_energy']:.2f} kJ/mol")
        
        # ====================================================================
        # STEP 7: Save outputs
        # ====================================================================
        if verbose:
            logger.info("STEP 7: Saving outputs...")
        
        minimized_positions = state.getPositions()
        
        # Save minimized structure as pdb and cif
        output_pdb_path = output_dir / f'{output_prefix}_minimized.pdb'
        complex_cif_path = output_dir / f'{output_prefix}_minimized.cif'
        
        with open(output_pdb_path, 'w') as f:
            PDBFile.writeFile(modeller.topology, minimized_positions, f)
        with open(output_pdb_path, 'r') as in_f, open(complex_cif_path, 'w') as out_f:
            for line in pdb_tocif.run(in_f):
                out_f.write(line)
        
        result['success'] = True
        logger.info(f"Success: {output_prefix}")
        
    except Exception as e:
        result['error'] = traceback.format_exc()
        logger.error(f"Failed: {pdb_path.name}")
        logger.error(f"Error: {str(e)}")
    
    return result


def process_single_pdb_wrapper(args):
    """
    Wrapper function for parallel processing.
    Unpacks arguments and calls run_constrained_em_single.
    """
    pdb_path, output_dir, params = args
    return run_constrained_em_single(
        pdb_path=pdb_path,
        output_dir=output_dir,
        protonation=params['protonation'],
        ligand_residue_names=params['ligand_residue_names'],
        restraint_radius=params['restraint_radius'],
        restraint_strength=params['restraint_strength'],
        minimization_steps=params['minimization_steps'],
        temperature=params['temperature'],
        timestep=params['timestep'],
        verbose=params['verbose']
    )


def constrained_em_main(
    pdb_dir: str,
    output_dir: str,
    protonation: bool,
    ligand_residue_names: Optional[List[str]] = None,
    restraint_radius: float = DEFAULT_RESTRAINT_RADIUS,
    restraint_strength: float = DEFAULT_RESTRAINT_STRENGTH,
    minimization_steps: int = DEFAULT_MINIMIZATION_STEPS,
    temperature: float = DEFAULT_TEMPERATURE,
    timestep: float = DEFAULT_TIMESTEP,
    verbose: bool = True,
    processes: Optional[int] = None
) -> List[dict]:
    """
    Run constrained energy minimization on multiple protein-ligand complexes in parallel.
    """
    pdb_dir = Path(pdb_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDB files
    logger.info(f"Scanning for PDB files in: {pdb_dir}")
    pdb_files = list(pdb_dir.glob('*.pdb'))
    
    if not pdb_files:
        raise ValueError(f"No PDB files found in {pdb_dir}")
    
    logger.info(f"Found {len(pdb_files)} PDB files")
    
    # Determine number of workers
    if processes is None:
        processes = max(1, cpu_count() - 4)
    
    logger.info(f"Using {processes} parallel workers")
    
    # Package parameters
    params = {
        'protonation': protonation,
        'ligand_residue_names': ligand_residue_names,
        'restraint_radius': restraint_radius,
        'restraint_strength': restraint_strength,
        'minimization_steps': minimization_steps,
        'temperature': temperature,
        'timestep': timestep,
        'verbose': verbose
    }
    
    # Prepare arguments for parallel processing
    args_list = [(pdb_path, output_dir, params) for pdb_path in pdb_files]
    
    # Parallel processing
    logger.info("=" * 60)
    logger.info("STARTING PARALLEL PROCESSING")
    logger.info("=" * 60)
    
    with Pool(processes=processes) as pool:
        results = pool.map(process_single_pdb_wrapper, args_list)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if failed > 0:
        logger.info("Failed jobs:")
        for r in results:
            if not r['success']:
                logger.info(f"{r['output_prefix']}")
    
    # Save summary JSON
    summary_path = output_dir / 'em_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")
    
    return results

def main(args):
    """CLI wrapper for constrained_em_main."""
    try:
        constrained_em_main(
            pdb_dir=args.pdb,
            output_dir=args.output,
            protonation=args.protonation,
            ligand_residue_names=args.ligand,
            restraint_radius=args.restraint_radius,
            restraint_strength=args.restraint_strength,
            minimization_steps=args.minimization_steps,
            temperature=args.temperature,
            timestep=args.timestep,
            verbose=not args.quiet,
            processes=args.processes
        )
        return 0
    except Exception:
        return 1

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(
        description='Constrained energy minimization for protein-ligand complexes.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )  
    parser.add_argument('--pdb', '-p', required=True, type=str,
                       help='Directory containing complex PDB files.')
    parser.add_argument('--output', '-o', required=True, type=str,
                       help='Directory for output files.')
    parser.add_argument('--ligand', '-l', type=str, nargs='+', default=None,
                       help='Ligand residue name (e.g. LIG, NAD). Auto-detect if not provided.')
    parser.add_argument('--protonation', '-prot', action='store_true',
                       help='Protonation of ligands (default=True).')                       
    parser.add_argument('--restraint-radius', '-rr', type=float, default=DEFAULT_RESTRAINT_RADIUS,
                       help='Mobile region radius in nm.')
    parser.add_argument('--restraint-strength', '-rs', type=float, default=DEFAULT_RESTRAINT_STRENGTH,
                       help='Restraint force constant (kJ/mol/nm²).')
    parser.add_argument('--minimization-steps', '-ms', type=int, default=DEFAULT_MINIMIZATION_STEPS,
                       help='Number of minimization steps.')
    parser.add_argument('--temperature', '-t', type=float, default=DEFAULT_TEMPERATURE,
                       help='Temperature in Kelvin.')
    parser.add_argument('--timestep', '-ts', type=float, default=DEFAULT_TIMESTEP,
                       help='Integration timestep in picoseconds.')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of processes to use for parallel processing (default: cpu_count - 4).')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress verbose output.')
    
    args = parser.parse_args()
    exit(main(args))
