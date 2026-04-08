import os
import sys

# The xtb quantum chemistry software uses OpenMP internally for parallelization. When combined: Python's multiprocessing.Pool (process-level parallelism), xtb's internal OpenMP (thread-level parallelism), and Intel's KMP OpenMP runtime, nested conflicts emerge that deadlocks everything. Solution is to force threads to 1 as we are only scoring anyways.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# PyInstaller imports
if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    os.environ['PATH'] = bundle_dir + os.pathsep + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = bundle_dir + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

import time
import logging
import traceback
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from multiprocessing import Pool, cpu_count
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS
RDLogger.DisableLog('rdApp.*')

# Add shepherd score to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static', 'third_party_software', 'shepherd-score')))

from shepherd_score.score.constants import ALPHA
from shepherd_score.conformer_generation import charges_from_single_point_conformer_with_xtb
from shepherd_score.container import Molecule, MoleculePair

# Configure logging for multiprocessing
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration constants for molecular alignment."""
    TIMEOUT = 15  # seconds
    MCS_TIMEOUT = 90  # seconds
    SHEPHERD_SURFACE_POINTS = 200
    SHEPHERD_LEARNING_RATE = 0.03
    SHEPHERD_MAX_STEPS = 0
    SHEPHERD_REPEATS = 1

# =============================================================================
# MOLECULE LOADING AND SAVING
# =============================================================================

def load_molecule_from_sdf(sdf_file: Path) -> Chem.Mol:
    """Load molecule from SDF file and generate 3D coordinates if missing."""
    try:
        sdf_supplier = Chem.SDMolSupplier(str(sdf_file))
        mol = sdf_supplier[0]
        if mol is None:
            raise ValueError(f"Failed to load template molecule from {sdf_file}")
        
        # Convert to SMILES and back to get consistent atom ordering
        smiles = Chem.MolToSmiles(mol)
        logger.info(f"Loaded molecule SMILES: {smiles}")
        new_mol = Chem.MolFromSmiles(smiles)
        
        # Transfer coordinates using MCS
        new_mol = _transfer_coordinates_via_mcs(mol, new_mol)
        Chem.SetAromaticity(new_mol, Chem.AromaticityModel.AROMATICITY_MDL)
        
        return new_mol
    except Exception as e:
        logger.error(f"Error loading molecule from {sdf_file}: {e}")
        raise

def _transfer_coordinates_via_mcs(original_mol: Chem.Mol, new_mol: Chem.Mol) -> Chem.Mol:
    """Transfer coordinates from original to new molecule using MCS."""
    mcs = rdFMCS.FindMCS([original_mol, new_mol], 
                        bondCompare=rdFMCS.BondCompare.CompareOrder,
                        atomCompare=rdFMCS.AtomCompare.CompareElements,
                        matchValences=True)
    
    pattern = Chem.MolFromSmarts(mcs.smartsString)
    matches_orig = original_mol.GetSubstructMatches(pattern)
    matches_new = new_mol.GetSubstructMatches(pattern)
    
    if not matches_orig or not matches_new:
        logger.warning("No MCS found for coordinate transfer")
        return new_mol
    
    # Use first match
    match_orig = matches_orig[0]
    match_new = matches_new[0]
    
    # Transfer coordinates
    conf_orig = original_mol.GetConformer()
    conf_new = Chem.Conformer(new_mol.GetNumAtoms())
    
    atom_map = {new: orig for new, orig in zip(match_new, match_orig)}
    
    for new_idx in range(new_mol.GetNumAtoms()):
        if new_idx in atom_map:
            orig_pos = conf_orig.GetAtomPosition(atom_map[new_idx])
            conf_new.SetAtomPosition(new_idx, orig_pos)
        else:
            conf_new.SetAtomPosition(new_idx, (0.0, 0.0, 0.0))
    
    new_mol.AddConformer(conf_new)
    return new_mol

def save_molecule_to_sdf(molecule: Chem.Mol, output_file: Path) -> None:
    """Save molecule to SDF file with MMFF partial charges."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute MMFF partial charges
    mmff = AllChem.MMFFGetMoleculeProperties(molecule)
    
    # Extract charges (one per atom, in atom order)
    charges = [mmff.GetMMFFPartialCharge(i) for i in range(molecule.GetNumAtoms())]
    
    # Format as space-separated string
    charges_str = " ".join(f"{charge:.6f}" for charge in charges)
    
    # Set as molecular property for SDF output
    molecule.SetProp("atom.dprop.PartialCharge", charges_str)
    
    with Chem.SDWriter(str(output_file)) as writer:
        writer.write(molecule)

# =============================================================================
# MCS FINDING AND ALIGNMENT
# =============================================================================

def find_mcs_with_params(mol1: Chem.Mol, mol2: Chem.Mol,
                        atom_compare=rdFMCS.AtomCompare.CompareAny, 
                        bond_compare=rdFMCS.BondCompare.CompareOrder,
                        ring_matches_ring_only=True,
                        complete_rings_only=True) -> Tuple[Optional[Chem.Mol], List[int], List[int]]:
    """Find MCS between two molecules with specified parameters."""
    try:
        mcs_result = rdFMCS.FindMCS([mol1, mol2],
                                   atomCompare=atom_compare,
                                   bondCompare=bond_compare,
                                   ringMatchesRingOnly=ring_matches_ring_only,
                                   completeRingsOnly=complete_rings_only,
                                   timeout=Config.MCS_TIMEOUT)
        
        if not mcs_result or not mcs_result.smartsString:
            return None, [], []
        
        mcs = Chem.MolFromSmarts(mcs_result.smartsString)
        mol1_match = mol1.GetSubstructMatch(mcs)
        mol2_match = mol2.GetSubstructMatch(mcs)
        #print("SMARTS:", Chem.MolToSmarts(mcs))
        
        if Chem.MolToSmarts(mcs) == "[0*]": #No single hydrogens
            return None, [], []
        
        return mcs, list(mol1_match), list(mol2_match)
    
    except Exception as e:
        logger.warning(f"Error finding MCS: {e}")
        return None, [], []

def prepare_molecule_for_atom_removal(mol: Chem.Mol) -> Chem.RWMol:
    """Prepare molecule for atom removal by clearing aromatic flags."""
    mol_copy = Chem.RWMol(mol)
    
    # Clear aromatic flags
    for atom in mol_copy.GetAtoms():
        atom.SetIsAromatic(False)
    
    for bond in mol_copy.GetBonds():
        bond.SetIsAromatic(False)
        if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
    
    return mol_copy

def remove_atoms_and_find_mcs(mol1: Chem.Mol, mol2: Chem.Mol, 
                             atoms_to_remove1: List[int], 
                             atoms_to_remove2: List[int],
                             atom_compare=rdFMCS.AtomCompare.CompareAny,
                             bond_compare=rdFMCS.BondCompare.CompareOrder,
                             ring_matches_ring_only=True,
                             complete_rings_only=True) -> Tuple[List[int], List[int]]:
    """Remove atoms from molecules and find MCS in remaining structure."""
    # Prepare molecules
    mol1_rem = prepare_molecule_for_atom_removal(mol1)
    mol2_rem = prepare_molecule_for_atom_removal(mol2)
    
    # Remove atoms in reverse order
    for idx in sorted(atoms_to_remove1, reverse=True):
        mol1_rem.RemoveAtom(idx)
    for idx in sorted(atoms_to_remove2, reverse=True):
        mol2_rem.RemoveAtom(idx)
    
    # Convert back to molecules
    mol1_copy = mol1_rem.GetMol()
    mol2_copy = mol2_rem.GetMol()
    
    # Sanitize if possible
    try:
        Chem.SanitizeMol(mol1_copy)
        Chem.SanitizeMol(mol2_copy)
    except Exception as e:
        logger.warning(f"Sanitization failed: {e}")
    
    # Find MCS
    _, mol1_match, mol2_match = find_mcs_with_params(
        mol1_copy, mol2_copy, atom_compare, bond_compare, ring_matches_ring_only, complete_rings_only
    )
    
    if not mol1_match or not mol2_match:
        return [], []
    
    # Map back to original indices
    mol1_idx_map = _create_index_mapping(mol1.GetNumAtoms(), atoms_to_remove1)
    mol2_idx_map = _create_index_mapping(mol2.GetNumAtoms(), atoms_to_remove2)
    
    mapped_mol1_match = [mol1_idx_map[idx] for idx in mol1_match]
    mapped_mol2_match = [mol2_idx_map[idx] for idx in mol2_match]
    
    return mapped_mol1_match, mapped_mol2_match

def _create_index_mapping(num_atoms: int, atoms_to_remove: List[int]) -> Dict[int, int]:
    """Create mapping from new indices to original indices."""
    mapping = {}
    current_idx = 0
    for i in range(num_atoms):
        if i not in atoms_to_remove:
            mapping[current_idx] = i
            current_idx += 1
    return mapping

def remove_1xx_terminal_atoms_paired(mol, tpl, mol_atoms, tpl_atoms):
    """
    Remove atoms from MCS that are terminal and part of ring. This is due to how complete_rings feature works, it takes the bonds of atoms, this changes it to only take ring atoms that are complete.
    """
    mol_atoms = list(mol_atoms)
    tpl_atoms = list(tpl_atoms)

    # Isotope assigner
    isotope_assigner(mol)
    isotope_assigner(tpl)
    
    changed = True
    while changed:
        changed = False
        indices_to_remove = []
        
        for i, (mol_idx, tpl_idx) in enumerate(zip(mol_atoms, tpl_atoms)):
            # Check molecule atom
            mol_atom = mol.GetAtomWithIdx(mol_idx)
            mol_isotope = mol_atom.GetIsotope()
            mol_is_1xx_terminal = False
            
            if 100 <= mol_isotope:
                mol_neighbors_in_mcs = sum(1 for neighbor in mol_atom.GetNeighbors() 
                                           if neighbor.GetIdx() in mol_atoms)
                if mol_neighbors_in_mcs <= 1:
                    mol_is_1xx_terminal = True
            
            # Check template atom
            tpl_atom = tpl.GetAtomWithIdx(tpl_idx)
            tpl_isotope = tpl_atom.GetIsotope()
            tpl_is_1xx_terminal = False
            
            if 100 <= tpl_isotope:
                tpl_neighbors_in_mcs = sum(1 for neighbor in tpl_atom.GetNeighbors() 
                                           if neighbor.GetIdx() in tpl_atoms)
                if tpl_neighbors_in_mcs <= 1:
                    tpl_is_1xx_terminal = True
            
            # Remove pair if either is a 1xx terminal atom
            if mol_is_1xx_terminal or tpl_is_1xx_terminal:
                indices_to_remove.append(i)
                changed = True
        
        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            del mol_atoms[i]
            del tpl_atoms[i]
    
    return mol_atoms, tpl_atoms

def isotope_assigner(mol):
    """
    Assign isotope indicating whether atom is in ring, aromatic, has exocylic neighbours or is terminal.
    """
    ring_info = mol.GetRingInfo()
    for atom in mol.GetAtoms():
        exocyclic_info = 0
        terminal_atom = 0
        terminal_hydrogen = 0
        if atom.IsInRing():
            for neighbor in atom.GetNeighbors():
                # If this atom is in a ring, check if neighbor is in a different ring
                if neighbor.IsInRing():
                    # Check if they share a ring
                    shares_ring = any(atom.GetIdx() in ring and neighbor.GetIdx() in ring for ring in ring_info.AtomRings())
                    if not shares_ring:
                        # Count number of rings the neighbor belongs to; In other words, for every 'neighbor' that is bonded to 'atom' and in the same ring(s) as 'atom', it adds the total number of rings that neighbor belongs to to exocyclic_info.
                        neighbor_ring_count = sum(1 for ring in ring_info.AtomRings() if neighbor.GetIdx() in ring)
                        exocyclic_info += neighbor_ring_count
                if not neighbor.IsInRing() and neighbor.GetAtomicNum() != 1:
                    exocyclic_info += 1
        if not atom.IsInRing() and not atom.GetAtomicNum() == 1:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    terminal_hydrogen += 1
            terminal_heavy_atom = atom.GetDegree() - terminal_hydrogen
            if terminal_heavy_atom == 1:
                terminal_atom += 1
        value = 100*atom.IsInRing() + 10*atom.GetIsAromatic() + exocyclic_info + terminal_atom
        atom.SetIsotope(value)

# =============================================================================
# MOLECULAR ALIGNMENT
# =============================================================================

def find_multiple_mcs_alignments(molecule: Chem.Mol, template: Chem.Mol) -> List[Tuple[List[int], List[int]]]:
    """Find multiple MCS alignments between molecule and template."""

    # Isotope assigner back to only rings and aromatic.
    isotope_assigner(molecule)
    isotope_assigner(template)
    
    alignments = []
    
    # First MCS - strict parameters
    _, mol_match1, tpl_match1 = find_mcs_with_params(
        molecule, template,
        atom_compare=rdFMCS.AtomCompare.CompareElements,
        bond_compare=rdFMCS.BondCompare.CompareOrder,
        ring_matches_ring_only=True,
        complete_rings_only=False
    )
    
    if mol_match1 and tpl_match1:
        # Remove 1xx terminal atoms from first MCS
        mol_match1, tpl_match1 = remove_1xx_terminal_atoms_paired(molecule, template, mol_match1, tpl_match1)
        
        alignments.append((mol_match1, tpl_match1))
        
        # Second MCS - remove first MCS atoms
        mol_match2, tpl_match2 = remove_atoms_and_find_mcs(
            molecule, template, mol_match1, tpl_match1,
            atom_compare=rdFMCS.AtomCompare.CompareElements,
            bond_compare=rdFMCS.BondCompare.CompareOrder,
            ring_matches_ring_only=True,
            complete_rings_only=False
        )
                    
        if mol_match2 and tpl_match2:
            # Remove 1xx terminal atoms from second MCS
            mol_match2, tpl_match2 = remove_1xx_terminal_atoms_paired(molecule, template, mol_match2, tpl_match2)

            if mol_match2 and tpl_match2:
                alignments.append((mol_match2, tpl_match2))
                
                # Third MCS - remove first and second MCS atoms
                all_mol_matches = mol_match1 + mol_match2
                all_tpl_matches = tpl_match1 + tpl_match2
                
                mol_match3, tpl_match3 = remove_atoms_and_find_mcs(
                    molecule, template, all_mol_matches, all_tpl_matches,
                    atom_compare=rdFMCS.AtomCompare.CompareIsotopes,
                    bond_compare=rdFMCS.BondCompare.CompareAny,
                    ring_matches_ring_only=False,
                    complete_rings_only=False
                )
                
                if mol_match3 and tpl_match3:
                    mol_match3, tpl_match3 = remove_1xx_terminal_atoms_paired(molecule, template, mol_match3, tpl_match3)
                    
                    if mol_match3 and tpl_match3:
                            alignments.append((mol_match3, tpl_match3))
                            
                            # Fourth MCS - remove first and second MCS atoms
                            all_mol_matches = mol_match1 + mol_match2 + mol_match3
                            all_tpl_matches = tpl_match1 + tpl_match2 + tpl_match3
                            
                            mol_match4, tpl_match4 = remove_atoms_and_find_mcs(
                                molecule, template, all_mol_matches, all_tpl_matches,
                                atom_compare=rdFMCS.AtomCompare.CompareIsotopes,
                                bond_compare=rdFMCS.BondCompare.CompareAny,
                                ring_matches_ring_only=False,
                                complete_rings_only=False
                            )
                            
                            if mol_match4 and tpl_match4:
                                alignments.append((mol_match4, tpl_match4))
                                
            else:
                mol_match2, tpl_match2 = remove_atoms_and_find_mcs(
                    molecule, template, mol_match1, tpl_match1,
                    atom_compare=rdFMCS.AtomCompare.CompareIsotopes,
                    bond_compare=rdFMCS.BondCompare.CompareAny,
                    ring_matches_ring_only=False,
                    complete_rings_only=False
                )
                    
                if mol_match2 and tpl_match2:
                    alignments.append((mol_match2, tpl_match2))
    
    return alignments


def apply_coordinate_alignment(molecule: Chem.Mol, template: Chem.Mol, 
                             alignments: List[Tuple[List[int], List[int]]]) -> None:
    """Apply coordinate alignment based on MCS matches."""
    mol_conformer = molecule.GetConformer()
    template_coords = template.GetConformer().GetPositions()
    
    for mol_match, tpl_match in alignments:
        for mol_idx, tpl_idx in zip(mol_match, tpl_match):
            mol_conformer.SetAtomPosition(mol_idx, template_coords[tpl_idx])

def optimize_molecule_geometry(molecule: Chem.Mol, fixed_atoms: List[int]) -> None:
    """Optimize molecule geometry with constrained and unconstrained steps."""
    best_energy = float('inf')
    best_conf = None
    original_conf = Chem.Conformer(molecule.GetConformer())

    for i in range(10):
        try:
            # Make a copy of the molecule to optimize independently
            mol_copy = Chem.Mol(molecule)

            # Generate a NEW random conformer and realign the fixed atoms to their original positions; This was done in order to avoid pretzel-like cubanes
            AllChem.EmbedMolecule(mol_copy)
            conf = mol_copy.GetConformer()
            for idx in fixed_atoms:
                original_pos = original_conf.GetAtomPosition(idx)
                conf.SetAtomPosition(idx, original_pos)

            # Forcefield, fixing atoms and EM
            ff_copy = AllChem.MMFFGetMoleculeForceField(mol_copy, AllChem.MMFFGetMoleculeProperties(mol_copy))
            for idx in fixed_atoms:
                ff_copy.AddFixedPoint(idx)
            ff_copy.Minimize(forceTol=1e-6, energyTol=1e-8)
        
            # Get energy of this minimization
            energy = ff_copy.CalcEnergy()
            logger.info(f"Run {i+1}: Energy = {energy}")
            # Keep track of the best one
            if energy < best_energy:
                best_energy = energy
                best_conf = mol_copy
                
        except Exception:
            #logger.warning(f"Optimization step {i+1} failed: {e}") # Happens if it is already well aligned
            continue

    # Use the best conformer
    if best_conf is not None:
        conf = best_conf.GetConformer()
        molecule.GetConformer().SetPositions(conf.GetPositions())
        logger.info(f"Best energy from runs: {best_energy}")
    else:
        # All optimizations failed, handle appropriately
        pass
        
    _progressive_constraint_optimization(molecule, fixed_atoms)

def _progressive_constraint_optimization(
    molecule: Chem.Mol,
    constrained_atom_indices: list[int]
) -> None:
    """Gradually release constraints during optimization."""
    
    # Start with stiff constraints, gradually reduce
    force_constants = [1000, 500, 100, 10, 5]
    
    for k in force_constants:
        ff = AllChem.MMFFGetMoleculeForceField(
            molecule, 
            AllChem.MMFFGetMoleculeProperties(molecule)
        )
        
        for atom_idx in constrained_atom_indices:
            ff.MMFFAddPositionConstraint(atom_idx, 0.1, k)
        
        ff.Minimize()

def align_and_optimize(molecule: Chem.Mol, template: Chem.Mol) -> Chem.Mol:
    """Main function to align and optimize molecule to template."""
    # Ensure conformers exist
    if not molecule.GetNumConformers():
        AllChem.EmbedMolecule(molecule)
    if not template.GetNumConformers():
        AllChem.EmbedMolecule(template)
        
    # Check for identical molecules
    mol_smiles = Chem.MolToSmiles(molecule, canonical=True)
    template_smiles = Chem.MolToSmiles(template, canonical=True)
    if mol_smiles == template_smiles:
        logger.info("Molecules are identical. Returning template.")
        return template
    
    # Find multiple MCS alignments
    alignments = find_multiple_mcs_alignments(molecule, template)
    
    if not alignments:
        raise ValueError("No Maximum Common Substructure (MCS) found.")
    
    #logger.info(f"Found {len(alignments)} MCS alignments")
    
    # Apply coordinate alignment
    apply_coordinate_alignment(molecule, template, alignments)
    
    # Collect all fixed atoms for optimization
    all_fixed_atoms = []
    for mol_match, _ in alignments:
        all_fixed_atoms.extend(mol_match)
    
    # Optimize geometry
    optimize_molecule_geometry(molecule, all_fixed_atoms)
        
    return molecule

# =============================================================================
# SHEPHERD SCORING
# =============================================================================

def shepherd_scoring(aligned_molecule: Chem.Mol, template: Chem.Mol) -> tuple[Chem.Mol, float]:
    """Perform shepherd scoring for molecular alignment."""
    try:
        # Calculate formal charges
        formal_charge_template = Chem.GetFormalCharge(template)
        formal_charge_aligned = Chem.GetFormalCharge(aligned_molecule)
        
        #logger.info(f"Template formal charge: {formal_charge_template}")
        #logger.info(f"Aligned molecule formal charge: {formal_charge_aligned}")
        
        # Calculate xTB charges
        ref_charges = charges_from_single_point_conformer_with_xtb(template, charge=formal_charge_template)
        fit_charges = charges_from_single_point_conformer_with_xtb(aligned_molecule, charge=formal_charge_aligned)
        
        # Create Molecule objects
        ref_molec = Molecule(template,
                           num_surf_points=Config.SHEPHERD_SURFACE_POINTS,
                           partial_charges=ref_charges,
                           pharm_multi_vector=False)
        
        fit_molec = Molecule(aligned_molecule,
                           num_surf_points=Config.SHEPHERD_SURFACE_POINTS,
                           partial_charges=fit_charges,
                           pharm_multi_vector=False)
        
        # Create molecular pair
        mp = MoleculePair(ref_molec, fit_molec, 
                         num_surf_points=Config.SHEPHERD_SURFACE_POINTS, 
                         do_center=False)
        
        # Perform alignment
        mp.align_with_vol_esp(ALPHA(mp.num_surf_points),
                             num_repeats=Config.SHEPHERD_REPEATS,
                             lr=Config.SHEPHERD_LEARNING_RATE,
                             max_num_steps=Config.SHEPHERD_MAX_STEPS,
                             no_H=True,
                             verbose=False)
        
        # Get transformed molecule and score
        transformed_fit_molec = mp.get_transformed_molecule(se3_transform=mp.transform_vol_esp_noH)
        vol_esp_score = mp.sim_aligned_vol_esp_noH
        
        return transformed_fit_molec.mol, vol_esp_score
        
    except Exception as e:
        logger.error(f"Shepherd scoring failed: {e}")
        raise

# =============================================================================
# PARALLELIZATION FUNCTIONS
# =============================================================================

def serialize_molecule(mol: Chem.Mol) -> str:
    """Serialize molecule to string for multiprocessing."""
    return Chem.MolToMolBlock(mol)

def deserialize_molecule(mol_block: str) -> Chem.Mol:
    """Deserialize molecule from string."""
    return Chem.MolFromMolBlock(mol_block, removeHs=False)

def process_single_alignment(args):
    """
    Process alignment for a single molecule - designed for multiprocessing.
    
    Args:
        args: Tuple containing (index, smiles, template_mol_block)
    
    Returns:
        Tuple: (success: bool, index: int, aligned_mol_block: str or None, error_msg: str)
    """
    index, smiles, template_mol_block = args
    
    try:
        # Deserialize template
        template = deserialize_molecule(template_mol_block)
        
        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (False, index, None, f"Invalid SMILES: {smiles}")
       
        # Generate 3D conformer
        Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
        status = AllChem.EmbedMolecule(mol)
        if status == -1:
            return (False, index, None, f"Failed to generate 3D conformer for: {smiles}")
            
        # Add hydrogens
        mol = Chem.AddHs(mol, addCoords=True, explicitOnly=True)
        
        # Align molecule
        aligned_mol = align_and_optimize(mol, template)
        
        # Add ALL hydrogens COMMENT
        aligned_mol = Chem.AddHs(aligned_mol, addCoords=True, explicitOnly=False)
        
        # Serialize result
        aligned_mol_block = serialize_molecule(aligned_mol)
        
        return (True, index, aligned_mol_block, f"Successfully aligned molecule {index}")
        
    except Exception as e:
        error_msg = f"Failed to align molecule {index} ({smiles}): {e}"
        return (False, index, None, error_msg)

def process_single_shepherd_scoring(args):
    """
    Process shepherd scoring for a single molecule - designed for multiprocessing.
    
    Args:
        args: Tuple containing (index, aligned_mol_block, reference_mol_block, smiles)
    
    Returns:
        Tuple: (success: bool, index: int, scored_mol_block: str or None, score: float or None, error_msg: str)
    """
    index, aligned_mol_block, reference_mol_block, smiles = args
    
    try:
        # Deserialize molecules
        aligned_mol = deserialize_molecule(aligned_mol_block)
        reference_mol = deserialize_molecule(reference_mol_block)
            
        # Perform shepherd scoring with timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(shepherd_scoring, aligned_mol, reference_mol)
            shepherd_aligned, vol_esp_score = future.result(timeout=Config.TIMEOUT)
        
        # Serialize result
        scored_mol_block = serialize_molecule(shepherd_aligned)
        
        return (True, index, scored_mol_block, vol_esp_score, f"Successfully scored molecule {index}")
        
    except concurrent.futures.TimeoutError:
        error_msg = f"Timeout: Failed to score molecule {index} ({smiles}) within {Config.TIMEOUT} seconds"
        return (False, index, aligned_mol_block, 0.0, error_msg)  # Return original alignment
    except Exception as e:
        error_msg = f"Failed to score molecule {index} ({smiles}): {e}"
        return (False, index, aligned_mol_block, 0.0, error_msg)  # Return original alignment

def align_molecules_parallel(smiles_list: List[str], template: Chem.Mol, n_processes: int = None) -> Tuple[List[Chem.Mol], List[str]]:
    """
    Align all molecules in parallel.
    
    Args:
        smiles_list: List of SMILES strings
        template: Template molecule
        n_processes: Number of processes to use
    
    Returns:
        Tuple of (aligned_molecules, error_messages)
    """
    logger.info(f"Starting parallel alignment of {len(smiles_list)} molecules")
    
    if n_processes is None:
        n_processes = max(1, cpu_count() - 4)
    
    logger.info(f"Using {n_processes} processes for alignment")
    
    # Serialize template for multiprocessing
    template_mol_block = serialize_molecule(template)
    
    # Prepare arguments
    args_list = [(i, smiles, template_mol_block) for i, smiles in enumerate(smiles_list)]
    
    # Process in parallel
    start_time = time.time()
    aligned_molecules = [None] * len(smiles_list)
    error_messages = []
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_alignment, args_list)
    
    # Process results
    successful_alignments = 0
    for success, index, aligned_mol_block, msg in results:
        if success and aligned_mol_block is not None:
            aligned_molecules[index] = deserialize_molecule(aligned_mol_block)
            successful_alignments += 1
            #logger.info(f"{msg}")
        else:
            logger.error(f"{msg}")
            error_messages.append(msg)
    
    end_time = time.time()
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"Parallel alignment completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Successfully aligned {successful_alignments}/{len(smiles_list)} molecules")
    logger.info("=" * 60)
        
    return aligned_molecules, error_messages

def score_molecules_parallel(aligned_molecules: List[Chem.Mol], smiles_list: List[str], 
                           reference_mol: Chem.Mol, n_processes: int = None) -> Tuple[List[Chem.Mol], List[float]]:
    """
    Score aligned molecules in parallel using shepherd scoring.
    
    Args:
        aligned_molecules: List of aligned molecules
        smiles_list: List of SMILES strings (for error reporting)
        reference_mol: Reference molecule for scoring
        n_processes: Number of processes to use
    
    Returns:
        Tuple of (scored_molecules, scores)
    """
    logger.info(f"Starting parallel shepherd scoring of {len(aligned_molecules)} molecules")
    
    if n_processes is None:
        n_processes = min(cpu_count(), len(aligned_molecules))
    
    logger.info(f"Using {n_processes} processes for shepherd scoring")
    
    # Serialize reference molecule
    reference_mol_block = serialize_molecule(reference_mol)
    
    # Prepare arguments (skip None molecules and first molecule which is the reference)
    args_list = []
    for i, (aligned_mol, smiles) in enumerate(zip(aligned_molecules, smiles_list)):
        if aligned_mol is not None and i > 0:  # Skip first molecule (reference)
            aligned_mol_block = serialize_molecule(aligned_mol)
            args_list.append((i, aligned_mol_block, reference_mol_block, smiles))
    
    # Process in parallel
    start_time = time.time()
    scored_molecules = aligned_molecules.copy()  # Start with aligned molecules
    scores = [1.0 if i == 0 else 0.0 for i in range(len(aligned_molecules))]  # Reference gets score 1.0
    
    if args_list:  # Only process if there are molecules to score
        with Pool(processes=n_processes) as pool:
            results = pool.map(process_single_shepherd_scoring, args_list)
        
        # Process results
        successful_scores = 0
        for success, index, scored_mol_block, score, msg in results:
            if scored_mol_block is not None:
                scored_molecules[index] = deserialize_molecule(scored_mol_block)
                scores[index] = score if score is not None else 0.0
                successful_scores += 1
                
                if success:
                    #logger.info(f"{msg} (score: {score:.4f})")
                    continue
                else:
                    logger.warning(f"{msg} (using alignment, score: {score:.4f})")
            else:
                logger.error(f"{msg}")
        
        end_time = time.time()
        logger.info(f"Parallel scoring completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Successfully scored {successful_scores}/{len(args_list)} molecules")
    
    return scored_molecules, scores

def align_molecules_serial(smiles_list: List[str], template: Chem.Mol) -> Tuple[List[Chem.Mol], List[float]]:
    """Serial processing for comparison/debugging."""
    logger.info(f"Starting serial processing of {len(smiles_list)} molecules")
    
    aligned_molecules = []
    scores = []
    start_time = time.time()
    
    for i, smiles in enumerate(smiles_list):
        try:
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES: {smiles}")
                aligned_molecules.append(None)
                scores.append(0.0)
                continue
            
            # Generate 3D conformer
            Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
            status = AllChem.EmbedMolecule(mol)
            if status == -1:
                logger.error(f"Failed to generate 3D conformer for: {smiles}")
                aligned_molecules.append(None)
                scores.append(0.0)
                continue
            
            # Align molecule
            aligned_mol = align_and_optimize(mol, template)
            aligned_molecules.append(aligned_mol)
            
            template_for_scoring = template
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(shepherd_scoring, aligned_mol, template_for_scoring)
                try:
                    shepherd_aligned, vol_esp_score = future.result(timeout=Config.TIMEOUT)
                    aligned_molecules[i] = shepherd_aligned
                    scores.append(vol_esp_score)
                    logger.info(f"Successfully processed molecule {i+1}/{len(smiles_list)}")
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout: Failed to score molecule {i} ({smiles}) within {Config.TIMEOUT} seconds")
                    scores.append(0.0)
                
        except Exception as e:
            logger.error(f"Failed to process molecule {i} ({smiles}): {e}")
            aligned_molecules.append(None)
            scores.append(0.0)
    
    end_time = time.time()
    logger.info(f"Serial processing completed in {end_time - start_time:.2f} seconds")
    
    return aligned_molecules, scores

# =============================================================================
# MAIN PROCESSING WITH PARALLELIZATION OPTIONS
# =============================================================================

def align_molecules_main(
    template_molecule: str,
    derivatives_to_align: str,
    aligned_molecules: str,
    SMILES_column:  str,
    method: str = 'parallel',
    processes: Optional[int] = None
) -> None:
    """
    Main function for aligning molecules - can be imported or run from CLI.
    
    Args:
        template_molecule: Path to the template SDF file
        derivatives_to_align: Path to CSV file containing derivatives to align
        aligned_molecules: Path to save output directory for aligned molecules
        method: Processing method ('serial' or 'parallel')
        processes: Number of processes to use (None = auto-detect)
    """
    try:
        # Load data
        derivatives_df = pd.read_csv(derivatives_to_align)
        smiles_list = derivatives_df[SMILES_column].dropna().tolist()
        logger.info(f"Loaded {len(smiles_list)} SMILES for processing")
        
        # Load template
        template = load_molecule_from_sdf(Path(template_molecule))
        template = Chem.AddHs(template, addCoords=True, explicitOnly=True)
        logger.info(f"Loaded template molecule with {template.GetNumAtoms()} atoms")
        
        # Process molecules based on method choice
        if method == 'serial':
            aligned_molecules_list, scores = align_molecules_serial(smiles_list, template)
        elif method == 'parallel':
            # Two-phase parallel processing
            aligned_molecules_list, error_messages = align_molecules_parallel(smiles_list, template, processes)
            
            # Parallel scoring
            scored_molecules, scores = score_molecules_parallel(aligned_molecules_list, smiles_list, template, processes)
            aligned_molecules_list = scored_molecules
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Save aligned molecules (skip first one and None values)
        output_dir = Path(aligned_molecules)
        saved_count = 0
        for i, aligned_mol in enumerate(aligned_molecules_list[1:]):  # Skip template
            if aligned_mol is not None:
                output_file = output_dir / f"aligned_derivative_{i:03d}.sdf"
                #logger.info(f"Saving derivative {i} to {output_file}")
                save_molecule_to_sdf(aligned_mol, output_file)
                saved_count += 1
        
        logger.info(f"Saved {saved_count} aligned molecules")
        
        # Update CSV with scores
        if len(scores) == len(derivatives_df):
            derivatives_df['Volumetric shape and ESP similarity score'] = scores
            derivatives_df.to_csv(derivatives_to_align, index=False)
            logger.info("Updated CSV with similarity scores")
        else:
            logger.warning(f"Score count mismatch: {len(scores)} scores vs {len(derivatives_df)} molecules")
        
        logger.info("Alignment process completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        raise

def main(args) -> int:
    """CLI wrapper for align_molecules_main."""
    try:
        align_molecules_main(
            template_molecule=args.template_molecule,
            derivatives_to_align=args.derivatives_to_align,
            aligned_molecules=args.aligned_molecules,
            SMILES_column=args.SMILES_column,
            method=args.method,
            processes=args.processes
        )
        return 0
    except Exception:
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align molecules to a template and score volumetric and electrostatic complementarity using ShEPhERD with parallelization options.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--template_molecule", "-t", type=str, required=True, 
                       help="Path to the template SDF file")
    parser.add_argument("--derivatives_to_align", "-d", type=str, required=True, 
                       help="Path to CSV file containing derivatives to align")
    parser.add_argument("--aligned_molecules", "-a", type=str, default="aligned_molecules",
                       help="Path to save output directory for aligned molecules (default: aligned_molecules)")
    parser.add_argument("--SMILES_column", "-s", type=str, default="product_SMILES",
                       help="Name of SMILES column (default: product_SMILES)")                           
    parser.add_argument("--method", "-m", type=str, choices=['serial', 'parallel'], 
                       default='parallel',
                       help="Processing method: serial (original) or parallel (two-phase) (default: parallel).")
    parser.add_argument("--processes", "-p", type=int, default=None,
                       help="Number of processes to use for parallel processing (default: cpu_count - 4).")                   
    
    args = parser.parse_args()
    exit(main(args))
