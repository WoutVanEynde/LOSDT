import pandas as pd
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import concurrent.futures
import logging
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import multiprocessing as mp

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles, rdMolAlign, rdForceFieldHelpers, rdFMCS

# Add shepherd score to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../LOSDT/', 'static', 'third_party_software', 'shepherd-score')))

from shepherd_score.score.constants import ALPHA, LAM_SCALING
from shepherd_score.conformer_generation import embed_conformer_from_smiles, charges_from_single_point_conformer_with_xtb
from shepherd_score.extract_profiles import get_atomic_vdw_radii, get_molecular_surface, get_pharmacophores, get_electrostatic_potential, get_pharmacophores_dict
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
    TIMEOUT = 60  # seconds
    MCS_TIMEOUT = 30  # seconds
    RMS_THRESHOLD = 1.0
    MAX_OPTIMIZATION_ITERATIONS = 1000
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
    """Save molecule to SDF file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with rdmolfiles.SDWriter(str(output_file)) as writer:
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
        
        return mcs, list(mol1_match), list(mol2_match)
    
    except Exception as e:
        logger.error(f"Error finding MCS: {e}")
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

# =============================================================================
# MOLECULAR ALIGNMENT
# =============================================================================

def find_multiple_mcs_alignments(molecule: Chem.Mol, template: Chem.Mol) -> List[Tuple[List[int], List[int]]]:
    """Find multiple MCS alignments between molecule and template."""
    alignments = []
    
    # First MCS - strict parameters
    _, mol_match1, tpl_match1 = find_mcs_with_params(
        molecule, template,
        atom_compare=rdFMCS.AtomCompare.CompareElements,
        bond_compare=rdFMCS.BondCompare.CompareOrder,
        ring_matches_ring_only=True,
        complete_rings_only=True
    )
    
    if mol_match1 and tpl_match1:
        alignments.append((mol_match1, tpl_match1))
        
        # Second MCS - remove first MCS atoms
        mol_match2, tpl_match2 = remove_atoms_and_find_mcs(
            molecule, template, mol_match1, tpl_match1,
            atom_compare=rdFMCS.AtomCompare.CompareAny, #WAS atom_compare=rdFMCS.AtomCompare.CompareElements
            bond_compare=rdFMCS.BondCompare.CompareOrder,
            ring_matches_ring_only=True,
            complete_rings_only=True
        )
        
        if mol_match2 and tpl_match2:
            alignments.append((mol_match2, tpl_match2))
            
            # Third MCS - remove first and second MCS atoms
            all_mol_matches = mol_match1 + mol_match2
            all_tpl_matches = tpl_match1 + tpl_match2
            
            mol_match3, tpl_match3 = remove_atoms_and_find_mcs(
                molecule, template, all_mol_matches, all_tpl_matches,
                atom_compare=rdFMCS.AtomCompare.CompareAny,
                bond_compare=rdFMCS.BondCompare.CompareAny,
                ring_matches_ring_only=False,
                complete_rings_only=False
            )
            
            if mol_match3 and tpl_match3:
                alignments.append((mol_match3, tpl_match3))
    
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
    # Constrained optimization
    ff = AllChem.MMFFGetMoleculeForceField(molecule, AllChem.MMFFGetMoleculeProperties(molecule))
    
    # Fix matched atoms
    for idx in fixed_atoms:
        ff.AddFixedPoint(idx)
    
    # Multiple constrained optimization steps
    for _ in range(5):
        try:
            ff.Minimize()
        except Exception as e:
            logger.warning(f"Constrained optimization step failed: {e}")
            break
    
    # Unconstrained optimization with RMS gradient check
    _unconstrained_optimization(molecule)

def _unconstrained_optimization(molecule: Chem.Mol) -> None:
    """Perform unconstrained geometry optimization with RMS gradient monitoring."""
    ff = AllChem.MMFFGetMoleculeForceField(molecule, AllChem.MMFFGetMoleculeProperties(molecule))
    
    # Initial gradient
    initial_gradients = ff.CalcGrad()
    initial_rms = _calculate_rms_gradient(initial_gradients)
    logger.info(f"Initial RMS gradient: {initial_rms}")
    
    # Optimize with convergence check
    for iteration in range(Config.MAX_OPTIMIZATION_ITERATIONS):
        ff.Minimize(maxIts=1)
        current_gradients = ff.CalcGrad()
        current_rms = _calculate_rms_gradient(current_gradients)
        
        if iteration % 100 == 0:
            logger.info(f"Iteration {iteration}, RMS gradient: {current_rms}")
        
        if current_rms < Config.RMS_THRESHOLD:
            logger.info(f"Converged at iteration {iteration}! Final RMS gradient: {current_rms}")
            break
    else:
        logger.warning(f"Optimization did not converge. Final RMS gradient: {current_rms}")

def _calculate_rms_gradient(gradients) -> float:
    """Calculate RMS from gradient components."""
    grad_list = list(gradients)
    return (sum(x*x for x in grad_list) / len(grad_list))**0.5

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
    
    logger.info(f"Found {len(alignments)} MCS alignments")
    
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

def shepherd_scoring(aligned_molecule: Chem.Mol, template: Chem.Mol) -> Tuple[Chem.Mol, float]:
    """Perform shepherd scoring for molecular alignment."""
    logger.info("Starting Shepherd alignment using ESP and volumetric shape")
    
    try:
        # Calculate formal charges
        formal_charge_template = Chem.GetFormalCharge(template)
        formal_charge_aligned = Chem.GetFormalCharge(aligned_molecule)
        
        logger.info(f"Template formal charge: {formal_charge_template}")
        logger.info(f"Aligned molecule formal charge: {formal_charge_aligned}")
        
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
        logger.info("Performing volumetric shape and ESP alignment")
        mp.align_with_vol_esp(ALPHA(mp.num_surf_points),
                             num_repeats=Config.SHEPHERD_REPEATS,
                             lr=Config.SHEPHERD_LEARNING_RATE,
                             max_num_steps=Config.SHEPHERD_MAX_STEPS,
                             no_H=True,
                             verbose=True)
        
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
    return Chem.MolFromMolBlock(mol_block)

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
        
        # Align molecule
        aligned_mol = align_and_optimize(mol, template)
        
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
        n_processes = min(cpu_count(), len(smiles_list))
    
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
            logger.info(f"✓ {msg}")
        else:
            logger.error(f"✗ {msg}")
            error_messages.append(msg)
    
    end_time = time.time()
    logger.info(f"Parallel alignment completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Successfully aligned {successful_alignments}/{len(smiles_list)} molecules")
    
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
                    logger.info(f"✓ {msg} (score: {score:.4f})")
                else:
                    logger.warning(f"⚠ {msg} (using alignment, score: {score:.4f})")
            else:
                logger.error(f"✗ {msg}")
        
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
            
            # Shepherd scoring (skip first molecule which is the template)
            if i > 0:
                template_for_scoring = aligned_molecules[0]
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
            else:
                scores.append(1.0)  # Perfect score for template
                
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
    method: str = 'parallel',
    processes: Optional[int] = None,
    SMILES_column:  str = 'product_SMILES'
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
        smiles_list = derivatives_df['product_SMILES'].dropna().tolist()
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
        for i, aligned_mol in enumerate(aligned_molecules_list[1:], start=1):  # Skip first molecule
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
            method=args.method,
            processes=args.processes
        )
        return 0
    except Exception:
        return 1

if __name__ == "__main__":
    # Important for multiprocessing on Windows
    mp.set_start_method('spawn', force=True)
    
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
    parser.add_argument("--method", "-m", type=str, choices=['serial', 'parallel'], 
                       default='parallel',
                       help="Processing method: serial (original) or parallel (two-phase). Default: parallel")
    parser.add_argument("--processes", "-p", type=int, default=None,
                       help="Number of processes to use for parallel processing (default: auto-detect)")
    
    args = parser.parse_args()
    exit(main(args))
