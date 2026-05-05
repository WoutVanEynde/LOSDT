import os
import sys
import warnings

# Only use CPU; otherwise GPU overhead
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Suppress RDKit warnings about boost converters
warnings.filterwarnings("ignore", message=".*to-Python converter.*already registered.*")

# PyInstaller imports
if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    os.environ['PATH'] = bundle_dir + os.pathsep + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = bundle_dir + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

import pandas as pd
import json
import argparse
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from admet_ai import ADMETModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static', 'third_party_software', 'pKaLearn', 'GNN')))
from predict import predict

# Configure logging for multiprocessing
def setup_logging():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

logger = setup_logging()

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration constants for ADMET prediction."""
    MAX_MOLECULAR_WEIGHT = 5000
    MIN_TPSA = 0
    MAX_TPSA = 2000
    FINGERPRINT_RADIUS = 2
    FINGERPRINT_BITS = 1024
    PERFECT_SIMILARITY = 1.0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_reactants(reactants_file_path: Path) -> List[str]:
    """Load and clean reactants from file."""
    try:
        with open(reactants_file_path, "r") as file:
            reactants = [line.strip() for line in file if line.strip()]
        logger.info(f"Loaded {len(reactants)} reactants from {reactants_file_path}")
        return reactants
    except Exception as e:
        logger.error(f"Error loading reactants: {e}")
        raise

def validate_input_molecule(smiles: str) -> Chem.Mol:
    """Validate input SMILES and return RDKit molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return mol

def calculate_molecular_properties(mol: Chem.Mol) -> Dict[str, float]:
    """Calculate basic molecular properties."""
    return {
        'molecular_weight': Descriptors.MolWt(mol),
        'tpsa': Descriptors.TPSA(mol),
        'fingerprint': AllChem.GetMorganFingerprintAsBitVect(mol, Config.FINGERPRINT_RADIUS, nBits=Config.FINGERPRINT_BITS)
    }

# =============================================================================
# REACTION PROCESSING
# =============================================================================

def apply_reaction_to_molecule(reaction: AllChem.ChemicalReaction, 
                             input_molecule: Chem.Mol, 
                             educt2: Chem.Mol = None) -> List[Chem.Mol]:
    """Apply SMIRKS reaction to molecule(s) and return products."""
    try:
        if reaction.GetNumReactantTemplates() == 1:
            reaction_results = reaction.RunReactants((input_molecule,))
        elif reaction.GetNumReactantTemplates() == 2:
            if educt2 is None:
                return []
            reaction_results = reaction.RunReactants((input_molecule, educt2))
        else:
            logger.warning(f"Unexpected number of reactants: {reaction.GetNumReactantTemplates()}")
            return []
        
        products = []
        for product_set in reaction_results:
            for product in product_set:
                products.append(product)
        
        return products
    except Exception as e:
        logger.error(f"Error applying reaction: {e}")
        return []

def process_product(product: Chem.Mol, input_fingerprint: DataStructs.ExplicitBitVect) -> Dict[str, Any]:
    """Process a single product molecule and return its properties."""
    try:
        # Ensure proper structure
        Chem.rdDepictor.Compute2DCoords(product)
        Chem.SanitizeMol(product)
        
        # Calculate properties
        mol_weight = Descriptors.MolWt(product)
        tpsa = Descriptors.TPSA(product)
        
        # Apply filters
        if not (mol_weight < Config.MAX_MOLECULAR_WEIGHT and Config.MIN_TPSA <= tpsa <= Config.MAX_TPSA):
            return None
        
        # Calculate similarity
        product_fp = AllChem.GetMorganFingerprintAsBitVect(product, Config.FINGERPRINT_RADIUS, nBits=Config.FINGERPRINT_BITS)
        tanimoto_similarity = DataStructs.TanimotoSimilarity(input_fingerprint, product_fp)
        
        return {
            "Smiles": Chem.MolToSmiles(product),
            "molecular_weight_RDkit": mol_weight,
            "TPSA_RDKit": tpsa,
            "tanimoto_similarity_input_SMILES": tanimoto_similarity
        }
        
    except Exception as e:
        logger.error(f"Error processing product: {e}")
        return None

def process_single_reaction(reaction_name: str, 
                          smirks: str, 
                          input_molecule: Chem.Mol,
                          input_fingerprint: DataStructs.ExplicitBitVect,
                          educt2_smiles_list: List[str]) -> List[Dict[str, Any]]:
    """Process a single reaction with all reactants."""
    logger.info(f"Processing reaction: {reaction_name}")
    
    try:
        reaction = AllChem.ReactionFromSmarts(smirks)
        results = []
        
        # Process reaction with each educt2
        for educt2_smiles in educt2_smiles_list:
            educt2 = Chem.MolFromSmiles(educt2_smiles) if educt2_smiles else None
            
            if educt2 is None and reaction.GetNumReactantTemplates() == 2:
                continue
            
            products = apply_reaction_to_molecule(reaction, input_molecule, educt2)
            
            for product in products:
                product_data = process_product(product, input_fingerprint)
                if product_data:
                    product_data["reaction_name"] = reaction_name
                    results.append(product_data)
        
        logger.info(f"Generated {len(results)} valid products for reaction {reaction_name}")
        return results
        
    except Exception as e:
        logger.error(f"Error processing reaction {reaction_name}: {e}")
        return []

def process_all_reactions(reactions_df: pd.DataFrame, 
                        input_molecule: Chem.Mol,
                        input_fingerprint: DataStructs.ExplicitBitVect,
                        educt2_smiles_list: List[str]) -> List[Dict[str, Any]]:
    """Process all reactions and return combined results."""
    all_results = []
    
    for _, row in reactions_df.iterrows():
        reaction_results = process_single_reaction(
            row['reaction_name'], 
            row['smirks'], 
            input_molecule, 
            input_fingerprint, 
            educt2_smiles_list
        )
        all_results.extend(reaction_results)
    
    return all_results

# =============================================================================
# ADMET PREDICTION
# =============================================================================

def predict_admet_for_molecules(molecules_df: pd.DataFrame, model: ADMETModel, protonation: bool) -> pd.DataFrame:
    """Predict ADMET properties for all molecules in the dataframe."""
    logger.info("Starting ADMET predictions...")
    
    # Predict ADMET properties
    if protonation is True:
        molecules_df['ADMET_properties'] = molecules_df['Protonated SMILES'].apply(
            lambda smiles: model.predict(smiles=smiles)
        )
    else: 
        molecules_df['ADMET_properties'] = molecules_df['Smiles'].apply(
            lambda smiles: model.predict(smiles=smiles)
        )    
    # Handle ADMET properties based on their type
    def process_admet_properties(x):
        if isinstance(x, dict):
            # Already a dictionary, return as is
            return x
        elif isinstance(x, str):
            # String representation, need to parse
            try:
                # Replace single quotes with double quotes and parse
                return json.loads(x.replace("'", '"'))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse ADMET properties: {x}")
                return {}
        else:
            logger.warning(f"Unexpected ADMET properties type: {type(x)}")
            return {}
    
    molecules_df['ADMET_properties'] = molecules_df['ADMET_properties'].apply(process_admet_properties)
    
    # Expand ADMET properties
    admet_expanded = pd.json_normalize(molecules_df['ADMET_properties'])
    final_df = pd.concat([molecules_df.drop(columns=['ADMET_properties']), admet_expanded], axis=1)
    
    logger.info("ADMET predictions completed")
    return final_df

def create_input_molecule_entry(input_smiles: str, 
                              input_mol_properties: Dict[str, float],
                              model: ADMETModel,
                              protonation: bool) -> Dict[str, Any]:
    """Create entry for input molecule with ADMET properties."""
    if protonation is True:
        SMILES_df = pd.DataFrame([input_smiles], columns=['Smiles'])
        predicted_pkas, protonated_SMILES = predict(SMILES_df, pH=7, device='cpu')
        input_smiles_protonated = protonated_SMILES[0]
        try:
            input_admet = model.predict(smiles=input_smiles_protonated)
            if input_admet is None:
                raise ValueError("model.predict returned None for protonated SMILES")
        except Exception as e:
            logger.warning(f"ADMET prediction failed for protonated SMILES '{input_smiles_protonated}', falling back to original: {e}")
            input_smiles_protonated = input_smiles
            input_admet = model.predict(smiles=input_smiles)

    else:
        input_admet = model.predict(smiles=input_smiles)
    
    # Handle ADMET properties based on their type
    if isinstance(input_admet, dict):
        # Already a dictionary, use as is
        admet_dict = input_admet
    elif isinstance(input_admet, str):
        # String representation, need to parse
        try:
            admet_dict = json.loads(input_admet.replace("'", '"'))
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse input ADMET properties: {input_admet}")
            admet_dict = {}
    else:
        logger.warning(f"Unexpected input ADMET properties type: {type(input_admet)}")
        admet_dict = {}
        
    if protonation is True:
        return {
            "reaction_name": "Input Molecule",
            "Smiles": input_smiles,
            'Protonated SMILES': input_smiles_protonated,
            "molecular_weight_RDkit": input_mol_properties['molecular_weight'],
            "TPSA_RDKit": input_mol_properties['tpsa'],
            "tanimoto_similarity_input_SMILES": Config.PERFECT_SIMILARITY,
            **admet_dict
        }
        
    else:
        return {
            "reaction_name": "Input Molecule",
            "Smiles": input_smiles,
            "molecular_weight_RDkit": input_mol_properties['molecular_weight'],
            "TPSA_RDKit": input_mol_properties['tpsa'],
            "tanimoto_similarity_input_SMILES": Config.PERFECT_SIMILARITY,
            **admet_dict
        }        

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def predict_admet_for_reactions(input_smiles: str, 
                              reactions_csv_path: Path, 
                              reactants_txt_path: Path, 
                              output_csv_path: Path,
                              protonation: bool) -> pd.DataFrame:
    """Main function to predict ADMET properties for reactions."""
    logger.info("Starting ADMET prediction workflow")
    
    # Load data
    reactions_df = pd.read_csv(reactions_csv_path)
    educt2_smiles_list = load_reactants(reactants_txt_path)
    
    # Validate input molecule
    input_molecule = validate_input_molecule(input_smiles)
    input_mol_properties = calculate_molecular_properties(input_molecule)
    
    # Initialize ADMET model
    model = ADMETModel()
    
    # Process all reactions
    all_results = process_all_reactions(
        reactions_df, 
        input_molecule, 
        input_mol_properties['fingerprint'],
        educt2_smiles_list
    )
    
    if not all_results:
        logger.warning("No valid reaction products generated")
        return pd.DataFrame()
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(all_results)
    
    # Log initial results
    logger.info(f"Generated {len(results_df)} total products before deduplication")
    
    # Sort by similarity and remove duplicates
    results_df = results_df.sort_values(by='tanimoto_similarity_input_SMILES', ascending=False)
    results_df = results_df.drop_duplicates(subset='Smiles', keep='first')
    
    # Log after deduplication
    logger.info(f"After deduplication: {len(results_df)} unique products")
    
    # Reset index after deduplication to ensure clean indexing
    results_df = results_df.reset_index(drop=True)
    
    # PROTONATE
    if protonation is True:
        predicted_pkas, protonated_SMILES = predict(results_df, pH=7, device='cpu')
        protonated_SMILES_df = pd.DataFrame(protonated_SMILES, columns=['Protonated SMILES'])
        results_df_protonated = protonated_SMILES_df.join(results_df)
        results_df = results_df_protonated
    
    # Predict ADMET properties
    results_with_admet = predict_admet_for_molecules(results_df, model, protonation)
    
    # Create input molecule entry
    input_data = create_input_molecule_entry(input_smiles, input_mol_properties, model, protonation)
    input_df = pd.DataFrame([input_data])
    
    # Combine results - input molecule first, then derivatives
    final_result = pd.concat([input_df, results_with_admet], ignore_index=True)
    
    # Get references in there
    final_result = final_result.merge(reactions_df, on="reaction_name")
    final_result = final_result.drop(columns=['smirks'])
    final_result = pd.concat([input_df, final_result], ignore_index=True)
    final_result = final_result.rename(columns={'Smiles': 'product_SMILES'})

    # Log final result
    logger.info(f"Final result has {len(final_result)} rows (1 input + {len(results_with_admet)} derivatives)")
    
    # Save results
    final_result.to_csv(output_csv_path, index=False)
    logger.info(f"Results saved to {output_csv_path}")
    
    return final_result

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(args) -> int:
    """Main function for ADMET prediction."""
    try:
        predict_admet_for_reactions(
            args.smiles,
            Path(args.reactions),
            Path(args.reactants),
            Path(args.output),
            protonation=args.protonation,
        )
        logger.info("ADMET prediction completed successfully")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predicts ADMET properties for the bioisosters of input SMILES using ADMET-AI from Swanson et al; https://doi.org/10.1093/bioinformatics/btae416",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--smiles", "-s", type=str, required=True, 
                       help="SMILES string for the input molecule")
    parser.add_argument("--reactions", "-r", type=str, required=True, 
                       help="Path to the reactions CSV file")
    parser.add_argument("--reactants", "-r2",  type=str, required=True, 
                       help="Path to the reactants text file")
    parser.add_argument("--output", "-o", type=str, default="results_with_admet.csv", 
                       help="Path to save the output CSV file (default: results_with_admet.csv)")
    parser.add_argument('--protonation', '-prot', action='store_true',
                       help='Protonation of ligands (default=True).')                       
    
    args = parser.parse_args()
    exit(main(args))
