import os
import sys

# PyInstaller imports
if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    os.environ['PATH'] = bundle_dir + os.pathsep + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = bundle_dir + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

import argparse
import subprocess
import logging
import traceback
from rdkit import Chem
from pdbtools import pdb_merge, pdb_tocif

# Configure logging for multiprocessing
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def sdf_to_pdb(sdf_path, pdb_path):
    """Convert SDF to PDB using RDKit."""
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    if mol is None:
        raise ValueError(f"Could not read SDF: {sdf_path}")
    Chem.MolToPDBFile(mol, pdb_path)

def merge_pdbs(protein_pdb, ligand_pdb, output_pdb):
    """Merge protein and ligand PDB files using pdb-tools."""
    # Open both input files and merge them
    with open(protein_pdb, 'r') as f1, open(ligand_pdb, 'r') as f2, open(output_pdb, 'w') as out_f:
        # pdb_merge.run takes multiple file handles
        for line in pdb_merge.run([f1, f2]):
            out_f.write(line)
        
def clean_pdb(input_file, output_file):
    with open(input_file) as f:
        lines = f.readlines()
    # strip all END lines, keep everything else
    stripped = [line for line in lines if not line.startswith(("END", "HEADER"))]
    # add a single END at the end
    stripped.append("END\n")
    with open(output_file, "w") as f:
        f.writelines(stripped)

def renumber_ligand(input_file, output_file):
    """
    Renumber ligand atoms in merged PDB to avoid duplicate atom numbers.
    Finds the ligand section (UNL residues) and renumbers atoms sequentially
    after the last protein atom number.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Find max atom number in protein section and identify ligand atoms
    max_protein_atom = 0
    ligand_atom_lines = []
    ligand_atom_nums = set()
    
    for i, line in enumerate(lines):
        if line.startswith(('ATOM', 'HETATM')) and not line.startswith('END'):
            atom_num = int(line[6:11].strip())
            
            # Check if this is a ligand atom (UNL residue)
            if line.startswith('HETATM'):
                res_name = line[17:20].strip()
                if res_name == 'UNL':
                    ligand_atom_lines.append(i)
                    ligand_atom_nums.add(atom_num)
                else:
                    # Water or other HETATM
                    max_protein_atom = max(max_protein_atom, atom_num)
            else:
                # Regular protein atom
                max_protein_atom = max(max_protein_atom, atom_num)
    
    if not ligand_atom_lines:
        logger.warning("No UNL ligand atoms found, skipping renumbering")
        with open(output_file, 'w') as f:
            f.writelines(lines)
        return
    
    # Create mapping from old to new atom numbers
    # Find the original numbering of ligand atoms
    original_ligand_nums = sorted(ligand_atom_nums)
    
    # New numbering starts after max protein atom
    start_num = max_protein_atom + 1
    atom_mapping = {}
    for idx, old_num in enumerate(original_ligand_nums):
        atom_mapping[old_num] = start_num + idx
    
    logger.info(f"Renumbering ligand: {len(original_ligand_nums)} atoms from {original_ligand_nums[0]}-{original_ligand_nums[-1]} to {start_num}-{start_num + len(original_ligand_nums) - 1}")
    
    # Process lines and renumber
    new_lines = []
    for i, line in enumerate(lines):
        if line.startswith('HETATM') and i in ligand_atom_lines:
            # Renumber ligand HETATM line
            old_num = int(line[6:11].strip())
            new_num = atom_mapping[old_num]
            new_line = line[:6] + f"{new_num:5d}" + line[11:]
            new_lines.append(new_line)
            
        elif line.startswith('CONECT'):
            # Update CONECT records
            parts = line.split()
            if len(parts) > 1:
                old_atom = int(parts[1])
                
                # Check if this CONECT involves ligand atoms
                if old_atom in atom_mapping:
                    # This is a ligand CONECT line, renumber all atoms
                    new_atom = atom_mapping[old_atom]
                    conect_str = f"CONECT{new_atom:5d}"
                    
                    for j in range(2, len(parts)):
                        old_connected = int(parts[j])
                        new_connected = atom_mapping.get(old_connected, old_connected)
                        conect_str += f"{new_connected:5d}"
                    
                    new_lines.append(conect_str + '\n')
                else:
                    # Protein CONECT, keep as is
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            # Keep line as is
            new_lines.append(line)
    
    with open(output_file, 'w') as f:
        f.writelines(new_lines)

def pdb_to_cif(input_pdb, output_cif):
    """Convert pdb to cif using pdb-tools."""
    with open(input_pdb, 'r') as in_f, open(output_cif, 'w') as out_f:
        for line in pdb_tocif.run(in_f):
            out_f.write(line)

def create_complexes_main(protein: str, ligands: str, complexes: str = "complexes") -> None:
    """
    Create protein-ligand complexes - can be imported or run from CLI.
    
    Args:
        protein: Protein PDB file path
        ligands: Directory with SDF ligands
        complexes: Output directory for complexes
    """
    try:
        # Ensure output directory exists
        os.makedirs(complexes, exist_ok=True)
        
        # Process each SDF in the ligand directory
        for sdf_file in os.listdir(ligands):
            if sdf_file.lower().endswith(".sdf"):
                ligand_name = os.path.splitext(sdf_file)[0]
                sdf_path = os.path.join(ligands, sdf_file)
                ligand_pdb_path = os.path.join(complexes, ligand_name + "_ligand.pdb")
                complex_pdb_path = os.path.join(complexes, ligand_name + "_complex_inital.pdb")
                complex_pdb_path_intermediate = os.path.join(complexes, ligand_name + "_complex_intermediate.pdb")
                complex_pdb_path_final = os.path.join(complexes, ligand_name + "_complex_final.pdb")
                complex_cif_path = os.path.join(complexes, ligand_name + "_complex.cif")
                
                logger.info(f"[INFO] Processing ligand: {ligand_name}")
                
                try:
                    # Convert SDF to PDB
                    sdf_to_pdb(sdf_path, ligand_pdb_path)
                    
                    # Merge with protein
                    merge_pdbs(protein, ligand_pdb_path, complex_pdb_path)
                    
                    # Clean pdb
                    clean_pdb(complex_pdb_path, complex_pdb_path_intermediate)
                    
                    # Renumber ligand atoms to avoid duplicates
                    renumber_ligand(complex_pdb_path_intermediate, complex_pdb_path_final)
                    
                    # Output cif
                    pdb_to_cif(complex_pdb_path_final, complex_cif_path)
                    
                    logger.info(f"[SUCCESS] Complex saved: {complex_cif_path}")
                    
                    # Clean files 
                    os.unlink(ligand_pdb_path)
                    os.unlink(complex_pdb_path)
                    os.unlink(complex_pdb_path_intermediate)
                    
                except Exception as e:
                    logger.info(f"[ERROR] Failed for {ligand_name}: {e}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        raise

def main(args) -> int:
    """CLI wrapper for create_complexes_main."""
    try:
        create_complexes_main(
            protein=args.protein,
            ligands=args.ligands,
            complexes=args.complexes
        )
        return 0
    except Exception:
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make protein-ligand complexes from SDF ligands and a protein PDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--protein", "-p", required=True,
                       help="Protein PDB file")
    parser.add_argument("--ligands", "-l", required=True,
                       help="Directory with SDF ligands")
    parser.add_argument("--complexes", "-c", default="complexes", 
                       help="Output directory for complexes (default: complexes)")
    
    args = parser.parse_args()
    exit(main(args))
