import os
import sys
import signal
import shutil
import uuid
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import base64
from streamlit_ketcher import st_ketcher
from streamlit_molstar import st_molstar
import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, Draw
import pandas as pd
from pdbtools import pdb_selresname, pdb_delresname
import io

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="LOSDT - Lead Optimization Tool",
    page_icon="static/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    """Application configuration constants."""

    # Directory structure
    BASE_DIR = Path("sessions")
    ALIGNED_MOLECULES_DIR = "aligned_molecules"
    COMPLEXES_DIR = "complexes"
    EM_COMPLEXES_DIR = "constrained_EM_complexes"
    RADIAL_PLOTS_DIR = "radial_plots"

    # Reaction databases
    REACTIONS = "reactions_validated_mapped_full_all_combined.csv"
    REACTANTS = "reactants.txt"

    # File naming conventions
    EXTRACTED_LIGAND_NAME = "0_extracted_ligand"
    RESULTS_CSV_NAME = "results_with_admet.csv"
    ZIP_FILENAME = "structural_data.zip"
    RECEPTOR_NAME = "receptor"

    # Security and limits
    ALLOWED_EXTENSIONS = {"pdb", "cif"}
    # SCRIPT_TIMEOUT = 3600  # seconds


# Initialize directory structure
Config.BASE_DIR.mkdir(exist_ok=True)


# Configure logging for multiprocessing
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
if "processing" not in st.session_state:
    st.session_state.processing = False
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "ketcher_smiles" not in st.session_state:
    st.session_state.ketcher_smiles = ""
if "current_structure_index" not in st.session_state:
    st.session_state.current_structure_index = 0

# =============================================================================
# INPUT VALIDATION UTILITIES
# =============================================================================


def allowed_file(filename: str) -> bool:
    """Validate file extension against whitelist."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def is_valid_smiles(smiles: str) -> bool:
    """
    Validate SMILES string using RDKit parser.

    Args:
        smiles: SMILES molecular representation

    Returns:
        True if valid SMILES, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller"""
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================


def create_session_folder() -> Tuple[str, Path]:
    """
    Create unique session directory for temporary file storage.

    Returns:
        Tuple of (session_id, session_folder_path)
    """
    session_id = str(uuid.uuid4())
    session_folder = Config.BASE_DIR / session_id
    session_folder.mkdir(exist_ok=True)
    return session_id, session_folder


def cleanup_temp_file(temp_file_path: Path) -> None:
    """Safely remove temporary file if it exists."""
    if temp_file_path.exists():
        temp_file_path.unlink()


def cleanup_session(session_id: str) -> None:
    """Remove session data after processing completion."""
    session_folder = Config.BASE_DIR / session_id

    if session_folder.exists():
        try:
            shutil.rmtree(session_folder)
            logger.info(f"Cleaned up session: {session_id}")
        except Exception as e:
            logger.error(f"Cleanup failed for {session_id}: {e}")


# =============================================================================
# PDB FILE PROCESSING
# =============================================================================


def extract_ligand_with_pdbtools(pdb_file_path: Path, output_path: Path, ligand_name: str) -> None:
    """Extract organic ligand from a protein–ligand complex using pdb-tools."""
    ligand_resname = {ligand_name}

    # Extract ligand (select residues named "*")
    with open(pdb_file_path, "r") as in_f, open(output_path, "w") as out_f:
        for line in pdb_selresname.run(in_f, ligand_resname):
            out_f.write(line)

    # Define receptor output
    receptor_output_path = pdb_file_path.parent / f"{Config.RECEPTOR_NAME}.pdb"

    # Extract receptor (delete residues named "*")
    # First pass: collect ligand atom serial numbers
    ligand_atom_serials = set()
    with open(pdb_file_path, "r") as in_f:
        for line in pdb_selresname.run(in_f, ligand_resname):
            if line.startswith(("ATOM", "HETATM")):
                serial = int(line[6:11].strip())
                ligand_atom_serials.add(serial)

    # Second pass: write receptor without ligand and filter CONECT records
    with open(pdb_file_path, "r") as in_f, open(receptor_output_path, "w") as out_f:
        for line in pdb_delresname.run(in_f, ligand_resname):
            # Filter out CONECT records that reference ligand atoms
            if line.startswith("CONECT"):
                atoms = [
                    int(line[i : i + 5].strip())
                    for i in range(6, len(line.strip()), 5)
                    if line[i : i + 5].strip().isdigit()
                ]
                # Keep CONECT only if none of the atoms are ligand atoms
                if not any(atom in ligand_atom_serials for atom in atoms):
                    out_f.write(line)
            else:
                out_f.write(line)


def _convert_pdb_to_mol_with_bonds(pdb_file_path: Path) -> Chem.Mol:
    """
    Convert PDB to RDKit molecule with proper bond determination.
    """
    mol = Chem.MolFromPDBFile(str(pdb_file_path), removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError("Could not parse ligand from PDB file")

    formal_charge = Chem.GetFormalCharge(mol)
    rdDetermineBonds.DetermineBonds(mol, charge=formal_charge)

    return mol


def extract_smiles_from_pdb(pdb_file_path: Path, ligand_name: str) -> str:
    """Extract SMILES representation from PDB file."""
    temp_ligand_path = pdb_file_path.parent / f"{pdb_file_path.stem}_temp_ligand.pdb"

    try:
        extract_ligand_with_pdbtools(pdb_file_path, temp_ligand_path, ligand_name)

        if not temp_ligand_path.exists() or temp_ligand_path.stat().st_size == 0:
            raise ValueError("No ligand extracted from PDB file")

        mol = _convert_pdb_to_mol_with_bonds(temp_ligand_path)
        smiles = Chem.MolToSmiles(mol)

        return smiles

    finally:
        cleanup_temp_file(temp_ligand_path)


def extract_ligand_from_pdb(
    pdb_file_path: Path, ligand_sdf_path: Path, ligand_pdb_path: Path, ligand_name: str
) -> None:
    """
    Extract ligand from PDB file and save in both SDF and PDB formats.
    """
    temp_ligand_path = pdb_file_path.parent / f"{pdb_file_path.stem}_temp_ligand.pdb"

    try:
        extract_ligand_with_pdbtools(pdb_file_path, temp_ligand_path, ligand_name)

        if not temp_ligand_path.exists() or temp_ligand_path.stat().st_size == 0:
            raise ValueError("No ligand extracted from PDB file")

        mol = _convert_pdb_to_mol_with_bonds(temp_ligand_path)

        writer = Chem.SDWriter(str(ligand_sdf_path))
        writer.write(mol)
        writer.close()

        shutil.copy2(temp_ligand_path, ligand_pdb_path)

    finally:
        cleanup_temp_file(temp_ligand_path)


# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================


def ensure_templates_in_path():
    """Ensure templates directory is in sys.path for imports."""
    templates_dir = get_resource_path("templates")
    if templates_dir not in sys.path:
        sys.path.insert(0, templates_dir)


def predict_admet_properties(
    smiles: str, output_csv_path: Path, protonation: bool
) -> None:
    """Run ADMET-AI prediction using direct import."""
    try:
        ensure_templates_in_path()
        from admet_predictor import predict_admet_for_reactions

        reactions_path = Path(get_resource_path(f"static/{Config.REACTIONS}"))
        reactants_path = Path(get_resource_path(f"static/{Config.REACTANTS}"))

        predict_admet_for_reactions(
            smiles,
            reactions_path,
            reactants_path,
            output_csv_path,
            protonation=protonation,
        )
    except Exception as e:
        raise RuntimeError(f"ADMET prediction failed: {str(e)}")


def align_molecules(
    template_molecule_path: Path,
    aligned_molecules_dir: Path,
    derivatives_to_align_path: Path,
    SMILES_column: str,
) -> None:
    """Run molecular alignment using direct import."""
    try:
        ensure_templates_in_path()
        from align_molecules import align_molecules_main

        aligned_molecules_dir.mkdir(exist_ok=True, parents=True)

        align_molecules_main(
            template_molecule=str(template_molecule_path),
            derivatives_to_align=str(derivatives_to_align_path),
            aligned_molecules=str(aligned_molecules_dir),
            SMILES_column=SMILES_column,
            method="parallel",
            processes=None,
        )
    except Exception as e:
        raise RuntimeError(f"Alignment failed: {str(e)}")


def create_complexes(
    receptor_pdb: Path, aligned_molecules_dir: Path, complexes_dir: Path
) -> None:
    """Create protein-ligand complexes using direct import."""
    try:
        ensure_templates_in_path()
        from complex_maker import create_complexes_main

        complexes_dir.mkdir(exist_ok=True, parents=True)

        create_complexes_main(
            protein=str(receptor_pdb),
            ligands=str(aligned_molecules_dir),
            complexes=str(complexes_dir),
        )
    except Exception as e:
        raise RuntimeError(f"Complex creation failed: {str(e)}")


def constrained_em_complexes(pdb_dir: Path, em_complexes_dir: Path) -> None:
    """Constrained energy minimization of complexes using direct import."""
    try:
        ensure_templates_in_path()
        from constrained_em import constrained_em_main

        em_complexes_dir.mkdir(exist_ok=True, parents=True)

        constrained_em_main(
            pdb_dir=str(pdb_dir), output_dir=str(em_complexes_dir), protonation=False
        )

    except Exception as e:
        raise RuntimeError(f"Complex creation failed: {str(e)}")


def plot_radials(results_csv_path: Path, radial_plots_dir: Path) -> None:
    """Generate radial plots using direct import."""
    try:
        ensure_templates_in_path()
        from plot_radials import create_radial_plots_main

        radial_plots_dir.mkdir(exist_ok=True, parents=True)

        create_radial_plots_main(
            derivatives=str(results_csv_path), output=str(radial_plots_dir)
        )
    except Exception as e:
        raise RuntimeError(f"Radial plot creation failed: {str(e)}")


def zip_files(file_paths: List[Path], zip_path: Path) -> None:
    """Create a zip archive of specified files and directories."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in file_paths:
            if path.is_file():
                zipf.write(path, path.name)
            elif path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        base_dir = path.parent
                        arcname = str(file_path.relative_to(base_dir))
                        zipf.write(file_path, arcname)


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================


def process_input_data(
    smiles_input: Optional[str], pdb_file, session_folder: Path, ligand_name: str
) -> Tuple[str, Optional[Path]]:
    """
    Validate and process input (PDB file and/or SMILES).

    Returns:
        Tuple of (validated_smiles, pdb_file_path_or_none)
    """
    pdb_file_path = None

    # Process PDB file if provided
    if pdb_file is not None:
        if not allowed_file(pdb_file.name):
            raise ValueError("Invalid file type, only PDB files are allowed!")

        pdb_file_path = session_folder / pdb_file.name
        with open(pdb_file_path, "wb") as f:
            f.write(pdb_file.getbuffer())

        # Extract SMILES from PDB
        extracted_smiles = extract_smiles_from_pdb(pdb_file_path, ligand_name=ligand_name)

        # Use extracted SMILES if no valid SMILES provided
        if not smiles_input or not is_valid_smiles(smiles_input):
            smiles_input = extracted_smiles

    # Final validation
    if not smiles_input or not is_valid_smiles(smiles_input):
        raise ValueError(
            "Invalid, missing SMILES or wrong ligand name in PDB. Please provide valid SMILES or change ligand name in PDB file."
        )

    return smiles_input, pdb_file_path


def run_molecular_pipeline(
    smiles_string: str,
    pdb_file_path: Optional[Path],
    session_folder: Path,
    energy_minimization: bool,
    protonation: bool,
    ligand_name: str,
) -> Dict:
    """
    Execute complete molecular processing pipeline.

    Returns:
        Dictionary with all result paths and metadata
    """
    results = {}

    # Step 1: ADMET prediction
    results_csv_path = session_folder / Config.RESULTS_CSV_NAME
    predict_admet_properties(smiles_string, results_csv_path, protonation)
    results["csv_path"] = results_csv_path

    # Step 2: 3D alignment and complex creation (if PDB provided)
    if pdb_file_path:
        ligand_sdf = session_folder / f"{Config.EXTRACTED_LIGAND_NAME}.sdf"
        ligand_pdb = session_folder / f"{Config.EXTRACTED_LIGAND_NAME}.pdb"
        receptor_pdb = session_folder / f"{Config.RECEPTOR_NAME}.pdb"

        extract_ligand_from_pdb(pdb_file_path, ligand_sdf, ligand_pdb, ligand_name)

        # Align molecules
        aligned_dir = session_folder / Config.ALIGNED_MOLECULES_DIR
        if protonation is True:
            align_molecules(
                ligand_sdf, aligned_dir, results_csv_path, "Protonated SMILES"
            )
        else:
            align_molecules(ligand_sdf, aligned_dir, results_csv_path, "product_SMILES")

        # Create complexes from aligned molecules
        shutil.copy2(ligand_sdf, aligned_dir)
        complexes_dir = session_folder / Config.COMPLEXES_DIR
        create_complexes(receptor_pdb, aligned_dir, complexes_dir)

        # Constrained EM complexes from complexes
        if energy_minimization is True:
            em_complexes_dir = session_folder / Config.EM_COMPLEXES_DIR
            constrained_em_complexes(complexes_dir, em_complexes_dir)

        # Create downloadable archive with both directories
        zip_path = session_folder / Config.ZIP_FILENAME
        if energy_minimization is True:
            zip_files(
                [ligand_sdf, aligned_dir, complexes_dir, em_complexes_dir], zip_path
            )
        else:
            zip_files([ligand_sdf, aligned_dir, complexes_dir], zip_path)

        results["zip_path"] = zip_path
        results["aligned_dir"] = aligned_dir
        results["complexes_dir"] = complexes_dir
        if energy_minimization is True:
            results["em_complexes_dir"] = em_complexes_dir

    # Step 3: Visualization
    if not results_csv_path.exists():
        raise RuntimeError("No molecules generated.")
    radial_dir = session_folder / Config.RADIAL_PLOTS_DIR
    plot_radials(results_csv_path, radial_dir)
    results["radial_dir"] = radial_dir

    return results


# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================


@st.cache_data
def smiles_to_base64(smiles):
    if smiles is None or pd.isna(smiles) or str(smiles).strip() == "":
        return ""

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return ""

        img = Draw.MolToImage(mol, size=(300, 300))
        if img is None:
            return ""

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    except Exception:
        return ""


@st.cache_data
def path_to_base64(image_path: Path) -> str:
    if not image_path.exists():
        return ""
    with open(image_path, "rb") as img_file:
        return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"


def color_lipinski(val):
    color = "#90ee90" if val == 4 else "#ff524d"
    return f"background-color: {color}"


def display_results(session_folder: Path, smiles_string: str, has_pdb: bool):
    """Display analysis results in Streamlit with enhanced 3D visualization."""

    # Results tabs
    tab1, tab2, tab3 = st.tabs(["Radial plots", "3D Structures", "Downloads"])

    with tab1:
        radial_dir = session_folder / Config.RADIAL_PLOTS_DIR

        if radial_dir.exists():
            st.warning("""Be cautious of ADMET-AI and Dimorphite-DL predictions, they are not always accurate and should be used as a guide rather than absolute truth. A critical review on ADMET-AI can be found here: https://openadmet.ghost.io/zero-shot-expansiorx-admet-predictions/""")
            plot_files = sorted(list(radial_dir.glob("*.png")))

            if plot_files:
                current_plot = plot_files[
                    st.slider(
                        label="Select radial plot",
                        min_value=0,
                        max_value=len(plot_files) - 1,
                    )
                ]
                col1, _, col2, _, col3 = st.columns([1, 4, 4, 3, 1])
                with col2:
                    st.image(str(current_plot), width="content")

                csv_path = session_folder / Config.RESULTS_CSV_NAME
                df = pd.read_csv(csv_path)
                display_cols = [
                    col
                    for col in df.columns
                    if col
                    not in [
                        "molecular_weight_RDkit",
                        "TPSA_RDKit",
                        "NR-AR-LBD",
                        "NR-AR",
                        "NR-AhR",
                        "NR-Aromatase",
                        "NR-ER-LBD",
                        "NR-ER",
                        "NR-PPAR-gamma",
                        "SR-ARE",
                        "SR-ATAD5",
                        "SR-HSE",
                        "SR-MMP",
                        "SR-p53",
                        "NR-AR-LBD_drugbank_approved_percentile",
                        "NR-AR_drugbank_approved_percentile",
                        "NR-AhR_drugbank_approved_percentile",
                        "NR-Aromatase_drugbank_approved_percentile",
                        "NR-ER-LBD_drugbank_approved_percentile",
                        "NR-ER_drugbank_approved_percentile",
                        "NR-PPAR-gamma_drugbank_approved_percentile",
                        "SR-ARE_drugbank_approved_percentile",
                        "SR-ATAD5_drugbank_approved_percentile",
                        "SR-HSE_drugbank_approved_percentile",
                        "SR-MMP_drugbank_approved_percentile",
                        "SR-p53_drugbank_approved_percentile",
                    ]
                ]
                display_df = df[display_cols].copy()
                display_df["structure_image"] = display_df["product_SMILES"].apply(
                    smiles_to_base64
                )
                if "Protonated SMILES" in df.columns:
                    display_df["structure_image"] = display_df[
                        "Protonated SMILES"
                    ].apply(smiles_to_base64)
                display_df["reaction_name"] = display_df["reaction_name"].str.replace(
                    r"[{}]", "", regex=True
                )
                st.dataframe(
                    display_df.style.map(color_lipinski, subset=["Lipinski"]),
                    column_config={
                        "structure_image": st.column_config.ImageColumn(
                            "Structure", width="medium", pinned=True
                        ),
                        "reaction_name": st.column_config.TextColumn(
                            "Reaction name", pinned=True
                        ),
                        "reference": st.column_config.LinkColumn(
                            "Reference", pinned=True, width="small"
                        ),
                        "product_SMILES": st.column_config.TextColumn("SMILES"),
                        "tanimoto_similarity_input_SMILES": st.column_config.ProgressColumn(
                            "Tanimoto similarity",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "molecular_weight": st.column_config.NumberColumn(
                            "Molecular weight", format="%.2f"
                        ),
                        "logP": st.column_config.NumberColumn("LogP", format="%.2f"),
                        "hydrogen_bond_acceptors": st.column_config.NumberColumn(
                            "HBA", format="%.0f"
                        ),
                        "hydrogen_bond_donors": st.column_config.NumberColumn(
                            "HBD", format="%.0f"
                        ),
                        "Lipinski": st.column_config.NumberColumn(
                            "Lipinski", format="%.0f"
                        ),
                        "QED": st.column_config.NumberColumn("QED", format="%.2f"),
                        "stereo_centers": st.column_config.NumberColumn(
                            "Stereo centers", format="%.0f"
                        ),
                        "tpsa": st.column_config.NumberColumn("TPSA", format="%.2f"),
                        "PAINS_alert": st.column_config.NumberColumn(
                            "PAINS alert",
                            format="%d",
                        ),
                        "BRENK_alert": st.column_config.NumberColumn(
                            "BRENK alert",
                            format="%d",
                        ),
                        "NIH_alert": st.column_config.NumberColumn(
                            "NIH alert",
                            format="%d",
                        ),                                                                        
                        "AMES": st.column_config.ProgressColumn(
                            "Mutagenicity (AMES)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "BBB_Martins": st.column_config.ProgressColumn(
                            "BBB permeability (Martins)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "Bioavailability_Ma": st.column_config.ProgressColumn(
                            "Bioavailability (Ma)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "CYP1A2_Veith": st.column_config.ProgressColumn(
                            "CYP1A2 inhibition (Veith)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "CYP2C19_Veith": st.column_config.ProgressColumn(
                            "CYP2C19 inhibition (Veith)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "CYP2C9_Substrate_CarbonMangels": st.column_config.ProgressColumn(
                            "CYP2C9 inhibition (CarbonMangels)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "CYP2C9_Veith": st.column_config.ProgressColumn(
                            "CYP2C9 inhibition (Veith)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "CYP2D6_Substrate_CarbonMangels": st.column_config.ProgressColumn(
                            "CYP2D6 inhibition (CarbonMangels)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "CYP2D6_Veith": st.column_config.ProgressColumn(
                            "CYP2D6 inhibition (Veith)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "CYP3A4_Substrate_CarbonMangels": st.column_config.ProgressColumn(
                            "CYP3A4 inhibition (CarbonMangels)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "CYP3A4_Veith": st.column_config.ProgressColumn(
                            "CYP3A4 inhibition (Veith)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "Carcinogens_Lagunin": st.column_config.ProgressColumn(
                            "Carcinogic (Lagunin)",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "ClinTox": st.column_config.ProgressColumn(
                            "Clinical trial toxicity",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "DILI": st.column_config.ProgressColumn(
                            "Drug Induced Liver Injury",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "HIA_Hou": st.column_config.ProgressColumn(
                            "Human Intestinal Absorption",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "PAMPA_NCATS": st.column_config.ProgressColumn(
                            "PAMPA Permeability",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "Pgp_Broccatelli": st.column_config.ProgressColumn(
                            "P-glycoprotein inhibition",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "Skin_Reaction": st.column_config.ProgressColumn(
                            "Skin reaction", format="%.2f", min_value=0.0, max_value=1.0
                        ),
                        "hERG": st.column_config.ProgressColumn(
                            "hERG inhibition",
                            format="%.2f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "Caco2_Wang": st.column_config.NumberColumn(
                            "Cell Effective Permeability (Caco-2 Wang)", format="%.2f"
                        ),
                        "Clearance_Hepatocyte_AZ": st.column_config.NumberColumn(
                            "Clearance hepatocyte (AstraZeneca)", format="%.2f"
                        ),
                        "Clearance_Microsome_AZ": st.column_config.NumberColumn(
                            "Clearance microsome (AstraZeneca)", format="%.2f"
                        ),
                        "Half_Life_Obach": st.column_config.NumberColumn(
                            "Half life (Obach)", format="%.2f"
                        ),
                        "HydrationFreeEnergy_FreeSolv": st.column_config.NumberColumn(
                            "Hydration Free Energy", format="%.2f"
                        ),
                        "LD50_Zhu": st.column_config.NumberColumn(
                            "LD50 (Zhu)", format="%.2f"
                        ),
                        "Lipophilicity_AstraZeneca": st.column_config.NumberColumn(
                            "Lipophilicity (AstraZeneca)", format="%.2f"
                        ),
                        "PPBR_AZ": st.column_config.NumberColumn(
                            "Plasma Protein Binding Rate (AstraZeneca)", format="%.2f"
                        ),
                        "Solubility_AqSolDB": st.column_config.NumberColumn(
                            "Solubility (AqSolDB)", format="%.2f"
                        ),
                        "VDss_Lombardo": st.column_config.NumberColumn(
                            "Volumn of distribution at steady state (Lombardo)",
                            format="%.2f",
                        ),
                        "molecular_weight_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Molecular weight drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "logP_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "LogP drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "hydrogen_bond_acceptors_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "HBA drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "hydrogen_bond_donors_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "HBD drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "Lipinski_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Lipinski drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "QED_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "QED drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "stereo_centers_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Stereo centers drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "tpsa_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "TPSA drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "AMES_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Mutagenicity (AMES) drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "BBB_Martins_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "BBB permeability (Martins) drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "Bioavailability_Ma_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Bioavailability (Ma) drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "CYP1A2_Veith_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "CYP1A2 inhibition (Veith) drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "CYP2C19_Veith_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "CYP2C19 inhibition (Veith) drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "CYP2C9_Substrate_CarbonMangels_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "CYP2C9 inhibition (CarbonMangels) drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "CYP2C9_Veith_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "CYP2C9 inhibition (Veith) drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "CYP2D6_Substrate_CarbonMangels_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "CYP2D6 inhibition (CarbonMangels) drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "CYP2D6_Veith_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "CYP2D6 inhibition (Veith) drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "CYP3A4_Substrate_CarbonMangels_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "CYP3A4 inhibition (CarbonMangels) drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "CYP3A4_Veith_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "CYP3A4 inhibition (Veith) drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "Carcinogens_Lagunin_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Carcinogic (Lagunin) drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "ClinTox_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Clinical trial toxicity drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "DILI_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Drug Induced Liver Injury drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "HIA_Hou_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Human Intestinal Absorption drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "PAMPA_NCATS_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "PAMPA Permeability drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "Pgp_Broccatelli_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "P-glycoprotein inhibition drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "Skin_Reaction_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Skin reaction drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "hERG_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "hERG inhibition drugbank approved percentile",
                            format="%.2f",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                        "Caco2_Wang_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Cell Effective Permeability (Caco-2 Wang) drugbank approved percentile",
                            format="%.2f",
                            max_value=100.0,
                        ),
                        "Clearance_Hepatocyte_AZ_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Clearance hepatocyte (AstraZeneca) drugbank approved percentile",
                            format="%.2f",
                            max_value=100.0,
                        ),
                        "Clearance_Microsome_AZ_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Clearance microsome (AstraZeneca) drugbank approved percentile",
                            format="%.2f",
                            max_value=100.0,
                        ),
                        "Half_Life_Obach_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Half life (Obach) drugbank approved percentile",
                            format="%.2f",
                            max_value=100.0,
                        ),
                        "HydrationFreeEnergy_FreeSolv_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Hydration Free Energy drugbank approved percentile",
                            format="%.2f",
                            max_value=100.0,
                        ),
                        "LD50_Zhu_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "LD50 (Zhu) drugbank approved percentile",
                            format="%.2f",
                            max_value=100.0,
                        ),
                        "Lipophilicity_AstraZeneca_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Lipophilicity (AstraZeneca) drugbank approved percentile",
                            format="%.2f",
                            max_value=100.0,
                        ),
                        "PPBR_AZ_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Plasma Protein Binding Rate (AstraZeneca) drugbank approved percentile",
                            format="%.2f",
                            max_value=100.0,
                        ),
                        "Solubility_AqSolDB_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Solubility (AqSolDB) drugbank approved percentile",
                            format="%.2f",
                            max_value=100.0,
                        ),
                        "VDss_Lombardo_drugbank_approved_percentile": st.column_config.ProgressColumn(
                            "Volumn of distribution at steady state (Lombardo) drugbank approved percentile",
                            format="%.2f",
                            max_value=100.0,
                        ),
                        "Volumetric shape and ESP similarity score": st.column_config.ProgressColumn(
                            format="%.2f", max_value=1.0, pinned=True
                        )
                        if "Volumetric shape and ESP similarity score" in display_df
                        else None,
                    },
                    width="stretch",
                    height=500,
                    row_height=100,
                )

            else:
                st.info("No radial plots available.")
        else:
            st.info("No radial plots generated.")

    with tab2:
        if has_pdb:
            em_complexes_dir = session_folder / Config.EM_COMPLEXES_DIR
            complexes_dir = session_folder / Config.COMPLEXES_DIR
            tab4, tab5 = st.tabs(["3D Structures", "EM 3D Structures"])

            with tab4:
                if complexes_dir.exists():
                    complex_files = sorted(list(complexes_dir.glob("*final.pdb")))

                    if complex_files:
                        # Display current structure
                        current_structure = complex_files[
                            st.slider(
                                label="Select 3D structure",
                                min_value=0,
                                max_value=len(complex_files) - 1,
                                key=1,
                            )
                        ]

                        # Display with Molstar
                        st_molstar(
                            current_structure,
                            height=500,
                            key=f"molstar_{st.session_state.current_structure_index}",
                        )
                else:
                    st.warning("No complex structures available.")

            with tab5:
                if em_complexes_dir.exists():
                    complex_em_files = sorted(
                        list(em_complexes_dir.glob("*_minimized.pdb"))
                    )

                    if complex_em_files:
                        # Display current structure
                        current_structure = complex_em_files[
                            st.slider(
                                label="Select 3D structure",
                                min_value=0,
                                max_value=len(complex_files) - 1,
                                key=2,
                            )
                        ]

                        # Display with Molstar
                        st_molstar(
                            current_structure,
                            height=500,
                            key=f"molstar_EM_{st.session_state.current_structure_index}",
                        )
                else:
                    st.warning("No energy minimized complex structures available.")

        else:
            st.warning("3D structures are only available when a PDB file is uploaded.")

    with tab3:
        # ADMET Results
        st.markdown("### ADMET Data")
        csv_path = session_folder / Config.RESULTS_CSV_NAME
        if csv_path.exists():
            st.markdown(
                "Download the ADMET properties of the derivatives. More information on the different properties: "
                "[ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview)"
            )

            with open(csv_path, "rb") as f:
                st.download_button(
                    label="Download ADMET results",
                    data=f,
                    file_name=Config.RESULTS_CSV_NAME,
                    mime="text/csv",
                )

        # Structural Data
        st.markdown("### Structural Data")

        if has_pdb:
            zip_path = session_folder / Config.ZIP_FILENAME
            if zip_path.exists():
                st.markdown(
                    "Download the structural data; Volumetric shape and ESP similarity score can be found in the last column of the ADMET results file."
                )

                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="Download Structural Data (ZIP)",
                        data=f,
                        file_name=Config.ZIP_FILENAME,
                        mime="application/zip",
                    )
        else:
            st.warning(
                "Structural data is not available because no PDB file was uploaded."
            )

    st.markdown(f"Session ID: {st.session_state.session_id}")
    st.markdown(f"Analyzed SMILES: {smiles_string}")


# =============================================================================
# MAIN APPLICATION
# =============================================================================


def main():
    # Header
    col1, col2 = st.columns([8, 1], vertical_alignment="bottom")
    with col1:
        st.markdown("# LOSDT - Lead Optimization with Safety by Design Tool")
    with col2:
        exit_app = st.button("**SHUT DOWN APP**", width="stretch", type="primary")
        if exit_app:
            os.kill(os.getpid(), signal.SIGKILL)

    # Description
    st.markdown("""
    An open-source tool for lead design and optimization, built on the principle of safety by design. 
    This versatile tool serves as both an ADMET optimizer and an idea generator for navigating beyond patent space. 
    It integrates multiple features, including ADMET-AI and ShEPhERD-score, leveraging a comprehensive bioisosteric 
    reaction library for enhanced functionality.
    """)

    # Check if results are ready
    if st.session_state.results_ready and st.session_state.session_id:
        session_folder = Config.BASE_DIR / st.session_state.session_id

        if session_folder.exists():
            display_results(
                session_folder,
                st.session_state.get("analyzed_smiles", ""),
                st.session_state.get("had_pdb", False),
            )
            col1, col2, col3 = st.columns([3, 2, 3])
            with col2:
                if st.button("**NEW ANALYSIS**", type="primary", width="stretch"):
                    # Cleanup old session
                    cleanup_session(st.session_state.session_id)

                    # Reset state
                    st.session_state.session_id = None
                    st.session_state.results_ready = False
                    st.session_state.processing = False
                    st.session_state.error_message = None
                    st.session_state.plot_index = 0
                    st.session_state.current_structure_index = 0
                    st.session_state.ketcher_smiles = ""
                    st.rerun()
        else:
            st.error("Session data not found. Please start a new analysis.")
            st.session_state.results_ready = False

    # Input form (only shown when not showing results)
    if not st.session_state.results_ready:
        # Input tabs
        input_tab1, input_tab2 = st.tabs(["PDB upload", "SMILES or molecule editor"])

        smiles_input = None
        pdb_file = None

        with input_tab1:
            st.info("""Upload a cleaned, protonated PDB file with the bioactive conformation of the ligand. Please note that covalent drugs are not supported at the moment.""")
            pdb_file = st.file_uploader(
                "Choose PDB file", type=["pdb"], label_visibility="collapsed"
            )
            ligand_name = st.text_input(
                "Ligand name in PDB file", value="*", help="The residue name of the ligand in the PDB file. If your ligand has a specific residue name (e.g., 'LIG'), please enter it here."
            )            
            energy_minimization_button = st.checkbox(
                "Perform local energy minimization on the complexes. This drastically increases computational time and is not recommended on smaller PCs, instead use the CCM module afterwards on complexes of interest."
            )

        with input_tab2:
            st.info("""Click "Apply" to input the SMILES.""")
            # Ketcher molecular editor
            smile_code = st.text_input("Enter or draw:")
            molecule_data = st_ketcher(smile_code)

            if molecule_data:
                smiles_from_editor = molecule_data
                st.session_state.ketcher_smiles = smiles_from_editor

                if smiles_from_editor and is_valid_smiles(smiles_from_editor):
                    st.success(f"Valid structure: `{smiles_from_editor}`")
                    smiles_input = smiles_from_editor
                elif smiles_from_editor:
                    st.error("Invalid molecular structure")

        protonation_button = st.checkbox("Protonate the ligand(s) using Dimorphite-DL.")

        # Submission
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("**ANALYZE MOLECULE**", type="primary", width="stretch"):
                # Validation
                if not smiles_input and pdb_file is None:
                    st.error(
                        "Please provide either a SMILES string or upload a PDB file."
                    )
                elif (
                    smiles_input
                    and not is_valid_smiles(smiles_input)
                    and pdb_file is None
                ):
                    st.error(
                        "Invalid SMILES notation. Please provide a valid SMILES or upload a PDB file."
                    )
                else:
                    # Start processing
                    st.session_state.processing = True

                    with st.spinner(
                        "Processing data...", width="stretch", show_time=True
                    ):
                        try:
                            # Create session
                            session_id, session_folder = create_session_folder()
                            st.session_state.session_id = session_id

                            # Process inputs
                            validated_smiles, pdb_path = process_input_data(
                                smiles_input, pdb_file, session_folder, ligand_name=ligand_name
                            )

                            # Run pipeline
                            run_molecular_pipeline(
                                validated_smiles,
                                pdb_path,
                                session_folder,
                                energy_minimization=energy_minimization_button,
                                protonation=protonation_button,
                                ligand_name=ligand_name,
                            )

                            # Store results
                            st.session_state.results_ready = True
                            st.session_state.analyzed_smiles = validated_smiles
                            st.session_state.had_pdb = pdb_path is not None
                            st.session_state.processing = False

                            st.rerun()

                        except Exception as e:
                            st.session_state.processing = False
                            st.error(f"Error during processing: {str(e)}")
                            logger.error(f"Pipeline error: {str(e)}", exc_info=True)

                            # Cleanup on error
                            if st.session_state.session_id:
                                cleanup_session(st.session_state.session_id)
                                st.session_state.session_id = None

    # References section (always visible)
    st.markdown("### References for open-source tools")

    with st.expander("View references", expanded=False):
        st.markdown("""
        **[ADMET-AI](https://github.com/swansonk14/admet_ai):**

        Swanson, K.; Walther, P.; Leitz, J.; Mukherjee, S.; Wu, J. C.; Shivnaraine, R. V.; Zou, J. 
        ADMET-AI: A Machine Learning ADMET Platform for Evaluation of Large-Scale Chemical Libraries. 
        *Bioinformatics* 2024.
        [https://doi.org/10.1093/bioinformatics/btae416](https://doi.org/10.1093/bioinformatics/btae416)

        ---

        **[ShEPhERD](https://github.com/coleygroup/shepherd-score):**

        Adams, K.; Abeywardane, K.; Fromer, J.; Coley, C. W. ShEPhERD: diffusing shape, electrostatics, 
        and pharmacophores for bioisosteric drug design. *Arxiv.org*.
        [https://arxiv.org/html/2411.04130v1](https://arxiv.org/html/2411.04130v1)

        ---

        **[Dimorphite-DL](https://github.com/durrantlab/dimorphite_dl):**

        Ropp PJ, Kaminsky JC, Yablonski S, Durrant JD (2019) Dimorphite-DL: An open-source program for enumerating the ionization states of drug-like small molecules. J Cheminform 11:14. doi: [10.1186/s13321-019-0336-9](https://link.springer.com/article/10.1186/s13321-019-0336-9)

        ---

        **[OpenMM](https://openmm.org/):**

        A high-performance toolkit for molecular simulation. Use it as an application, a library, or a flexible programming environment. We include extensive language bindings for Python, C, C++, and even Fortran. 

        ---

        **[RDKit](https://github.com/rdkit/rdkit):**

        The RDKit is a collection of cheminformatics and machine-learning software written in C++ and Python.

        ---

        **[pdb-tools](https://github.com/haddocking/pdb-tools):**

        A swiss army knife for manipulating and editing PDB files.

        ---

        **[streamlit](https://github.com/streamlit/streamlit):**

        Python-based webserver builder.

        ---

        **[streamlit-ketcher](https://github.com/mik-laj/streamlit-ketcher):**

        Streamlit components that adds the ability to draw chemical compounds.

        ---

        **[streamlit-molstar](https://github.com/pragmatic-streamlit/streamlit-molstar):**

        Mol* is a modern web-based open-source toolkit for visualisation and analysis of large-scale molecular data.
        """)

    # Footer
    # st.markdown("""
    # <div style='text-align: center; color: #666; padding: 2rem 0;'>
    #    <p>&copy; 2025 LOSDT. All rights reserved.</p>
    # </div>
    # """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
