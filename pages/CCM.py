import os
import sys
import multiprocessing
import subprocess
import shutil
import uuid
import zipfile
import logging
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from streamlit_molstar import st_molstar
import streamlit as st

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="CCM - Constrained Complex Minimization",
    page_icon="static/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration constants."""
    # Directory structure
    BASE_DIR = Path('sessions')
    COMPLEXES_DIR = 'complexes'
    EM_COMPLEXES_DIR = 'constrained_EM_complexes'
    
    # File naming conventions
    ZIP_FILENAME = 'structural_data.zip'
    
    # Security and limits
    ALLOWED_EXTENSIONS = {'pdb'}

# Initialize directory structure
Config.BASE_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'current_structure_index' not in st.session_state:
    st.session_state.current_structure_index = 0
    
# =============================================================================
# INPUT VALIDATION UTILITIES
# =============================================================================

def allowed_file(filename: str) -> bool:
    """Validate file extension against whitelist."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller"""
    if getattr(sys, 'frozen', False):
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

def save_uploaded_file(uploaded_file, session_folder: Path) -> Path:
    """
    Save Streamlit UploadedFile to session folder.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        session_folder: Path to session directory
        
    Returns:
        Path to saved file
    """
    if uploaded_file is None:
        return None
    
    file_path = session_folder / Path(uploaded_file.name).name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def save_uploaded_files(uploaded_files, session_folder: Path) -> List[Path]:
    """
    Save a list of Streamlit UploadedFile objects to session folder.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        session_folder: Path to session directory

    Returns:
        List of paths to saved files
    """
    if not uploaded_files:
        return []

    session_folder.mkdir(exist_ok=True, parents=True)
    return [save_uploaded_file(f, session_folder) for f in uploaded_files]
            
# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def ensure_templates_in_path():
    """Ensure templates directory is in sys.path for imports."""
    templates_dir = get_resource_path('templates')
    if templates_dir not in sys.path:
        sys.path.insert(0, templates_dir)
        
def constrained_em_complexes(pdb_dir: Path, 
                            em_complexes_dir: Path,
                            protonation: bool) -> None:
    """Constrained energy minimization of complexes using direct import."""
    try:
        ensure_templates_in_path()
        from constrained_em import constrained_em_main
        
        em_complexes_dir.mkdir(exist_ok=True, parents=True)
        
        constrained_em_main(
            pdb_dir=str(pdb_dir),
            output_dir=str(em_complexes_dir),
            protonation=protonation
        )
        
    except Exception as e:
        raise RuntimeError(f"Complex creation failed: {str(e)}")        
        
def zip_files(file_paths: List[Path], zip_path: Path) -> None:
    """Create a zip archive of specified files and directories."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in file_paths:
            if path.is_file():
                zipf.write(path, path.name)
            elif path.is_dir():
                for file_path in path.rglob('*'):
                    if file_path.is_file():
                        base_dir = path.parent
                        arcname = str(file_path.relative_to(base_dir))
                        zipf.write(file_path, arcname)        
        
def run_molecular_pipeline(complexes_dir: Path, 
                        session_folder: Path,
                        protonation: bool) -> Dict:
    """
    Execute complete molecular processing pipeline.
    
    Returns:
        Dictionary with all result paths and metadata
    """
    results = {}

    # Align molecules
    em_complexes_dir = session_folder / Config.EM_COMPLEXES_DIR
    
    constrained_em_complexes(complexes_dir, em_complexes_dir, protonation)    
        
    # Create downloadable archive with both directories
    zip_path = session_folder / Config.ZIP_FILENAME
    zip_files([complexes_dir, em_complexes_dir], zip_path)
        
    results['zip_path'] = zip_path
    results['em_complexes_dir'] = em_complexes_dir
    
    return results

def remove_ace_nme_lines_in_place(cif_path: Path) -> Path:
    """
    Overwrites the original .cif file:
    Completely removes every ATOM / HETATM line that belongs to ACE or NME.
    Returns the same path (now modified).
    """
    lines = cif_path.read_text(encoding="utf-8").splitlines(keepends=False)
    kept_lines = []
    removed_count = 0

    for line in lines:
        # Skip header-like lines and non-coordinate records
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            kept_lines.append(line)
            continue

        # Check for ACE or NME in residue name field
        # Typical mmCIF/PDB-like column: resname around columns 18-20 (0-based 17:20)
        # But we use string search — robust for most cases
        if any(tag in line for tag in [" ACE ", " NME "]):
            removed_count += 1
            continue  # drop this line

        kept_lines.append(line)

    if removed_count > 0:
        cif_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        print(f"Removed {removed_count} ACE/NME lines from: {cif_path.name}")
    else:
        print(f"No ACE/NME lines found in: {cif_path.name}")

    return cif_path

def display_results(session_folder: Path):
    """
    Display results of molecular alignment.
    
    Args:
        session_folder: Path to session directory
    """
    # Display summary
    em_complexes_dir = session_folder / Config.EM_COMPLEXES_DIR
    if em_complexes_dir.exists():
        complex_em_files = sorted(list(em_complexes_dir.glob('*_minimized.pdb')))
        if complex_em_files:
            #for cif_file in complex_em_files:
            #    remove_ace_nme_lines_in_place(cif_file)
        # Display current structure
            if len(complex_em_files) == 1:
                current_structure = complex_em_files[0]
            else:
                current_structure = complex_em_files[st.slider(label="Select 3D structure", min_value=0, max_value=len(complex_em_files) - 1)]
                    
            # Display with Molstar
            st_molstar(current_structure,height=500,key=f"molstar_EM_{st.session_state.current_structure_index}")
        else:
            st.warning("No energy minimized complex structures available.") 
    
    # Show download button for results
    zip_path = session_folder / Config.ZIP_FILENAME
    if zip_path.exists():
        with open(zip_path, 'rb') as f:
            st.download_button(
                label="**Download Energy Minimized Complexes**",
                data=f,
                file_name=Config.ZIP_FILENAME,
                mime="application/zip",
                use_container_width=True
            )

def main():
    col1, col2 = st.columns([8, 1], vertical_alignment="bottom")
    with col1:
        st.markdown("# CCM - Constrained Complex Minimization")
    with col2:
        exit_app = st.button("**SHUT DOWN APP**", width="stretch", type="primary")
        if exit_app:
            os.kill(os.getpid(), signal.SIGKILL)
    st.markdown("An OpenMM based energy minimization of complexes with a restrained environment, allowing relatively quick optimization of the ligand-protein-solvent complex.")
    # Check if results are ready
    if st.session_state.results_ready and st.session_state.session_id:
        session_folder = Config.BASE_DIR / st.session_state.session_id
        
        if session_folder.exists():
            display_results(
                session_folder
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
                    st.session_state.current_structure_index = 0
                    st.rerun()
        else:
            st.error("Session data not found. Please start a new analysis.")
            st.session_state.results_ready = False

    # Input form (only shown when not showing results)
    if not st.session_state.results_ready:
        
        pdb_inputs = None
        
        st.info("""Upload PDB file(s) with the bioactive conformation of the ligand.""")
        pdb_inputs = st.file_uploader(
            "Choose SDF file",
            type=['pdb'],
            label_visibility="collapsed",
            accept_multiple_files="directory"
        )
        
        protonation_button = st.checkbox("Protonate the ligand(s) using Dimorphite-DL.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("**MINIMIZE COMPLEXES**", type="primary", width="stretch"):
                # Validation
                if not pdb_inputs:
                    st.error("Please provide PDB file(s).")
                else:
                    # Start processing
                    st.session_state.processing = True
                    
                    with st.spinner("Processing data...", width="stretch", show_time=True):
                        try:
                            # Create session
                            session_id, session_folder = create_session_folder()
                            st.session_state.session_id = session_id
                            
                            # Create folder
                            complexes_dir = session_folder / Config.COMPLEXES_DIR
                            complexes_dir.mkdir(exist_ok=True, parents=True)
                            
                            # Save uploaded files to session folder
                            save_uploaded_files(pdb_inputs, complexes_dir)
                            
                            # Run pipeline with file paths
                            if protonation_button:
                                results = run_molecular_pipeline(
                                    complexes_dir,
                                    session_folder,
                                    protonation=True
                                )
                                
                            else:
                                results = run_molecular_pipeline(
                                    complexes_dir,
                                    session_folder,
                                    protonation=False
                                )                                
                            
                            # Store results
                            st.session_state.results_ready = True
                            st.session_state.had_pdb = False
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
    #st.markdown("""
    #<div style='text-align: center; color: #666; padding: 2rem 0;'>
    #    <p>&copy; 2025 LOSDT. All rights reserved.</p>
    #</div>
    #""", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()