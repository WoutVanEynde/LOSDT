import os
import sys
import shutil
import uuid
import zipfile
import logging
import signal
from pathlib import Path
from typing import Dict, Tuple
import streamlit as st

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="MCSAlign - Maximum Common Substructure Alignment",
    page_icon="static/icon.png",
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
    ALIGNED_MOLECULES_DIR = 'aligned_molecules'
    
    # File naming conventions
    ZIP_FILENAME = 'structural_data.zip'
    
    # Security and limits
    ALLOWED_EXTENSIONS = {'csv', 'sdf', 'pdb'}

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
    
    file_path = session_folder / uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    logger.info(f"Saved uploaded file: {uploaded_file.name}")
    return file_path
            
# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def ensure_templates_in_path():
    """Ensure templates directory is in sys.path for imports."""
    templates_dir = get_resource_path('templates')
    if templates_dir not in sys.path:
        sys.path.insert(0, templates_dir)
        
def align_molecules(sdf_input: Path,
                   csv_input: Path,
                   aligned_dir: Path,
                   SMILES_column: str
                   ) -> None:
    """Run molecular alignment using direct import."""
    try:
        ensure_templates_in_path()
        from align_molecules import align_molecules_main
                
        aligned_dir.mkdir(exist_ok=True, parents=True)
        
        align_molecules_main(
            template_molecule=str(sdf_input),
            derivatives_to_align=str(csv_input),
            aligned_molecules=str(aligned_dir),
            SMILES_column=SMILES_column
        )
    except Exception as e:
        raise RuntimeError(f"Alignment failed: {str(e)}")
        
def run_molecular_pipeline(sdf_input: Path, 
                          csv_input: Path, 
                          session_folder: Path,
                          SMILES_column: str) -> Dict:
    """
    Execute complete molecular processing pipeline.
    
    Returns:
        Dictionary with all result paths and metadata
    """
    results = {}
    
    # Align molecules
    aligned_dir = session_folder / Config.ALIGNED_MOLECULES_DIR
    align_molecules(sdf_input, csv_input, aligned_dir, SMILES_column)
        
    # Create downloadable archive with aligned molecules
    zip_path = session_folder / Config.ZIP_FILENAME
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the template SDF file
        zipf.write(sdf_input, arcname=f"{Config.ALIGNED_MOLECULES_DIR}/{sdf_input.name}")
        
        # Add all aligned molecules
        if aligned_dir.exists():
            for file_path in aligned_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(session_folder)
                    zipf.write(file_path, arcname=str(arcname))
        
    results['zip_path'] = zip_path
    results['aligned_dir'] = aligned_dir
    
    return results

def display_results(session_folder: Path):
    """
    Display results of molecular alignment.
    
    Args:
        session_folder: Path to session directory
    """
    # Display summary
    aligned_dir = session_folder / Config.ALIGNED_MOLECULES_DIR
    if aligned_dir.exists():
        aligned_files = list(aligned_dir.glob('*.sdf'))
        st.success(f"Generated {len(aligned_files)} aligned structure(s)")
    
    # Show download button for results
    zip_path = session_folder / Config.ZIP_FILENAME
    if zip_path.exists():
        with open(zip_path, 'rb') as f:
            st.download_button(
                label="**Download Aligned Molecules**",
                data=f,
                file_name=Config.ZIP_FILENAME,
                mime="application/zip",
                use_container_width=True
            )

def main():
    # Header
    col1, col2 = st.columns([8, 1], vertical_alignment="bottom")
    with col1:
        st.markdown("# MCSAlign - Maximum Common Substructure Alignment")
    with col2:
        exit_app = st.button("**SHUT DOWN APP**", width="stretch", type="primary")
        if exit_app:
            os.kill(os.getpid(), signal.SIGKILL)
    st.markdown("An RDKit based alignment tool using an iterative approach of finding Maximum Common Substructures between the 3D template and SMILES-derivatives, copy-pasting the common substructure and repeating on the leftover template.")
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
                    st.session_state.plot_index = 0
                    st.session_state.current_structure_index = 0
                    st.session_state.ketcher_smiles = ""
                    st.rerun()
        else:
            st.error("Session data not found. Please start a new analysis.")
            st.session_state.results_ready = False

    # Input form (only shown when not showing results)
    if not st.session_state.results_ready:
        
        sdf_input = None
        csv_input = None
        smiles_column_input = None 
        
        st.info("""Upload an SDF file with the bioactive conformation of the ligand and a CSV file with the derivatives.""")
        sdf_input = st.file_uploader(
            "Choose SDF file",
            type=['sdf'],
            label_visibility="collapsed"
        )
        csv_input = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            label_visibility="collapsed"
        )
        smiles_column_input = st.text_input("SMILES column name:")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("**ALIGN MOLECULES**", type="primary", width="stretch"):
                # Validation
                if not sdf_input or not csv_input:
                    st.error("Please provide both SDF and CSV files.")
                elif not smiles_column_input:
                    st.error("Please provide the SMILES column name.")
                else:
                    # Start processing
                    st.session_state.processing = True
                    
                    with st.spinner("Processing data...", width="stretch", show_time=True):
                        try:
                            # Create session
                            session_id, session_folder = create_session_folder()
                            st.session_state.session_id = session_id
                            
                            # Save uploaded files to session folder
                            sdf_path = save_uploaded_file(sdf_input, session_folder)
                            csv_path = save_uploaded_file(csv_input, session_folder)
                            
                            # Run pipeline with file paths
                            run_molecular_pipeline(
                                sdf_path, 
                                csv_path, 
                                session_folder,
                                smiles_column_input
                            )
                            
                            # Store results
                            st.session_state.results_ready = True
                            st.session_state.analyzed_smiles = ""  # Update this as needed
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
        **[RDKit](https://github.com/rdkit/rdkit):**

        The RDKit is a collection of cheminformatics and machine-learning software written in C++ and Python.

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
