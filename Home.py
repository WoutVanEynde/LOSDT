"""
LOSDT - Lead Optimization Structure-based Drug Design Tool
A Streamlit application with full molecular editing and 3D visualization.

This enhanced version includes:
- Streamlit-Ketcher for molecular editing
- Streamlit-Molstar for 3D structure visualization
- All original ADMET prediction and analysis features
"""

from streamlit_ketcher import st_ketcher
from streamlit_molstar import st_molstar
import streamlit as st

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="LOSDT - Lead Optimization Tool",
    page_icon="static/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Header
    col1, col2 = st.columns([8, 1], vertical_alignment="bottom")
    with col1:
        st.markdown("# Wout's toolkit")
    with col2:
        exit_app = st.button("**SHUT DOWN APP**", width="stretch", type="primary")
        if exit_app:
            import os, signal
            os.kill(os.getpid(), signal.SIGKILL)

    # Description
    st.markdown("""
    An open-source toolkit for lead design and optimization, which consists of following tools: 
    
    ---
    
    **MCSAlign - Maximum Common Substructure Alignment:**
    
    An RDKit based alignment tool using an iterative approach of finding Maximum Common Substructures between the 3D template and SMILES-derivatives, copy-pasting the common substructure and repeating on the leftover template. 
    
    ---
    
    **CCM - Constrained Complex Minimization:** 
    
    An OpenMM based energy minimization of complexes with a restrained environment, allowing relatively quick optimization of the ligand-protein-solvent complex.
    
    ---
    
    **LOSDT - Lead Optimization by Safty by Design Tool:** 
    
    Applies a manually curated bioisostere library on a given compound of interest - SMILES or PDB complex - and uses ADMET-AI in order to compare the ADMET properties. If given a complex, it would use the previously mentioned tools CCM and MCSalign to generate the proposed binding modes of the bioisostere derivatives.
    """)

    # Footer
    #st.markdown("""
    #<div style='text-align: center; color: #666; padding: 2rem 0;'>
    #    <p>&copy; 2025 LOSDT. All rights reserved.</p>
    #</div>
    #""", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()    
