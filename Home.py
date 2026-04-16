import streamlit as st

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="LOSDT - Lead Optimization Tool",
    page_icon="static/icon.png",
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
    
**MCSAlign - Maximum Common Substructure Alignment:**
    
An RDKit based alignment tool using an iterative approach of finding Maximum Common Substructures between the 3D template and SMILES-derivatives, copy-pasting the common substructure and iteratively repeating on the leftover template.
                
**LOSDT - Lead Optimization by Safty by Design Tool:** 
    
An open-source tool for lead design and optimization, built on the principle of safety by design. This tool serves as both an ADMET optimizer and an idea generator for navigating beyond patent space. It integrates multiple features, including ADMET-AI and ShEPhERD-score, leveraging a comprehensive bioisosteric reaction library for functionality.

Options for input are 2D compound or 3D ligand in bioactive confirmation inside protein complex. In case of the latter, the bioisosteric derivatives are automatically aligned to the input confirmation using the MCSAlign module. An option for both protonation with pKaLearn at pH 7.0 and energy minimized with the CCM module are given.    
                
**CCM - Constrained Complex Minimization:** 
    
An OpenMM based energy minimization of complexes with a restrained environment, allowing relatively quick optimization of the ligand-protein-solvent complex. An option is provided to protonate the ligand using pKaLearn at pH 7.0. The following forcefields are used:

- Protein: ff19SB
- DNA: OL21
- Lipids: lipids21
- Waters: OPC3
- Small molecules: Sage 2.3.0 with AshGC neural network charge model                            
    """)

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

        **[pKaLearn](https://github.com/MoitessierLab/pKaLearn):**

        Genzling, J., Luo, Z., Weiser, B. et al. Development of a pKa predictor (pKaLearn) by leveraging teaching experience to improve machine learning. Commun Chem (2026). ; [https://www.nature.com/articles/s42004-026-01983-y](https://www.nature.com/articles/s42004-026-01983-y)

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
