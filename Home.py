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

    # Footer
    #st.markdown("""
    #<div style='text-align: center; color: #666; padding: 2rem 0;'>
    #    <p>&copy; 2025 LOSDT. All rights reserved.</p>
    #</div>
    #""", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()    
