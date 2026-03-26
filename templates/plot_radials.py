import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import time

from rdkit import Chem
from rdkit.Chem import Draw

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
    """Configuration constants for plotting."""
    # Figure dimensions
    TOTAL_WIDTH = 12
    TOTAL_HEIGHT = 6
    MOL_WIDTH = 4
    MOL_HEIGHT = 4
    RADIAL_WIDTH = 4
    RADIAL_HEIGHT = 4
    
    # Plotting parameters
    MOLECULE_IMAGE_SIZE = (400, 400)
    PLOT_DPI = 300
    PADDING = 0.1
    MAX_PERCENTILE = 100
    
    # Radial plot levels
    RADIAL_LEVELS = [0, 50, 100]
    
    # Colors
    REFERENCE_COLOR = "red"
    CURRENT_COLOR = "blue"
    REFERENCE_ALPHA = 0.1
    CURRENT_ALPHA = 0.25
    GRID_COLOR = "black"
    GRID_ALPHA = 0.3

class PropertyConfig:
    """Configuration for molecular properties."""
    PROPERTIES = {
        "Blood-Brain\nBarrier Safe": {
            "column": "BBB_Martins_drugbank_approved_percentile",
            "invert": True,
            "vertical_alignment": "center"
        },
        "Soluble": {
            "column": "Solubility_AqSolDB_drugbank_approved_percentile",
            "invert": False,
            "vertical_alignment": "top"
        },
        "Bio-\navailable": {
            "column": "Bioavailability_Ma_drugbank_approved_percentile",
            "invert": False,
            "vertical_alignment": "top"
        },
        "hERG\nSafe": {
            "column": "hERG_drugbank_approved_percentile",
            "invert": True,
            "vertical_alignment": "bottom"
        },
        "Lipophilic": {
            "column": "Lipophilicity_AstraZeneca_drugbank_approved_percentile",
            "invert": False,
            "vertical_alignment": "bottom"
        },
        "3D Shape &\nElectrostatic Similarity": {
            "column": "Volumetric_shape_and_ESP_similarity_score_percentile",
            "invert": False,
            "vertical_alignment": "top"
        }
    }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def prepare_molecule_for_plotting(mol_input: str | Chem.Mol) -> Chem.Mol:
    """Prepare molecule for plotting."""
    if isinstance(mol_input, str):
        mol = Chem.MolFromSmiles(mol_input)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {mol_input}")
    else:
        mol = mol_input
    
    Chem.rdDepictor.Compute2DCoords(mol)
    return mol

def extract_property_values(row_data: Dict[str, Any], 
                          available_properties: List[str]) -> Dict[str, float]:
    """Extract property values from row data."""
    property_values = {}
    
    for prop_name, prop_config in PropertyConfig.PROPERTIES.items():
        column_name = prop_config["column"]
        
        # Skip if property not available
        if column_name not in available_properties:
            continue
        
        value = row_data.get(column_name, 0)
        
        # Apply inversion if needed
        if prop_config["invert"]:
            value = Config.MAX_PERCENTILE - value
        
        property_values[prop_name] = value
    
    return property_values

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def create_figure_layout() -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create figure with molecule and radial plot axes."""
    fig = plt.figure(figsize=(Config.TOTAL_WIDTH, Config.TOTAL_HEIGHT))
    
    # Calculate positions
    mol_left = Config.PADDING
    mol_bottom = (Config.TOTAL_HEIGHT - Config.MOL_HEIGHT) / Config.TOTAL_HEIGHT / 2
    mol_width_frac = Config.MOL_WIDTH / Config.TOTAL_WIDTH
    mol_height_frac = Config.MOL_HEIGHT / Config.TOTAL_HEIGHT
    
    radial_left = mol_left + mol_width_frac + Config.PADDING
    radial_bottom = (Config.TOTAL_HEIGHT - Config.RADIAL_HEIGHT) / Config.TOTAL_HEIGHT / 2
    radial_width_frac = Config.RADIAL_WIDTH / Config.TOTAL_WIDTH
    radial_height_frac = Config.RADIAL_HEIGHT / Config.TOTAL_HEIGHT
    
    # Create axes
    mol_ax = fig.add_axes([mol_left, mol_bottom, mol_width_frac, mol_height_frac])
    radial_ax = fig.add_axes([radial_left, radial_bottom, radial_width_frac, radial_height_frac], polar=True)
    
    return fig, mol_ax, radial_ax

def plot_molecule_structure(ax: plt.Axes, mol: Chem.Mol) -> None:
    """Plot molecule structure on given axes."""
    img = Draw.MolToImage(mol, size=Config.MOLECULE_IMAGE_SIZE)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title("Structure", pad=10)

def calculate_radial_positions(num_properties: int) -> List[float]:
    """Calculate angular positions for radial plot."""
    angles = np.linspace(0, 2 * np.pi, num_properties, endpoint=False)
    angles = (angles + np.pi/2) % (2 * np.pi)
    return angles.tolist()

def setup_radial_grid(ax: plt.Axes, angles: List[float]) -> None:
    """Setup grid for radial plot."""
    # Add concentric circles - close the circle by adding first angle at the end
    angles_closed = angles + [angles[0]]
    for level in Config.RADIAL_LEVELS:
        hex_points = level * np.ones(len(angles_closed))
        ax.plot(angles_closed, hex_points, '--', color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA)
        ax.text(0, level, str(level), ha='right', va='center')
    
    # Add radial lines - draw all radial lines
    for angle in angles:
        ax.plot([angle, angle], [0, 100], '--', color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA)

def plot_radial_data(ax: plt.Axes, 
                    angles: List[float], 
                    percentiles: List[float], 
                    property_names: List[str],
                    label: str,
                    color: str,
                    alpha: float,
                    reference_data: Optional[List[float]] = None) -> None:
    """Plot radial data with optional reference."""
    # Close the polygon by adding the first point to the end
    percentiles_closed = percentiles + percentiles[:1]
    angles_closed = angles + [angles[0]]
    
    # Plot reference if provided
    if reference_data is not None:
        ref_percentiles_closed = reference_data + reference_data[:1]
        ax.fill(angles_closed, ref_percentiles_closed, color=Config.REFERENCE_COLOR, alpha=Config.REFERENCE_ALPHA)
        ax.plot(angles_closed, ref_percentiles_closed, color=Config.REFERENCE_COLOR, linewidth=2, label="Reference")
    
    # Plot current data
    ax.fill(angles_closed, percentiles_closed, color=color, alpha=alpha)
    ax.plot(angles_closed, percentiles_closed, color=color, linewidth=2, label=label)
    
    # Configure axes
    ax.set_ylim(0, 100)
    ax.grid(False)
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    # Fix: Use all angles for ticks, not angles[:-1]
    ax.set_xticks(angles)  # Use all angles, not angles[:-1]
    ax.set_xticklabels(property_names)
    
    # Adjust label orientations
    _adjust_radial_labels(ax, angles)

def _adjust_radial_labels(ax: plt.Axes, angles: List[float]) -> None:
    """Adjust radial axis labels for better readability."""
    for label, angle in zip(ax.get_xticklabels(), angles):
        if abs(angle - np.pi/2) < 0.01 or abs(angle - 3*np.pi/2) < 0.01:
            ha = 'center'
        else:
            ha = 'right' if np.pi/2 < angle < 3*np.pi/2 else 'left'
        
        label.set_rotation(np.degrees(angle))
        label.set_ha(ha)

def add_legend_and_title(ax: plt.Axes, reaction_name: str, has_reference: bool) -> None:
    """Add legend and title to the plot."""
    if has_reference:
        ax.legend(loc='lower right', bbox_to_anchor=(0.1, -0.1))

def save_plot(fig: plt.Figure, output_path: Path, filename: str) -> None:
    """Save plot to file."""
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    
    fig.savefig(file_path,
                dpi=Config.PLOT_DPI,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close(fig)

def plot_combined_visualization(mol: str | Chem.Mol,
                              property_percentiles: Dict[str, float],
                              reaction_name: str,
                              file_id: str,
                              reference_data: Optional[Dict[str, float]] = None,
                              reference_name: Optional[str] = None,
                              save_path: str = "plots") -> None:
    """Create and save combined visualization with molecule structure and radial plot."""
    try:
        # Prepare molecule
        mol_obj = prepare_molecule_for_plotting(mol)
        
        # Define the properties we want to plot
        properties = {
            "Blood-Brain\nBarrier Safe": {
                "percentile": Config.MAX_PERCENTILE - property_percentiles.get("BBB_Martins_drugbank_approved_percentile", 50),
            },
            "Soluble": {
                "percentile": property_percentiles.get("Solubility_AqSolDB_drugbank_approved_percentile", 50),
            },
            "Bio-\navailable": {
                "percentile": property_percentiles.get("Bioavailability_Ma_drugbank_approved_percentile", 50),
            },
            "hERG\nSafe": {
                "percentile": Config.MAX_PERCENTILE - property_percentiles.get("hERG_drugbank_approved_percentile", 50),
            },
            "Lipophilic": {
                "percentile": property_percentiles.get("Lipophilicity_AstraZeneca_drugbank_approved_percentile", 50),
            },
        }
        
        # Add 3D similarity if available
        if "Volumetric_shape_and_ESP_similarity_score_percentile" in property_percentiles:
            properties["3D Shape &\nElectrostatic Similarity"] = {
                "percentile": property_percentiles["Volumetric_shape_and_ESP_similarity_score_percentile"],
            }
        
        # Prepare reference data if provided
        reference_percentiles = None
        if reference_data is not None:
            reference_percentiles = []
            for prop_name in properties.keys():
                if prop_name == "Blood-Brain\nBarrier Safe":
                    ref_val = Config.MAX_PERCENTILE - reference_data.get("BBB_Martins_drugbank_approved_percentile", 50)
                elif prop_name == "Soluble":
                    ref_val = reference_data.get("Solubility_AqSolDB_drugbank_approved_percentile", 50)
                elif prop_name == "Bio-\navailable":
                    ref_val = reference_data.get("Bioavailability_Ma_drugbank_approved_percentile", 50)
                elif prop_name == "hERG\nSafe":
                    ref_val = Config.MAX_PERCENTILE - reference_data.get("hERG_drugbank_approved_percentile", 50)
                elif prop_name == "Lipophilic":
                    ref_val = reference_data.get("Lipophilicity_AstraZeneca_drugbank_approved_percentile", 50)                    
                elif prop_name == "3D Shape &\nElectrostatic Similarity":
                    ref_val = reference_data.get("Volumetric_shape_and_ESP_similarity_score_percentile", 50)
                else:
                    ref_val = 50
                reference_percentiles.append(ref_val)
        
        # Extract percentiles for plotting
        property_names = list(properties.keys())
        percentiles = [properties[prop]["percentile"] for prop in property_names]
        
        # Create figure
        fig, mol_ax, radial_ax = create_figure_layout()
        
        # Plot molecule
        plot_molecule_structure(mol_ax, mol_obj)
        
        # Setup radial plot
        angles = calculate_radial_positions(len(property_names))
        setup_radial_grid(radial_ax, angles)
        
        # Plot radial data
        plot_radial_data(radial_ax, angles, percentiles, property_names, 
                        reaction_name, Config.CURRENT_COLOR, Config.CURRENT_ALPHA,
                        reference_percentiles)
        
        # Add legend and title
        add_legend_and_title(radial_ax, reaction_name, reference_data is not None)
        
        # Set main title
        plt.suptitle(f"Reaction: {reaction_name}", fontsize=14, y=1.05)
        
        # Save plot
        filename = f"{int(file_id):03d}.png"
        save_plot(fig, Path(save_path), filename)
        
        return f"Successfully created plot: {filename}"
        
    except Exception as e:
        error_msg = f"Error creating plot for {reaction_name}: {e}"
        logger.error(error_msg)
        return error_msg

# =============================================================================
# FUNCTIONS
# =============================================================================

def process_single_reaction(args):
    """
    Process a single reaction
    
    Args:
        args: Tuple containing (row_data, reference_data, save_path, is_reference)
    
    Returns:
        Tuple: (success: bool, message: str, index: int)
    """
    row_data, reference_data, save_path, is_reference = args
    
    try:
        # Extract data from row
        idx = row_data['index']
        reaction_name = row_data['reaction_name']
        file_id = str(idx)
        
        # Get percentile properties for this row
        property_dict = {}
        for col_name, value in row_data.items():
            if col_name.endswith('_percentile'):
                property_dict[col_name] = value
        
        # Create plot
        if is_reference:
            # First row - no reference comparison
            result = plot_combined_visualization(
                row_data['product_SMILES'],
                property_dict,
                reaction_name,
                file_id=file_id,
                save_path=save_path
            )
        else:
            # Other rows - compare with reference
            result = plot_combined_visualization(
                row_data['product_SMILES'],
                property_dict,
                reaction_name,
                file_id=file_id,
                reference_data=reference_data,
                save_path=save_path
            )
        
        return (True, result, idx)
        
    except Exception as e:
        error_msg = f"Error processing reaction {row_data.get('index', 'unknown')} ({row_data.get('reaction_name', 'unknown')}): {e}"
        return (False, error_msg, row_data.get('index', -1))

def analyze_reactions(df: pd.DataFrame, save_path: str = "plots") -> None:
    """
    Analyze all reactions.
    
    Args:
        df: DataFrame containing reaction data
        save_path: Path to save plots
    """
    logger.info(f"Starting analysis of {len(df)} reactions")
    
    # Get reference data from first row
    reference_row = df.iloc[0]
    reference_name = reference_row['reaction_name']
    logger.info(f"Using reference: {reference_name}")
    
    # Get all percentile columns for reference
    reference_dict = {}
    for col in reference_row.index:
        if col.endswith('_percentile'):
            reference_dict[col] = reference_row[col]
    
    logger.info(f"Reference has {len(reference_dict)} percentile properties")
    
    # Prepare arguments
    args_list = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        is_reference = (idx == 0)
        args_list.append((row_dict, reference_dict, save_path, is_reference))
    
    # Process sequentially
    start_time = time.time()
    successful_plots = 0
    failed_plots = 0

    results = [process_single_reaction(args) for args in args_list]

    # Process results
    for success, message, idx in results:
        if success:
            successful_plots += 1
            logger.info(f"{message}")
        else:
            failed_plots += 1
            logger.error(f"{message}")
    
    end_time = time.time()
    logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Successfully created {successful_plots} plots, {failed_plots} failed")

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe for plotting."""
    df = df.copy()
    
    # Add index column if missing
    if 'index' not in df.columns:
        df['index'] = df.index
    
    # Handle shepherd score column
    if ('Volumetric shape and ESP similarity score' in df.columns and 
        'Volumetric_shape_and_ESP_similarity_score_percentile' not in df.columns):
        df['Volumetric_shape_and_ESP_similarity_score_percentile'] = df['Volumetric shape and ESP similarity score'] * 100
    
    # Clean reaction names
    df['reaction_name'] = df['reaction_name'].str.replace(r'[{}]', '', regex=True)
    
    return df

def create_radial_plots_main(
    derivatives: str,
    output: str = "radial plots"
) -> None:
    """
    Generate radial plots - can be imported or run from CLI.
    
    Args:
        derivatives: Path to input CSV file containing reaction data
        output: Path to save output directory for radial plots
    """
    try:
        # Load data
        df = pd.read_csv(derivatives)
        logger.info(f"Successfully loaded derivatives file: {derivatives}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        
        # Check if required columns exist
        required_columns = ['product_SMILES', 'reaction_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Preprocess DataFrame
        if 'Volumetric shape and ESP similarity score' in df.columns and 'Volumetric_shape_and_ESP_similarity_score_percentile' not in df.columns:
            df['Volumetric_shape_and_ESP_similarity_score_percentile'] = df['Volumetric shape and ESP similarity score'] * 100
            logger.info("Added Volumetric_shape_and_ESP_similarity_score_percentile column")
        
        # Clean reaction names
        df['reaction_name'] = df['reaction_name'].str.replace(r'[{}]', '', regex=True)
        
        # Add index column if missing
        if 'index' not in df.columns:
            df['index'] = df.index
            logger.info("Added index column")
        
        # Log available percentile columns
        percentile_cols = [col for col in df.columns if col.endswith('_percentile')]
        logger.info(f"Found {len(percentile_cols)} percentile columns")
        
        # Check if we have the required percentile columns
        required_percentile_cols = [
            'BBB_Martins_drugbank_approved_percentile',
            'Solubility_AqSolDB_drugbank_approved_percentile',
            'Bioavailability_Ma_drugbank_approved_percentile',
            'hERG_drugbank_approved_percentile',
            'Lipophilicity_AstraZeneca_drugbank_approved_percentile'
        ]
        
        missing_percentile_cols = [col for col in required_percentile_cols if col not in df.columns]
        if missing_percentile_cols:
            raise ValueError(f"Missing required percentile columns: {missing_percentile_cols}")
        
        # Generate plots based on choice
        logger.info("Starting plot generation...")
        
        analyze_reactions(df, save_path=output)
        
        logger.info(f"Plots saved to: {output}")
        
    except FileNotFoundError:
        logger.error(f"File not found: {derivatives}")
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise

def main(args) -> int:
    """CLI wrapper for create_radial_plots_main."""
    try:
        create_radial_plots_main(
            derivatives=args.derivatives,
            output=args.output
        )
        return 0
    except Exception:
        return 1

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description="Generate radial plots for derivatives of LOSDT results.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--derivatives", "-d", type=str, required=True, 
                       help="Path to input CSV file containing reaction data")
    parser.add_argument("--output", "-o", type=str, default="radial plots", 
                       help="Path to save output directory for radial plots (default: radial plots)")
    
    args = parser.parse_args()
    exit(main(args))
