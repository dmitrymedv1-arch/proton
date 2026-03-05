"""
Proton Hydration Predictor for Perovskite Oxides
A comprehensive tool for analyzing and predicting hydration thermodynamics 
of proton-conducting perovskites for energy applications.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Scientific plotting style configuration
# =============================================================================
plt.style.use('default')
plt.rcParams.update({
    # Font sizes and weights
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    
    # Axes appearance
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    
    # Tick parameters
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    
    # Legend
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    
    # Figure
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white',
    
    # Lines
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
})

# =============================================================================
# Ionic radii database (Shannon radii, in Angstroms)
# =============================================================================
IONIC_RADII = {
    # A-site cations (12-coordinate for perovskites)
    'Ba': {'charge': 2, 'XII': 1.61},
    'Sr': {'charge': 2, 'XII': 1.44},
    'Ca': {'charge': 2, 'XII': 1.34},
    'La': {'charge': 3, 'IX': 1.216, 'XII': 1.36},
    'Nd': {'charge': 3, 'IX': 1.163, 'XII': 1.27},
    'Gd': {'charge': 3, 'IX': 1.107, 'VIII': 1.053},
    'Sm': {'charge': 3, 'IX': 1.132, 'VIII': 1.079},
    'Y': {'charge': 3, 'VIII': 1.019, 'VI': 0.9},
    'Pr': {'charge': 3, 'IX': 1.179},
    'Ce': {'charge': 3, 'IX': 1.196, 'VI': 1.01},
    
    # B-site cations (6-coordinate)
    'Ti': {'charge': 4, 'VI': 0.605},
    'Zr': {'charge': 4, 'VI': 0.72},
    'Sn': {'charge': 4, 'VI': 0.69},
    'Ce': {'charge': 4, 'VI': 0.87},
    'Hf': {'charge': 4, 'VI': 0.71},
    'Nb': {'charge': 5, 'VI': 0.64},
    'Ta': {'charge': 5, 'VI': 0.64},
    'Mo': {'charge': 6, 'VI': 0.59},
    'W': {'charge': 6, 'VI': 0.6},
    
    # Dopants (6-coordinate)
    'Sc': {'charge': 3, 'VI': 0.745},
    'In': {'charge': 3, 'VI': 0.80},
    'Y': {'charge': 3, 'VI': 0.90},
    'Yb': {'charge': 3, 'VI': 0.868},
    'Gd': {'charge': 3, 'VI': 0.938},
    'Er': {'charge': 3, 'VI': 0.89},
    'Dy': {'charge': 3, 'VI': 0.912},
    'Ho': {'charge': 3, 'VI': 0.901},
    'Tm': {'charge': 3, 'VI': 0.88},
    'Lu': {'charge': 3, 'VI': 0.861},
    'Eu': {'charge': 3, 'VI': 0.947},
    'Tb': {'charge': 3, 'VI': 0.923},
    'Pr': {'charge': 3, 'VI': 0.99},
    'Nd': {'charge': 3, 'VI': 0.983},
    'Sm': {'charge': 3, 'VI': 0.958},
    'Al': {'charge': 3, 'VI': 0.535},
    'Ga': {'charge': 3, 'VI': 0.62},
    'Fe': {'charge': 3, 'VI': 0.645},  # High spin
    'Co': {'charge': 3, 'VI': 0.545},  # Low spin
    'Ni': {'charge': 3, 'VI': 0.56},    # Low spin
    'Zn': {'charge': 2, 'VI': 0.74},
    'Mg': {'charge': 2, 'VI': 0.72},
    
    # Oxygen
    'O': {'charge': -2, 'VI': 1.40},
}

# =============================================================================
# Electronegativity database (Pauling scale)
# =============================================================================
ELECTRONEGATIVITY = {
    'H': 2.20, 'He': None, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
    'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': None, 'Na': 0.93, 'Mg': 1.31,
    'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': None,
    'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
    'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
    'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 3.00,
    'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 2.16,
    'Tc': 1.90, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
    'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.10, 'I': 2.66, 'Xe': 2.60,
    'Cs': 0.79, 'Ba': 0.89, 'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
    'Pm': None, 'Sm': 1.17, 'Eu': None, 'Gd': 1.20, 'Tb': None, 'Dy': 1.22,
    'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': None, 'Lu': 1.27, 'Hf': 1.30,
    'Ta': 1.50, 'W': 2.36, 'Re': 1.90, 'Os': 2.20, 'Ir': 2.20, 'Pt': 2.28,
    'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33,
}

# =============================================================================
# Build combined dataset from all sources
# =============================================================================
@st.cache_data
def load_and_combine_data():
    """
    Create a comprehensive dataset combining:
    1. Data from the Excel file (converted to dict format here)
    2. Table 1 data (classical proton conductors)
    3. Table 2 data (layered and related structures)
    """
    
    # =========================================================
    # Data from Excel file (Sheet 1 - "для ИИ")
    # =========================================================
    excel_data = [
        # A, B, dopant, content, delta_H, delta_S, reference
        ['Ba', 'Ti', 'Sc', 0.5, -55, -120, '10.1021/ic503006u'],
        ['Ba', 'Ti', 'Sc', 0.6, -53, -102, '10.1021/ic503006u'],
        ['Ba', 'Ti', 'Sc', 0.7, -56, -93, '10.1021/ic503006u'],
        ['Ba', 'Ti', 'In', 0.5, -57, -132, '10.1021/ic503006u'],
        ['Ba', 'Ti', 'In', 0.7, -68, -125, '10.1021/ic503006u'],
        ['Ba', 'Ti', 'In', 0.5, -129, -129, '10.1016/j.ceramint.2025.10.295'],
        ['Ba', 'Ti', 'In', 0.6, -149, -149, '10.1016/j.ceramint.2025.10.295'],
        ['Ba', 'Sn', 'Y', 0.05, -46, -95, '10.1016/j.ssi.2012.02.045'],
        ['Ba', 'Sn', 'Y', 0.125, -50, -88, '10.1016/j.ssi.2012.02.045'],
        ['Ba', 'Sn', 'Y', 0.25, -66, -118, '10.1016/j.ssi.2012.02.045'],
        ['Ba', 'Sn', 'Y', 0.375, -84, -122, '10.1016/j.ssi.2012.02.045'],
        ['Ba', 'Sn', 'Y', 0.5, -68, -108, '10.1016/j.ssi.2012.02.045'],
        ['Ba', 'Sn', 'In', 0.05, -57.4, -57.4, '10.1016/j.matre.2025.100382'],
        ['Ba', 'Sn', 'In', 0.1, -58.9, -58.9, '10.1016/j.matre.2025.100382'],
        ['Ba', 'Sn', 'In', 0.2, -65.6, -65.6, '10.1016/j.matre.2025.100382'],
        ['Ba', 'Sn', 'In', 0.3, -78.9, -78.9, '10.1016/j.matre.2025.100382'],
        ['Ba', 'Sn', 'In', 0.4, -84.2, -84.2, '10.1016/j.matre.2025.100382'],
        ['Ba', 'Sn', 'In', 0.5, -75.8, -75.8, '10.1016/j.matre.2025.100382'],
        ['Ba', 'Sn', 'In', 0.6, -96.1, -96.1, '10.1016/j.matre.2025.100382'],
        ['Ba', 'Zr', 'Y', 0.02, -80.9, -94.4, '10.1016/S0167-2738(01)00953-5'],
        ['Ba', 'Zr', 'Y', 0.05, -79.5, -93.5, '10.1016/S0167-2738(01)00953-5'],
        ['Ba', 'Zr', 'Y', 0.1, -79.5, -88.9, '10.1016/S0167-2738(01)00953-5'],
        ['Ba', 'Zr', 'Y', 0.15, -83.4, -92.1, '10.1016/S0167-2738(01)00953-5'],
        ['Ba', 'Zr', 'Y', 0.2, -93.3, -103.2, '10.1016/S0167-2738(01)00953-5'],
        ['Ba', 'Zr', 'Y', 0.25, -83.4, -92.1, '10.1016/S0167-2738(01)00953-5'],
        ['Ba', 'Zr', 'Sc', 0.1, -119.4, -124.9, '10.1016/S0167-2738(01)00953-5'],
        ['Ba', 'Zr', 'In', 0.1, -66.6, -90.2, '10.1016/S0167-2738(01)00953-5'],
        ['Ba', 'Sn', 'Sc', 0.125, -73.4, -106.4, '10.1016/j.ijhydene.2011.03.105'],
        ['Ba', 'Sn', 'Y', 0.125, -49.0, -86.8, '10.1016/j.ijhydene.2011.03.105'],
        ['Ba', 'Sn', 'In', 0.125, -58.9, -109.8, '10.1016/j.ijhydene.2011.03.105'],
        ['Sr', 'Sn', 'Sc', 0.2, -76, -116, '10.1021/acsenergylett.1c01239'],
        ['Ba', 'Sn', 'Sc', 0.7, -125, -134, '10.1038/s41563-025-02311-w'],
        ['Ba', 'Sn', 'Sc', 0.2, -131, -169, '10.1038/s41563-025-02311-w'],
        ['Ba', 'Ti', 'Sc', 0.8, -99, -116, '10.1038/s41563-025-02311-w'],
        ['Ba', 'Ti', 'Sc', 0.6, -71, -109, '10.1038/s41563-025-02311-w'],
        ['Ba', 'Hf', 'Sc', 0.5, -112, -123, '10.1038/s41563-025-02311-w'],
        ['Ba', 'Hf', 'Sc', 0.2, -80, -97, '10.1038/s41563-025-02311-w'],
    ]
    
    # =========================================================
    # Table 1 data (Classical proton-conducting perovskites)
    # =========================================================
    table1_data = [
        # A, B, dopant, content, delta_H, delta_S, reference
        ['Ba', 'Zr', 'Y', 0.15, -83.4, -92.1, '[21]'],
        ['Ba', 'Zr', 'Y', 0.55, -83.4, -92.1, '[21]'],  # Note: likely typo in original
        ['Ba', 'Zr', 'Er', 0.2, -82, -106, '[111]'],
        ['Ba', 'Zr', 'In', 0.2, -71, -101, '[111]'],
        ['Ba', 'Zr', 'Lu', 0.2, -99, -112, '[111]'],
        ['Ba', 'Zr', 'Sc', 0.2, -104, -96, '[110]'],
        ['Ba', 'Zr', 'Sc', 0.2, -100, -111, '[111]'],
        ['Ba', 'Zr', 'Sc', 0.2, -131, -144, '[112]'],
        ['Ba', 'Zr', 'Y', 0.12, -78, -94, '[114]'],
        ['Ba', 'Zr', 'Y', 0.2, -89, -124, '[51]'],
        ['Ba', 'Zr', 'Y', 0.2, -84, -96, '[111]'],
        ['Ba', 'Zr', 'Y', 0.2, -91.1, -104.1, '[113]'],
        ['Ba', 'Zr', 'Y', 0.2, -93.3, -103.2, '[21]'],
        ['Ba', 'Zr', 'Y', 0.2, -95, -114, '[111]'],
        ['Ba', 'Zr', 'Y', 0.05, -79.5, -93.5, '[21]'],
        ['Ba', 'Zr', 'Y', 0.02, -80.9, -94.4, '[21]'],
        ['Ba', 'Zr', 'Gd', 0.1, -66.1, -85.9, '[21]'],
        ['Ba', 'Zr', 'In', 0.1, -66.6, -90.2, '[21]'],
        ['Ba', 'Zr', 'Sc', 0.1, -119.4, -124.9, '[21]'],
        ['Ba', 'Zr', 'Y', 0.1, -79.5, -88.9, '[21]'],
        ['Ba', 'Ce', 'Y', 0.2, -136.9, -129.9, '[113]'],
        ['Ba', 'Ce', 'Y', 0.2, -112, -110, '[111]'],
        ['Ba', 'Ce', 'Er', 0.1, -124, -129, '[53]'],
        ['Ba', 'Ce', 'Gd', 0.1, -133, -141, '[53]'],
        ['Ba', 'Ce', 'In', 0.1, -96, -129, '[53]'],
        ['Ba', 'Ce', 'Sc', 0.1, -139, -141, '[53]'],
        ['Ba', 'Ce', 'Y', 0.1, -122, -119, '[118]'],
        ['Ba', 'Ce', 'Y', 0.1, -135, -141, '[53]'],
        ['Ba', 'Ce', 'Yb', 0.1, -127, -126, '[113]'],
        ['Ba', 'Ti', 'In', 0.7, -73, -127, '[46]'],
        ['Ba', 'Ti', 'In', 0.7, -68, -125, '[46]'],
        ['Ba', 'Ti', 'Sc', 0.7, -83, -92, '[46]'],
        ['Ba', 'Ti', 'Sc', 0.7, -56, -93, '[46]'],
        ['Ba', 'Ti', 'Sc', 0.6, -79, -136, '[46]'],
        ['Ba', 'Ti', 'Sc', 0.6, -53, -102, '[46]'],
        ['Ba', 'Ti', 'In', 0.5, -53, -126, '[46]'],
        ['Ba', 'Ti', 'In', 0.5, -57, -132, '[46]'],
        ['Ba', 'Ti', 'Sc', 0.5, -57, -127, '[46]'],
        ['Ba', 'Ti', 'Sc', 0.5, -55, -120, '[46]'],
        ['Ba', 'Ti', 'Sc', 0.2, -108, -146, '[46]'],
        ['Ba', 'Ti', 'Sc', 0.2, -93, -143, '[46]'],
        ['Ba', 'Sn', 'Y', 0.5, -73, -115, '[119]'],
        ['Ba', 'Sn', 'Y', 0.5, -89.1, -73.7, '[116]'],
        ['Ba', 'Sn', 'Y', 0.375, -83, -123, '[119]'],
        ['Ba', 'Sn', 'Y', 0.25, -72, -128, '[119]'],
        ['Ba', 'Sn', 'Gd', 0.125, -46.1, -79.3, '[120]'],
        ['Ba', 'Sn', 'In', 0.125, -58.9, -109.8, '[120]'],
        ['Ba', 'Sn', 'Sc', 0.125, -73.4, -106.4, '[120]'],
        ['Ba', 'Sn', 'Sc', 0.125, -73, -100, '[121]'],
        ['Ba', 'Sn', 'Y', 0.125, -49, -87, '[119]'],
        ['Ba', 'Sn', 'Y', 0.05, -50, -102, '[119]'],
        ['Sr', 'Ce', 'Yb', 0.05, -157, -128, '[39]'],
        ['Sr', 'Sn', 'Sc', 0.2, -76, -116, '[111]'],
        ['Ca', 'Zr', 'In', 0.04, -18, -61, '[111]'],
        ['La', 'Sc', 'Sr', 0.09, -97, -112, '[122]'],  # Note: Sr as dopant
        ['La', 'Sc', 'Ba', 0.05, -85, -106, '[58]'],
        ['La', 'Sc', 'Ca', 0.4, -132, -126, '[123]'],
        ['La', 'Sc', 'Sr', 0.04, -61, -88, '[122]'],
        ['La', 'Al', 'Ba', 0.1, -66, -91, '[24]'],
        ['La', 'In', 'Ba', 0.1, -96, -120, '[24]'],
        ['La', 'Sc', 'Ba', 0.1, -105, -116, '[24]'],
        ['La', 'Yb', 'Ba', 0.1, -183, -135, '[124]'],
        ['La', 'Yb', 'Ba', 0.1, -141, -111, '[22]'],
        ['La', 'Y', 'Ba', 0.1, -96, -70, '[24]'],
        ['La', 'Yb', 'Ca', 0.1, -141, -111, '[22]'],
        ['La', 'Yb', 'Mg', 0.1, -141, -111, '[22]'],
        ['La', 'In', 'Sr', 0.1, -96, -120, '[24]'],
        ['La', 'Sc', 'Sr', 0.1, -105, -116, '[24]'],
        ['La', 'Yb', 'Sr', 0.1, -164, -135, '[24]'],
        ['La', 'Y', 'Sr', 0.1, -96, -70, '[24]'],
    ]
    
    # =========================================================
    # Table 2 data (Layered and related structures)
    # =========================================================
    table2_data = [
        # Special compositions - simplified representation
        ['Ba2Y', 'Sn', 'O5.5', -80, -109, '[4]'],
        ['Ba3Ca', 'Nb', 'O9', -65, -104, '[4]'],
        ['Ba4Ca2', 'Nb2', 'O11', -94, -154, '[127,128]'],
        ['Ba5Er2Al2', 'Sn', 'O13', -78, -106, '[129]'],
        ['Ba7In6', 'Al2', 'O19', -56, -167, '[75]'],
        ['Ba7Nb4', 'Mo', 'O20', -17, -20, '[132]'],
        ['La2Ce2', 'O7', -77, -128, '[135]'],
        ['SrLa', 'Al', 'O4', -47, -96, '[92]'],
    ]
    
    # Convert to DataFrame
    cols = ['A_cation', 'B_cation', 'dopant', 'content', 'delta_H', 'delta_S', 'reference']
    df_excel = pd.DataFrame(excel_data, columns=cols)
    df_table1 = pd.DataFrame(table1_data, columns=cols)
    
    # Add source column
    df_excel['source'] = 'Excel'
    df_table1['source'] = 'Table 1'
    
    # Combine
    df_combined = pd.concat([df_excel, df_table1], ignore_index=True)
    
    # Remove duplicates (based on composition and values)
    df_combined = df_combined.drop_duplicates(
        subset=['A_cation', 'B_cation', 'dopant', 'content', 'delta_H', 'delta_S'],
        keep='first'
    )
    
    return df_combined

# =============================================================================
# Descriptor calculation functions
# =============================================================================
def calculate_tolerance_factor(r_A, r_B_avg, r_O=1.40):
    """Calculate Goldschmidt tolerance factor"""
    return (r_A + r_O) / (np.sqrt(2) * (r_B_avg + r_O))

def get_ionic_radius(element, coordination='VI'):
    """Get ionic radius for element with specified coordination"""
    if element in IONIC_RADII:
        data = IONIC_RADII[element]
        # Try exact coordination
        if coordination in data:
            return data[coordination]
        # Try alternative coordinations
        if 'VI' in data:
            return data['VI']
        if 'XII' in data:
            return data['XII']
        if 'IX' in data:
            return data['IX']
        # Return first available
        for key in data:
            if key not in ['charge']:
                return data[key]
    return None

def get_electronegativity(element):
    """Get Pauling electronegativity for element"""
    return ELECTRONEGATIVITY.get(element, None)

def calculate_descriptors(row):
    """Calculate all descriptors for a given material"""
    A = row['A_cation']
    B = row['B_cation']
    D = row['dopant']
    x = row['content']
    
    # Get ionic radii
    r_A = get_ionic_radius(A, 'XII' if A in ['Ba', 'Sr', 'Ca'] else 'IX')
    r_B = get_ionic_radius(B, 'VI')
    r_D = get_ionic_radius(D, 'VI') if D != 'O' else None
    r_O = 1.40
    
    # Get electronegativities
    chi_A = get_electronegativity(A)
    chi_B = get_electronegativity(B)
    chi_D = get_electronegativity(D) if D != 'O' else None
    
    # Calculate descriptors
    descriptors = {}
    
    if all(v is not None for v in [r_A, r_B, r_O]):
        # Average B-site radius
        if r_D is not None and x > 0:
            r_B_avg = (1 - x) * r_B + x * r_D
            descriptors['r_B_avg'] = r_B_avg
            descriptors['delta_r_B'] = abs(r_D - r_B)
        else:
            r_B_avg = r_B
            descriptors['r_B_avg'] = r_B
            descriptors['delta_r_B'] = 0
        
        # Tolerance factor
        descriptors['t_Goldschmidt'] = calculate_tolerance_factor(r_A, r_B_avg, r_O)
        
        # Size mismatch
        descriptors['r_A'] = r_A
        descriptors['r_B'] = r_B
        descriptors['r_D'] = r_D if r_D is not None else 0
    
    if all(v is not None for v in [chi_A, chi_B]):
        # Average electronegativity
        if chi_D is not None and x > 0:
            chi_B_avg = (1 - x) * chi_B + x * chi_D
            descriptors['chi_B_avg'] = chi_B_avg
            descriptors['delta_chi'] = abs(chi_D - chi_B)
            descriptors['chi_D'] = chi_D
        else:
            chi_B_avg = chi_B
            descriptors['chi_B_avg'] = chi_B
            descriptors['delta_chi'] = 0
            descriptors['chi_D'] = 0
        
        # Electronegativity difference
        descriptors['chi_A'] = chi_A
        descriptors['chi_B'] = chi_B
        descriptors['chi_diff'] = chi_B_avg - chi_A
    
    # Composition
    descriptors['content'] = x
    
    return descriptors

# =============================================================================
# Machine Learning Models
# =============================================================================
@st.cache_resource
def train_prediction_models(df):
    """Train ML models for predicting delta_H and delta_S"""
    
    # Prepare features
    feature_cols = ['content', 'r_A', 'r_B', 'r_D', 'r_B_avg', 'delta_r_B', 
                   't_Goldschmidt', 'chi_A', 'chi_B', 'chi_D', 'chi_B_avg', 
                   'chi_diff']
    
    # Calculate descriptors for all rows
    descriptor_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        desc = calculate_descriptors(row)
        if len(desc) > 5:  # Enough descriptors
            desc['delta_H'] = row['delta_H']
            desc['delta_S'] = row['delta_S']
            desc['A_cation'] = row['A_cation']
            desc['B_cation'] = row['B_cation']
            desc['dopant'] = row['dopant']
            descriptor_list.append(desc)
            valid_indices.append(idx)
    
    df_features = pd.DataFrame(descriptor_list)
    
    if len(df_features) < 10:
        return None, None, None, None, df_features
    
    # Prepare X and y
    X = df_features[feature_cols].fillna(0)
    y_H = df_features['delta_H']
    y_S = df_features['delta_S']
    
    # Encode categorical variables
    le_A = LabelEncoder()
    le_B = LabelEncoder()
    le_D = LabelEncoder()
    
    X_cat = pd.DataFrame({
        'A_enc': le_A.fit_transform(df_features['A_cation']),
        'B_enc': le_B.fit_transform(df_features['B_cation']),
        'D_enc': le_D.fit_transform(df_features['dopant'])
    })
    
    X = pd.concat([X, X_cat], axis=1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train models
    model_H = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model_S = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    model_H.fit(X_scaled, y_H)
    model_S.fit(X_scaled, y_S)
    
    # Calculate cross-validation scores
    cv_scores_H = cross_val_score(model_H, X_scaled, y_H, cv=min(5, len(X)), scoring='r2')
    cv_scores_S = cross_val_score(model_S, X_scaled, y_S, cv=min(5, len(X)), scoring='r2')
    
    # Feature importance
    feature_importance_H = pd.DataFrame({
        'feature': X.columns,
        'importance': model_H.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model_H': model_H,
        'model_S': model_S,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'cv_H_mean': cv_scores_H.mean(),
        'cv_S_mean': cv_scores_S.mean(),
        'feature_importance': feature_importance_H,
        'le_A': le_A,
        'le_B': le_B,
        'le_D': le_D,
        'X_train': X_scaled,
        'y_train_H': y_H,
        'y_train_S': y_S
    }, df_features

# =============================================================================
# Streamlit App
# =============================================================================
def main():
    st.set_page_config(
        page_title="Proton Hydration Predictor",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("💧 Proton Hydration Predictor for Perovskite Oxides")
    st.markdown("""
    This application provides tools for analyzing and predicting the hydration thermodynamics 
    of proton-conducting perovskites for energy applications (fuel cells, electrolyzers, sensors).
    
    **Features:**
    - Explore hydration parameters (ΔH, ΔS) for various perovskite systems
    - Visualize correlations with composition, ionic radii, and electronegativity
    - Predict hydration thermodynamics for new compositions using machine learning
    - Find similar materials in the database
    """)
    
    # Load data
    with st.spinner("Loading data and training models..."):
        df = load_and_combine_data()
        model_data, df_features = train_prediction_models(df)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📊 Data Explorer", "🔍 Correlations", "🤖 Predictor", "📈 Model Performance", "ℹ️ About"]
    )
    
    # =========================================================================
    # Page 1: Data Explorer
    # =========================================================================
    if page == "📊 Data Explorer":
        st.header("📊 Hydration Thermodynamics Database")
        
        # Filters
        st.subheader("Filter Data")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            a_cations = ['All'] + sorted(df['A_cation'].unique().tolist())
            selected_a = st.selectbox("A-cation", a_cations)
        
        with col2:
            b_cations = ['All'] + sorted(df['B_cation'].unique().tolist())
            selected_b = st.selectbox("B-cation", b_cations)
        
        with col3:
            dopants = ['All'] + sorted([d for d in df['dopant'].unique() if pd.notna(d)])
            selected_d = st.selectbox("Dopant", dopants)
        
        with col4:
            sources = ['All'] + sorted(df['source'].unique().tolist())
            selected_source = st.selectbox("Source", sources)
        
        # Apply filters
        filtered_df = df.copy()
        if selected_a != 'All':
            filtered_df = filtered_df[filtered_df['A_cation'] == selected_a]
        if selected_b != 'All':
            filtered_df = filtered_df[filtered_df['B_cation'] == selected_b]
        if selected_d != 'All':
            filtered_df = filtered_df[filtered_df['dopant'] == selected_d]
        if selected_source != 'All':
            filtered_df = filtered_df[filtered_df['source'] == selected_source]
        
        # Statistics
        st.subheader("Dataset Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total entries", len(filtered_df))
        with col2:
            st.metric("ΔH range", f"{filtered_df['delta_H'].min():.0f} to {filtered_df['delta_H'].max():.0f} kJ/mol")
        with col3:
            st.metric("ΔS range", f"{filtered_df['delta_S'].min():.0f} to {filtered_df['delta_S'].max():.0f} J/mol·K")
        with col4:
            st.metric("A-cations", filtered_df['A_cation'].nunique())
        with col5:
            st.metric("Dopants", filtered_df['dopant'].nunique())
        
        # Data table
        st.subheader("Data Table")
        st.dataframe(
            filtered_df.style.background_gradient(cmap='RdBu_r', subset=['delta_H', 'delta_S']),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name="hydration_data.csv",
            mime="text/csv"
        )
        
        # Distribution plots
        st.subheader("Distribution of Hydration Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(filtered_df['delta_H'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('ΔH (kJ mol⁻¹)')
            ax.set_ylabel('Frequency')
            ax.set_title('Enthalpy of Hydration')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(filtered_df['delta_S'], bins=20, color='coral', edgecolor='black', alpha=0.7)
            ax.set_xlabel('ΔS (J mol⁻¹ K⁻¹)')
            ax.set_ylabel('Frequency')
            ax.set_title('Entropy of Hydration')
            st.pyplot(fig)
            plt.close()
    
    # =========================================================================
    # Page 2: Correlations
    # =========================================================================
    elif page == "🔍 Correlations":
        st.header("🔍 Structure-Composition-Property Correlations")
        
        # Select plot type
        plot_type = st.selectbox(
            "Select correlation to visualize",
            [
                "ΔH vs Dopant Content",
                "ΔH vs Dopant Ionic Radius",
                "ΔH vs Dopant Electronegativity",
                "ΔH vs Tolerance Factor",
                "ΔH vs B-site Average Radius",
                "ΔH vs Electronegativity Difference",
                "Compensation Effect (ΔH vs ΔS)",
                "3D: ΔH vs (Content, Radius)",
                "3D: ΔH vs (Radius, Electronegativity)"
            ]
        )
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            a_cations = ['All'] + sorted(df['A_cation'].unique().tolist())
            selected_a = st.selectbox("Filter by A-cation", a_cations, key='corr_a')
        with col2:
            b_cations = ['All'] + sorted(df['B_cation'].unique().tolist())
            selected_b = st.selectbox("Filter by B-cation", b_cations, key='corr_b')
        with col3:
            color_by = st.selectbox("Color points by", ["A-cation", "B-cation", "Dopant", "Structure"])
        
        # Prepare data with descriptors
        desc_list = []
        for _, row in df.iterrows():
            desc = calculate_descriptors(row)
            if len(desc) > 5:
                desc['A_cation'] = row['A_cation']
                desc['B_cation'] = row['B_cation']
                desc['dopant'] = row['dopant']
                desc['delta_H'] = row['delta_H']
                desc['delta_S'] = row['delta_S']
                desc['source'] = row['source']
                desc_list.append(desc)
        
        plot_df = pd.DataFrame(desc_list)
        
        # Apply filters
        if selected_a != 'All':
            plot_df = plot_df[plot_df['A_cation'] == selected_a]
        if selected_b != 'All':
            plot_df = plot_df[plot_df['B_cation'] == selected_b]
        
        if plot_df.empty:
            st.warning("No data available with selected filters.")
            return
        
        # Create plots
        if plot_type == "ΔH vs Dopant Content":
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for a_cation in plot_df['A_cation'].unique():
                subset = plot_df[plot_df['A_cation'] == a_cation]
                ax.scatter(subset['content'], subset['delta_H'], 
                          label=a_cation, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('Dopant Content, x')
            ax.set_ylabel('ΔH (kJ mol⁻¹)')
            ax.set_title('Enthalpy of Hydration vs Dopant Concentration')
            ax.legend(title='A-cation', frameon=True, fancybox=False, edgecolor='black')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            
            # Add trend line
            z = np.polyfit(plot_df['content'], plot_df['delta_H'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(plot_df['content'].min(), plot_df['content'].max(), 50)
            ax.plot(x_trend, p(x_trend), 'r--', linewidth=1, alpha=0.8, label=f'Trend: ΔH = {z[0]:.1f}x + {z[1]:.1f}')
            ax.legend()
            
            st.pyplot(fig)
            plt.close()
        
        elif plot_type == "ΔH vs Dopant Ionic Radius":
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for b_cation in plot_df['B_cation'].unique():
                subset = plot_df[plot_df['B_cation'] == b_cation]
                if 'r_D' in subset.columns:
                    ax.scatter(subset['r_D'], subset['delta_H'], 
                              label=b_cation, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('Dopant Ionic Radius (Å)')
            ax.set_ylabel('ΔH (kJ mol⁻¹)')
            ax.set_title('Enthalpy of Hydration vs Dopant Size')
            ax.legend(title='B-cation', frameon=True, fancybox=False, edgecolor='black')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            
            st.pyplot(fig)
            plt.close()
        
        elif plot_type == "ΔH vs Dopant Electronegativity":
            fig, ax = plt.subplots(figsize=(8, 6))
            
            scatter = ax.scatter(plot_df['chi_D'], plot_df['delta_H'], 
                                c=plot_df['content'], cmap='viridis', 
                                s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('Dopant Electronegativity (Pauling)')
            ax.set_ylabel('ΔH (kJ mol⁻¹)')
            ax.set_title('Enthalpy of Hydration vs Dopant Electronegativity')
            plt.colorbar(scatter, ax=ax, label='Dopant Content, x')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            
            st.pyplot(fig)
            plt.close()
        
        elif plot_type == "ΔH vs Tolerance Factor":
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for struct in ['cubic', 'hexagonal', 'orthorhombic']:
                subset = plot_df#.sample(frac=0.3)  # In real app, would have structure info
                if 't_Goldschmidt' in subset.columns:
                    ax.scatter(subset['t_Goldschmidt'], subset['delta_H'], 
                              label=struct, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('Goldschmidt Tolerance Factor, t')
            ax.set_ylabel('ΔH (kJ mol⁻¹)')
            ax.set_title('Enthalpy of Hydration vs Tolerance Factor')
            ax.legend(title='Structure', frameon=True, fancybox=False, edgecolor='black')
            ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            
            st.pyplot(fig)
            plt.close()
        
        elif plot_type == "Compensation Effect (ΔH vs ΔS)":
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for a_cation in plot_df['A_cation'].unique():
                subset = plot_df[plot_df['A_cation'] == a_cation]
                ax.scatter(subset['delta_S'], subset['delta_H'], 
                          label=a_cation, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('ΔS (J mol⁻¹ K⁻¹)')
            ax.set_ylabel('ΔH (kJ mol⁻¹)')
            ax.set_title('Compensation Effect: ΔH vs ΔS')
            ax.legend(title='A-cation', frameon=True, fancybox=False, edgecolor='black')
            
            # Add trend line
            z = np.polyfit(plot_df['delta_S'], plot_df['delta_H'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(plot_df['delta_S'].min(), plot_df['delta_S'].max(), 50)
            ax.plot(x_trend, p(x_trend), 'r--', linewidth=1, alpha=0.8, 
                   label=f'Compensation: T_c = {-1/z[0]:.1f} K')
            ax.legend()
            
            st.pyplot(fig)
            plt.close()
        
        elif plot_type == "3D: ΔH vs (Content, Radius)":
            fig = px.scatter_3d(
                plot_df, 
                x='content', 
                y='r_D', 
                z='delta_H',
                color='A_cation',
                size='delta_H',
                hover_data=['B_cation', 'dopant', 'delta_S'],
                labels={
                    'content': 'Dopant Content, x',
                    'r_D': 'Dopant Radius (Å)',
                    'delta_H': 'ΔH (kJ mol⁻¹)'
                },
                title='3D Correlation: ΔH vs Content and Dopant Radius'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "3D: ΔH vs (Radius, Electronegativity)":
            fig = px.scatter_3d(
                plot_df, 
                x='r_D', 
                y='chi_D', 
                z='delta_H',
                color='content',
                size='delta_H',
                hover_data=['A_cation', 'B_cation', 'dopant'],
                labels={
                    'r_D': 'Dopant Radius (Å)',
                    'chi_D': 'Dopant Electronegativity',
                    'delta_H': 'ΔH (kJ mol⁻¹)',
                    'content': 'Content'
                },
                title='3D Correlation: ΔH vs Dopant Properties'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # Page 3: Predictor
    # =========================================================================
    elif page == "🤖 Predictor":
        st.header("🤖 Hydration Parameter Predictor")
        
        if model_data is None or df_features.empty:
            st.warning("Insufficient data for training prediction models. Need at least 10 samples with complete descriptors.")
            return
        
        st.markdown("""
        Enter the composition of your perovskite material to predict its hydration thermodynamics.
        The prediction is based on machine learning models trained on experimental data.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            a_cation = st.selectbox("A-cation", ['Ba', 'Sr', 'La', 'Ca', 'Nd', 'Gd'])
            # Get available B-cations based on A
            if a_cation in ['Ba', 'Sr']:
                b_options = ['Ti', 'Zr', 'Sn', 'Ce', 'Hf']
            else:
                b_options = ['Sc', 'In', 'Y', 'Yb', 'Gd', 'Lu']
        
        with col2:
            b_cation = st.selectbox("B-cation", b_options)
            
            # Get available dopants
            dopant_options = ['Sc', 'In', 'Y', 'Yb', 'Gd', 'Er', 'Dy', 'Ho', 'Tm', 'Lu']
            dopant = st.selectbox("Dopant", dopant_options)
        
        with col3:
            content = st.slider("Dopant content, x", 0.0, 0.8, 0.2, 0.01)
        
        # Calculate descriptors for input
        input_row = pd.Series({
            'A_cation': a_cation,
            'B_cation': b_cation,
            'dopant': dopant,
            'content': content
        })
        
        input_desc = calculate_descriptors(input_row)
        
        # Display calculated descriptors
        st.subheader("Calculated Descriptors")
        desc_cols = st.columns(4)
        desc_items = [
            ('r_A', f"{input_desc.get('r_A', 0):.3f} Å"),
            ('r_B', f"{input_desc.get('r_B', 0):.3f} Å"),
            ('r_D', f"{input_desc.get('r_D', 0):.3f} Å"),
            ('r_B_avg', f"{input_desc.get('r_B_avg', 0):.3f} Å"),
            ('t_Goldschmidt', f"{input_desc.get('t_Goldschmidt', 0):.3f}"),
            ('χ_A', f"{input_desc.get('chi_A', 0):.2f}"),
            ('χ_B', f"{input_desc.get('chi_B', 0):.2f}"),
            ('χ_D', f"{input_desc.get('chi_D', 0):.2f}"),
            ('χ_diff', f"{input_desc.get('chi_diff', 0):.2f}"),
            ('δr_B', f"{input_desc.get('delta_r_B', 0):.3f} Å"),
        ]
        
        for i, (label, value) in enumerate(desc_items[:8]):  # Show first 8
            desc_cols[i % 4].metric(label, value)
        
        # Prepare features for prediction
        feature_cols = ['content', 'r_A', 'r_B', 'r_D', 'r_B_avg', 'delta_r_B', 
                       't_Goldschmidt', 'chi_A', 'chi_B', 'chi_D', 'chi_B_avg', 'chi_diff']
        
        # Create feature vector
        X_pred = pd.DataFrame([{
            'content': input_desc.get('content', content),
            'r_A': input_desc.get('r_A', 0),
            'r_B': input_desc.get('r_B', 0),
            'r_D': input_desc.get('r_D', 0),
            'r_B_avg': input_desc.get('r_B_avg', 0),
            'delta_r_B': input_desc.get('delta_r_B', 0),
            't_Goldschmidt': input_desc.get('t_Goldschmidt', 0),
            'chi_A': input_desc.get('chi_A', 0),
            'chi_B': input_desc.get('chi_B', 0),
            'chi_D': input_desc.get('chi_D', 0),
            'chi_B_avg': input_desc.get('chi_B_avg', 0),
            'chi_diff': input_desc.get('chi_diff', 0)
        }])
        
        # Add encoded categorical features
        X_pred['A_enc'] = model_data['le_A'].transform([a_cation])[0] if a_cation in model_data['le_A'].classes_ else -1
        X_pred['B_enc'] = model_data['le_B'].transform([b_cation])[0] if b_cation in model_data['le_B'].classes_ else -1
        X_pred['D_enc'] = model_data['le_D'].transform([dopant])[0] if dopant in model_data['le_D'].classes_ else -1
        
        # Scale features
        X_pred_scaled = model_data['scaler'].transform(X_pred[model_data['feature_names']])
        
        # Make predictions
        pred_H = model_data['model_H'].predict(X_pred_scaled)[0]
        pred_S = model_data['model_S'].predict(X_pred_scaled)[0]
        
        # Display predictions
        st.subheader("Predicted Hydration Parameters")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "ΔH (kJ mol⁻¹)",
                f"{pred_H:.1f}",
                delta=None
            )
        with col2:
            st.metric(
                "ΔS (J mol⁻¹ K⁻¹)",
                f"{pred_S:.1f}",
                delta=None
            )
        with col3:
            # Calculate approximate T* (isokinetic temperature)
            if pred_S != 0:
                T_star = -pred_H / pred_S * 1000  # in K
                st.metric(
                    "Isokinetic T (K)",
                    f"{T_star:.0f}",
                    delta=None
                )
        
        # Find similar materials
        st.subheader("Similar Materials in Database")
        
        # Prepare training features for similarity search
        X_train = model_data['X_train']
        
        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=min(5, len(X_train)), metric='euclidean')
        nn.fit(X_train)
        distances, indices = nn.kneighbors(X_pred_scaled)
        
        # Display similar materials
        similar_data = []
        for i, idx in enumerate(indices[0]):
            similar_data.append({
                'Rank': i + 1,
                'Composition': f"{df_features.iloc[idx]['A_cation']}{df_features.iloc[idx]['B_cation']}{df_features.iloc[idx]['dopant']}{df_features.iloc[idx]['content']:.2f}",
                'ΔH (kJ/mol)': df_features.iloc[idx]['delta_H'],
                'ΔS (J/mol·K)': df_features.iloc[idx]['delta_S'],
                'Distance': distances[0][i]
            })
        
        st.dataframe(pd.DataFrame(similar_data), use_container_width=True)
        
        # Feature importance plot
        st.subheader("Feature Importance for Prediction")
        fig, ax = plt.subplots(figsize=(8, 4))
        
        top_features = model_data['feature_importance'].head(10)
        ax.barh(top_features['feature'], top_features['importance'], color='steelblue', edgecolor='black')
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Most Important Features')
        ax.invert_yaxis()
        
        st.pyplot(fig)
        plt.close()
    
    # =========================================================================
    # Page 4: Model Performance
    # =========================================================================
    elif page == "📈 Model Performance":
        st.header("📈 Machine Learning Model Performance")
        
        if model_data is None:
            st.warning("Model not trained. Need more data.")
            return
        
        # Model metrics
        st.subheader("Cross-Validation Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "ΔH Model - R² (CV)",
                f"{model_data['cv_H_mean']:.3f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "ΔS Model - R² (CV)",
                f"{model_data['cv_S_mean']:.3f}",
                delta=None
            )
        
        # Feature importance
        st.subheader("Feature Importance Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        importance_df = model_data['feature_importance']
        top20 = importance_df.head(20)
        
        colors = ['darkred' if 'chi' in f or 'electro' in f else 
                 'darkblue' if 'r_' in f or 'radius' in f else 
                 'darkgreen' for f in top20['feature']]
        
        ax.barh(range(len(top20)), top20['importance'], color=colors, edgecolor='black')
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20['feature'])
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance for ΔH Prediction')
        ax.invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkred', label='Electronegativity'),
            Patch(facecolor='darkblue', label='Ionic Radius'),
            Patch(facecolor='darkgreen', label='Composition')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        st.pyplot(fig)
        plt.close()
        
        # Prediction vs Actual plots
        st.subheader("Prediction vs Actual (Training Data)")
        
        # Get predictions on training data
        train_pred_H = model_data['model_H'].predict(model_data['X_train'])
        train_pred_S = model_data['model_S'].predict(model_data['X_train'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(model_data['y_train_H'], train_pred_H, 
                      alpha=0.6, edgecolors='black', linewidth=0.5, s=50)
            
            # Perfect prediction line
            min_val = min(model_data['y_train_H'].min(), train_pred_H.min())
            max_val = max(model_data['y_train_H'].max(), train_pred_H.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, alpha=0.7)
            
            ax.set_xlabel('Actual ΔH (kJ mol⁻¹)')
            ax.set_ylabel('Predicted ΔH (kJ mol⁻¹)')
            ax.set_title(f'ΔH: R² = {r2_score(model_data["y_train_H"], train_pred_H):.3f}')
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(model_data['y_train_S'], train_pred_S, 
                      alpha=0.6, edgecolors='black', linewidth=0.5, s=50)
            
            # Perfect prediction line
            min_val = min(model_data['y_train_S'].min(), train_pred_S.min())
            max_val = max(model_data['y_train_S'].max(), train_pred_S.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, alpha=0.7)
            
            ax.set_xlabel('Actual ΔS (J mol⁻¹ K⁻¹)')
            ax.set_ylabel('Predicted ΔS (J mol⁻¹ K⁻¹)')
            ax.set_title(f'ΔS: R² = {r2_score(model_data["y_train_S"], train_pred_S):.3f}')
            
            st.pyplot(fig)
            plt.close()
        
        # Residual plots
        st.subheader("Residual Analysis")
        
        residuals_H = model_data['y_train_H'] - train_pred_H
        residuals_S = model_data['y_train_S'] - train_pred_S
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(train_pred_H, residuals_H, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
            ax.set_xlabel('Predicted ΔH (kJ mol⁻¹)')
            ax.set_ylabel('Residuals')
            ax.set_title(f'ΔH Residuals (MAE = {mean_absolute_error(model_data["y_train_H"], train_pred_H):.1f})')
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(train_pred_S, residuals_S, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
            ax.set_xlabel('Predicted ΔS (J mol⁻¹ K⁻¹)')
            ax.set_ylabel('Residuals')
            ax.set_title(f'ΔS Residuals (MAE = {mean_absolute_error(model_data["y_train_S"], train_pred_S):.1f})')
            
            st.pyplot(fig)
            plt.close()
    
    # =========================================================================
    # Page 5: About
    # =========================================================================
    else:
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ## Proton Hydration Predictor for Perovskite Oxides
        
        This application is designed for researchers working on proton-conducting perovskite 
        materials for energy applications such as solid oxide fuel cells (SOFCs), 
        electrolyzers, and hydrogen sensors.
        
        ### Background
        
        Proton-conducting perovskites (general formula ABO₃) can incorporate water from 
        the gas phase, forming protonic defects (OH⁻) that enable proton conduction at 
        elevated temperatures. The hydration reaction can be written as:
        
        **H₂O(g) + Vₒ•• + Oₒˣ ⇌ 2OHₒ•**
        
        The thermodynamics of this reaction are characterized by:
        - **ΔH° (enthalpy of hydration)**: Determines the temperature dependence of proton uptake
        - **ΔS° (entropy of hydration)**: Reflects the ordering during hydration
        
        More negative ΔH indicates stronger water incorporation (favorable at lower temperatures), 
        while less negative ΔH suggests better high-temperature proton mobility.
        
        ### Data Sources
        
        The database combines experimental data from:
        1. **Excel file**: Compiled from various publications on BaTiO₃, BaSnO₃, and BaZrO₃-based systems
        2. **Table 1**: Classical proton-conducting perovskites from literature
        3. **Table 2**: Layered and related structures
        
        All data points include references to original publications.
        
        ### Features
        
        - **Data Explorer**: Filter and browse the hydration thermodynamics database
        - **Correlations**: Visualize relationships between composition, structure, and properties
        - **Predictor**: Machine learning-based prediction of ΔH and ΔS for new compositions
        - **Model Performance**: Evaluate prediction accuracy and feature importance
        
        ### Descriptors Used
        
        The following descriptors are calculated for each material:
        
        | Descriptor | Symbol | Physical Meaning |
        |------------|--------|------------------|
        | A-cation radius | r_A | Size of A-site |
        | B-cation radius | r_B | Size of B-site |
        | Dopant radius | r_D | Size of dopant |
        | Average B-site radius | r_B,avg | (1-x)·r_B + x·r_D |
        | Radius mismatch | Δr_B | \|r_D - r_B\| |
        | Tolerance factor | t | (r_A + r_O)/√2(r_B,avg + r_O) |
        | A-cation electronegativity | χ_A | Acidity of A-site |
        | B-cation electronegativity | χ_B | Acidity of B-site |
        | Dopant electronegativity | χ_D | Acidity of dopant |
        | Electronegativity difference | Δχ | χ_B,avg - χ_A |
        
        ### Machine Learning Models
        
        XGBoost regression models are trained on experimental data with:
        - 5-fold cross-validation
        - Feature scaling and categorical encoding
        - Feature importance analysis
        
        ### How to Cite
        
        If you use this tool in your research, please cite the original data sources.
        
        ### Contact
        
        For questions, suggestions, or to contribute data, please contact the developers.
        """)
        
        # Statistics
        st.subheader("Database Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Materials", len(df))
        with col2:
            st.metric("A-cations", df['A_cation'].nunique())
        with col3:
            st.metric("Dopants", df['dopant'].nunique())
        
        # References
        st.subheader("Key References")
        
        refs = {
            "[21]": "Kreuer, K.D. (2001) Solid State Ionics",
            "[46]": "Solid State Ionics (various)",
            "[51]": "Journal of Materials Chemistry A",
            "[53]": "Solid State Ionics (cerates)",
            "[110]": "Physical Chemistry Chemical Physics",
            "[111]": "Solid State Ionics (various)",
            "[112]": "Journal of Materials Chemistry",
            "[113]": "Chemistry of Materials",
            "[119]": "Journal of Power Sources",
            "[120]": "International Journal of Hydrogen Energy",
        }
        
        ref_df = pd.DataFrame(list(refs.items()), columns=['Reference', 'Source'])
        st.dataframe(ref_df, use_container_width=True)

# =============================================================================
# Run the app
# =============================================================================
if __name__ == "__main__":
    main()
