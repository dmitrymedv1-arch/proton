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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb
import shap
import warnings
from itertools import combinations
from scipy import stats
import joblib
from datetime import datetime
import time

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
    'Al': {'charge': 3, 'VI': 0.535},
    'Ga': {'charge': 3, 'VI': 0.62},
    'In': {'charge': 3, 'VI': 0.80},
    'Sc': {'charge': 3, 'VI': 0.745},
    'Y': {'charge': 3, 'VI': 0.90},
    'Yb': {'charge': 3, 'VI': 0.868},
    
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
        ['Ba', 'Zr', 'Y', 0.55, -83.4, -92.1, '[21]'],
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
        ['La', 'Sc', 'Sr', 0.09, -97, -112, '[122]'],
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
# Machine Learning Models with SHAP and advanced interpretation
# =============================================================================
@st.cache_resource
def train_prediction_models(df):
    """Train ML models for predicting delta_H and delta_S with advanced features"""
    
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
            desc['reference'] = row['reference']
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
    
    # =========================================================
    # Multiple models for comparison
    # =========================================================
    
    # 1. XGBoost (original)
    xgb_model_H = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8
    )
    xgb_model_S = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    xgb_model_H.fit(X_scaled, y_H)
    xgb_model_S.fit(X_scaled, y_S)
    
    # 2. Linear Regression with polynomial features (for interpretability)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)
    
    lr_model_H = LinearRegression()
    lr_model_S = LinearRegression()
    
    lr_model_H.fit(X_poly, y_H)
    lr_model_S.fit(X_poly, y_S)
    
    # 3. ElasticNet (for feature selection)
    elastic_H = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)
    elastic_S = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000)
    
    elastic_H.fit(X_scaled, y_H)
    elastic_S.fit(X_scaled, y_S)
    
    # Cross-validation scores
    cv_scores_H = cross_val_score(xgb_model_H, X_scaled, y_H, cv=min(5, len(X)), scoring='r2')
    cv_scores_S = cross_val_score(xgb_model_S, X_scaled, y_S, cv=min(5, len(X)), scoring='r2')
    
    # Feature importance (XGBoost)
    feature_importance_H = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model_H.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # ElasticNet coefficients (for interpretability)
    elastic_coef_H = pd.DataFrame({
        'feature': X.columns,
        'coefficient': elastic_H.coef_
    }).sort_values('coefficient', ascending=False)
    
    # =========================================================
    # SHAP analysis
    # =========================================================
    try:
        explainer_H = shap.TreeExplainer(xgb_model_H)
        shap_values_H = explainer_H.shap_values(X_scaled)
        shap_expected_H = explainer_H.expected_value
        
        explainer_S = shap.TreeExplainer(xgb_model_S)
        shap_values_S = explainer_S.shap_values(X_scaled)
        shap_expected_S = explainer_S.expected_value
        
        # SHAP summary for top features
        shap_summary_H = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': np.abs(shap_values_H).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
    except Exception as e:
        shap_values_H = None
        shap_values_S = None
        shap_expected_H = None
        shap_expected_S = None
        shap_summary_H = feature_importance_H.copy()
        shap_summary_H.columns = ['feature', 'mean_abs_shap']
    
    return {
        'models': {
            'xgb_H': xgb_model_H,
            'xgb_S': xgb_model_S,
            'lr_H': lr_model_H,
            'lr_S': lr_model_S,
            'elastic_H': elastic_H,
            'elastic_S': elastic_S
        },
        'poly': poly,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'cv_H_mean': cv_scores_H.mean(),
        'cv_S_mean': cv_scores_S.mean(),
        'cv_H_std': cv_scores_H.std(),
        'cv_S_std': cv_scores_S.std(),
        'feature_importance': feature_importance_H,
        'elastic_coef': elastic_coef_H,
        'shap_values_H': shap_values_H,
        'shap_values_S': shap_values_S,
        'shap_expected_H': shap_expected_H,
        'shap_expected_S': shap_expected_S,
        'shap_summary': shap_summary_H,
        'le_A': le_A,
        'le_B': le_B,
        'le_D': le_D,
        'X_train': X_scaled,
        'X_train_df': X,
        'y_train_H': y_H,
        'y_train_S': y_S,
        'df_features': df_features
    }, df_features

# =============================================================================
# Correlation analysis functions
# =============================================================================
def create_compensation_plot(df_features, group_by='B_cation'):
    """Create enhanced compensation plot with group-specific trends"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    groups = df_features[group_by].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    
    for idx, (group, color) in enumerate(zip(groups, colors)):
        if idx >= 6:  # Max 6 subplots
            break
            
        ax = axes[idx]
        subset = df_features[df_features[group_by] == group]
        
        ax.scatter(subset['delta_S'], subset['delta_H'], 
                  c=[color], s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Linear regression for this group
        if len(subset) >= 3:
            z = np.polyfit(subset['delta_S'], subset['delta_H'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(subset['delta_S'].min(), subset['delta_S'].max(), 50)
            ax.plot(x_trend, p(x_trend), '--', color=color, linewidth=1.5, alpha=0.8)
            
            # Calculate isokinetic temperature
            if abs(z[0]) > 1e-6:
                T_iso = -1000 / z[0]  # K
                ax.text(0.05, 0.95, f"T_iso = {T_iso:.0f} K",
                       transform=ax.transAxes, va='top', fontsize=9,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.set_xlabel('ΔS (J mol⁻¹ K⁻¹)', fontsize=10)
        ax.set_ylabel('ΔH (kJ mol⁻¹)', fontsize=10)
        ax.set_title(f'{group_by}: {group}', fontsize=11, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(groups), 6):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_2d_heatmap(df_features, x_col, y_col, z_col='delta_H', bins=20):
    """Create 2D heatmap of average ΔH in descriptor space"""
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(
        df_features[x_col], df_features[y_col], 
        bins=bins, weights=df_features[z_col]
    )
    counts, _, _ = np.histogram2d(
        df_features[x_col], df_features[y_col], bins=bins
    )
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        H_avg = np.where(counts > 0, H / counts, np.nan)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(H_avg.T, origin='lower', aspect='auto',
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                  cmap='RdBu_r', vmin=-150, vmax=-50)
    
    plt.colorbar(im, ax=ax, label=f'Average {z_col} (kJ mol⁻¹)')
    
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title(f'2D Hydration Map: {z_col} vs {x_col} and {y_col}')
    
    # Add scatter points on top
    ax.scatter(df_features[x_col], df_features[y_col], 
              c='black', s=20, alpha=0.5, edgecolors='white', linewidth=0.3)
    
    plt.tight_layout()
    return fig

def create_pairplot_with_correlation(df_features, feature_subset):
    """Create pairplot with correlation coefficients"""
    
    # Calculate correlation matrix
    corr_matrix = df_features[feature_subset].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create figure with subplots
    n_features = len(feature_subset)
    fig, axes = plt.subplots(n_features, n_features, figsize=(14, 14))
    
    for i, feat_i in enumerate(feature_subset):
        for j, feat_j in enumerate(feature_subset):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(df_features[feat_i], bins=15, color='steelblue', 
                       edgecolor='black', alpha=0.7)
                ax.set_xlabel('')
                ax.set_ylabel('')
                
            elif i < j:
                # Upper triangle: hide
                ax.set_visible(False)
                
            else:
                # Lower triangle: scatter plot
                ax.scatter(df_features[feat_j], df_features[feat_i], 
                          c='steelblue', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
                
                # Add correlation coefficient
                corr = corr_matrix.loc[feat_i, feat_j]
                ax.text(0.05, 0.95, f'r = {corr:.2f}', 
                       transform=ax.transAxes, va='top', fontsize=9,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                # Set labels only on edges
                if j == 0:
                    ax.set_ylabel(feat_i.replace('_', ' ').title(), fontsize=9)
                else:
                    ax.set_ylabel('')
                
                if i == n_features - 1:
                    ax.set_xlabel(feat_j.replace('_', ' ').title(), fontsize=9)
                else:
                    ax.set_xlabel('')
    
    plt.suptitle('Feature Correlation Matrix with Distributions', fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    return fig

# =============================================================================
# Progress bar context manager
# =============================================================================
class ProgressBar:
    def __init__(self, message, total_steps):
        self.message = message
        self.total_steps = total_steps
        self.progress_bar = None
        self.status_text = None
    
    def __enter__(self):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress_bar.empty()
        self.status_text.empty()
    
    def update(self, step, sub_message=""):
        progress = step / self.total_steps
        self.progress_bar.progress(progress)
        self.status_text.text(f"{self.message}: {sub_message} ({int(progress*100)}%)")

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
    - Advanced SHAP analysis and model interpretation
    - Find similar materials in the database
    """)
    
    # Initialize progress bar for data loading
    with ProgressBar("Loading data and training models", 4) as pb:
        pb.update(1, "Loading database")
        # Load data
        df = load_and_combine_data()
        
        pb.update(2, "Calculating descriptors")
        model_data, df_features = train_prediction_models(df)
        
        pb.update(3, "Training models")
        # Model training happens inside train_prediction_models
        
        pb.update(4, "Ready!")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📊 Data Explorer", "🔍 Correlations", "🤖 Predictor", "📈 Model Performance", "📊 SHAP Analysis", "ℹ️ About"]
    )
    
    # Sidebar statistics
    st.sidebar.markdown("---")
    st.sidebar.subheader("Database Stats")
    st.sidebar.metric("Total entries", len(df))
    st.sidebar.metric("A-cations", df['A_cation'].nunique())
    st.sidebar.metric("B-cations", df['B_cation'].nunique())
    st.sidebar.metric("Dopants", df['dopant'].nunique())
    
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
        
        # Coverage heatmap
        st.subheader("Data Coverage Matrix")
        pivot = filtered_df.pivot_table(
            index='A_cation', columns='B_cation',
            values='delta_H', aggfunc='count', fill_value=0
        )
        fig = px.imshow(pivot, text_auto=True, color_continuous_scale='Blues',
                       title='Number of samples per A-B combination')
        st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # Page 2: Correlations
    # =========================================================================
    elif page == "🔍 Correlations":
        st.header("🔍 Structure-Composition-Property Correlations")
        
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
                desc['reference'] = row['reference']
                desc_list.append(desc)
        
        plot_df = pd.DataFrame(desc_list)
        
        if plot_df.empty:
            st.warning("No data available with selected filters.")
            return
        
        # Select plot type with categories
        plot_category = st.selectbox(
            "Select correlation category",
            [
                "Compensation Effect (ΔH-ΔS)",
                "Structure Descriptors (tolerance factor, radii)",
                "Electronic Effects (electronegativity)",
                "Composition Trends",
                "Statistical Overview",
                "2D Hydration Maps"
            ]
        )
        
        if plot_category == "Compensation Effect (ΔH-ΔS)":
            st.subheader("ΔH–ΔS Compensation Effect by Material Families")
            
            group_by = st.selectbox("Group by", ["B_cation", "A_cation", "dopant"])
            
            fig = create_compensation_plot(plot_df, group_by)
            st.pyplot(fig)
            plt.close()
            
            st.markdown("""
            **Interpretation:**
            - The slope of each line gives the isokinetic temperature T_iso = -1000/slope
            - Different families show distinct compensation behavior
            - Higher T_iso indicates stronger proton-lattice coupling
            """)
            
        elif plot_category == "Structure Descriptors (tolerance factor, radii)":
            st.subheader("Structure-Property Correlations")
            
            plot_type = st.selectbox(
                "Select plot",
                [
                    "ΔH vs Tolerance Factor",
                    "ΔH vs B-site Average Radius",
                    "ΔH vs Dopant Radius",
                    "ΔH vs A-cation Radius",
                    "ΔH vs Radius Mismatch"
                ]
            )
            
            color_by = st.selectbox("Color by", ["A_cation", "B_cation", "dopant"])
            
            fig, ax = plt.subplots(figsize=(9, 6))
            
            if plot_type == "ΔH vs Tolerance Factor":
                x_col = 't_Goldschmidt'
                xlabel = 'Goldschmidt Tolerance Factor, t'
            elif plot_type == "ΔH vs B-site Average Radius":
                x_col = 'r_B_avg'
                xlabel = 'Average B-site Radius (Å)'
            elif plot_type == "ΔH vs Dopant Radius":
                x_col = 'r_D'
                xlabel = 'Dopant Ionic Radius (Å)'
            elif plot_type == "ΔH vs A-cation Radius":
                x_col = 'r_A'
                xlabel = 'A-cation Radius (Å)'
            elif plot_type == "ΔH vs Radius Mismatch":
                x_col = 'delta_r_B'
                xlabel = '|r_D - r_B| (Å)'
            
            groups = plot_df[color_by].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
            
            for group, color in zip(groups, colors):
                subset = plot_df[plot_df[color_by] == group]
                ax.scatter(subset[x_col], subset['delta_H'], 
                          label=group, c=[color], s=60, alpha=0.7, 
                          edgecolors='black', linewidth=0.5)
                
                # Add trend line for groups with enough points
                if len(subset) >= 4:
                    z = np.polyfit(subset[x_col], subset['delta_H'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(subset[x_col].min(), subset[x_col].max(), 50)
                    ax.plot(x_trend, p(x_trend), '--', color=color, linewidth=1, alpha=0.5)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel('ΔH (kJ mol⁻¹)')
            ax.set_title(f'{plot_type}')
            ax.legend(title=color_by, frameon=True, fancybox=False, edgecolor='black')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
            
        elif plot_category == "Electronic Effects (electronegativity)":
            st.subheader("Electronic Structure Correlations")
            
            plot_type = st.selectbox(
                "Select plot",
                [
                    "ΔH vs Dopant Electronegativity",
                    "ΔH vs B-site Average Electronegativity",
                    "ΔH vs Electronegativity Difference (χ_B_avg - χ_A)",
                    "ΔH vs A-cation Electronegativity"
                ]
            )
            
            color_by = st.selectbox("Color by", ["B_cation", "A_cation", "dopant"])
            
            fig, ax = plt.subplots(figsize=(9, 6))
            
            if plot_type == "ΔH vs Dopant Electronegativity":
                x_col = 'chi_D'
                xlabel = 'Dopant Electronegativity (Pauling)'
            elif plot_type == "ΔH vs B-site Average Electronegativity":
                x_col = 'chi_B_avg'
                xlabel = 'Average B-site Electronegativity'
            elif plot_type == "ΔH vs Electronegativity Difference (χ_B_avg - χ_A)":
                x_col = 'chi_diff'
                xlabel = 'χ_B_avg - χ_A'
            elif plot_type == "ΔH vs A-cation Electronegativity":
                x_col = 'chi_A'
                xlabel = 'A-cation Electronegativity'
            
            groups = plot_df[color_by].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
            
            for group, color in zip(groups, colors):
                subset = plot_df[plot_df[color_by] == group]
                ax.scatter(subset[x_col], subset['delta_H'], 
                          label=group, c=[color], s=60, alpha=0.7,
                          edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel('ΔH (kJ mol⁻¹)')
            ax.set_title(f'{plot_type}')
            ax.legend(title=color_by, frameon=True, fancybox=False, edgecolor='black')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
            
        elif plot_category == "Composition Trends":
            st.subheader("Composition-Dependent Behavior")
            
            plot_type = st.selectbox(
                "Select plot",
                [
                    "ΔH vs Dopant Content",
                    "ΔS vs Dopant Content",
                    "ΔH vs Content by Dopant Type",
                    "ΔH vs Content by B-cation"
                ]
            )
            
            if plot_type == "ΔH vs Dopant Content":
                fig, ax = plt.subplots(figsize=(9, 6))
                
                for dopant in plot_df['dopant'].unique():
                    subset = plot_df[plot_df['dopant'] == dopant]
                    if len(subset) >= 3:
                        ax.scatter(subset['content'], subset['delta_H'], 
                                  label=dopant, s=60, alpha=0.7, 
                                  edgecolors='black', linewidth=0.5)
                        
                        # Add line
                        subset_sorted = subset.sort_values('content')
                        ax.plot(subset_sorted['content'], subset_sorted['delta_H'], 
                               '--', linewidth=1, alpha=0.5)
                
                ax.set_xlabel('Dopant Content, x')
                ax.set_ylabel('ΔH (kJ mol⁻¹)')
                ax.set_title('ΔH vs Dopant Content by Dopant Type')
                ax.legend(title='Dopant', frameon=True, fancybox=False, edgecolor='black')
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
            elif plot_type == "ΔS vs Dopant Content":
                fig, ax = plt.subplots(figsize=(9, 6))
                
                for dopant in plot_df['dopant'].unique():
                    subset = plot_df[plot_df['dopant'] == dopant]
                    if len(subset) >= 3:
                        ax.scatter(subset['content'], subset['delta_S'], 
                                  label=dopant, s=60, alpha=0.7,
                                  edgecolors='black', linewidth=0.5)
                        
                        subset_sorted = subset.sort_values('content')
                        ax.plot(subset_sorted['content'], subset_sorted['delta_S'], 
                               '--', linewidth=1, alpha=0.5)
                
                ax.set_xlabel('Dopant Content, x')
                ax.set_ylabel('ΔS (J mol⁻¹ K⁻¹)')
                ax.set_title('ΔS vs Dopant Content by Dopant Type')
                ax.legend(title='Dopant', frameon=True, fancybox=False, edgecolor='black')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
            elif plot_type == "ΔH vs Content by Dopant Type":
                # Interactive plotly version
                fig = px.scatter(plot_df, x='content', y='delta_H', color='dopant',
                                hover_data=['A_cation', 'B_cation', 'delta_S', 'reference'],
                                trendline='lowess', trendline_options=dict(frac=0.3),
                                labels={'content': 'Dopant Content, x', 
                                       'delta_H': 'ΔH (kJ mol⁻¹)'},
                                title='ΔH vs Content by Dopant Type')
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot_type == "ΔH vs Content by B-cation":
                fig = px.scatter(plot_df, x='content', y='delta_H', color='B_cation',
                                hover_data=['A_cation', 'dopant', 'delta_S', 'reference'],
                                trendline='lowess', trendline_options=dict(frac=0.3),
                                labels={'content': 'Dopant Content, x',
                                       'delta_H': 'ΔH (kJ mol⁻¹)'},
                                title='ΔH vs Content by B-cation')
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_category == "Statistical Overview":
            st.subheader("Statistical Analysis of Hydration Parameters")
            
            plot_type = st.selectbox(
                "Select plot",
                [
                    "Violin Plot by B-cation",
                    "Box Plot by Dopant",
                    "Correlation Heatmap",
                    "Pairplot of Key Descriptors"
                ]
            )
            
            if plot_type == "Violin Plot by B-cation":
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # ΔH by B-cation
                sns.violinplot(data=plot_df, x='B_cation', y='delta_H', ax=ax1,
                              palette='Set2', cut=0)
                ax1.set_xlabel('B-cation')
                ax1.set_ylabel('ΔH (kJ mol⁻¹)')
                ax1.set_title('ΔH Distribution by B-cation')
                ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                
                # ΔS by B-cation
                sns.violinplot(data=plot_df, x='B_cation', y='delta_S', ax=ax2,
                              palette='Set2', cut=0)
                ax2.set_xlabel('B-cation')
                ax2.set_ylabel('ΔS (J mol⁻¹ K⁻¹)')
                ax2.set_title('ΔS Distribution by B-cation')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            elif plot_type == "Box Plot by Dopant":
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # ΔH by dopant
                sns.boxplot(data=plot_df, x='dopant', y='delta_H', ax=ax1,
                           palette='Set3')
                ax1.set_xlabel('Dopant')
                ax1.set_ylabel('ΔH (kJ mol⁻¹)')
                ax1.set_title('ΔH Distribution by Dopant')
                ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                
                # ΔS by dopant
                sns.boxplot(data=plot_df, x='dopant', y='delta_S', ax=ax2,
                           palette='Set3')
                ax2.set_xlabel('Dopant')
                ax2.set_ylabel('ΔS (J mol⁻¹ K⁻¹)')
                ax2.set_title('ΔS Distribution by Dopant')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            elif plot_type == "Correlation Heatmap":
                # Select numeric columns
                numeric_cols = ['content', 'r_A', 'r_B', 'r_D', 'r_B_avg', 
                               'delta_r_B', 't_Goldschmidt', 'chi_A', 'chi_B',
                               'chi_D', 'chi_B_avg', 'chi_diff', 'delta_H', 'delta_S']
                
                available_cols = [col for col in numeric_cols if col in plot_df.columns]
                corr_matrix = plot_df[available_cols].corr()
                
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                           center=0, square=True, ax=ax,
                           cbar_kws={'label': 'Correlation Coefficient'})
                ax.set_title('Descriptor Correlation Matrix')
                
                st.pyplot(fig)
                plt.close()
                
            elif plot_type == "Pairplot of Key Descriptors":
                key_features = ['content', 'r_B_avg', 't_Goldschmidt', 
                               'chi_diff', 'delta_H', 'delta_S']
                available_features = [f for f in key_features if f in plot_df.columns]
                
                if len(available_features) >= 2:
                    fig = create_pairplot_with_correlation(plot_df, available_features)
                    st.pyplot(fig)
                    plt.close()
        
        elif plot_category == "2D Hydration Maps":
            st.subheader("2D Hydration Property Maps")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X-axis", ['r_B_avg', 't_Goldschmidt', 'chi_diff', 'content'])
            with col2:
                y_col = st.selectbox("Y-axis", ['chi_diff', 'r_B_avg', 't_Goldschmidt', 'content'])
            
            fig = create_2d_heatmap(plot_df, x_col, y_col, 'delta_H')
            st.pyplot(fig)
            plt.close()
    
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
            # Logical grouping of A-cations
            a_cation = st.selectbox("A-cation", ['Ba', 'Sr', 'Ca', 'La', 'Nd', 'Gd'])
            
            # Define B-cation options based on A-cation valence requirement
            if a_cation in ['Ba', 'Sr', 'Ca']:
                # These require 4+ B-cations
                b_options = ['Ti', 'Zr', 'Sn', 'Ce', 'Hf']
                b_default = 'Zr'
            else:  # La, Nd, Gd - require 3+ B-cations
                b_options = ['Sc', 'In', 'Y', 'Yb', 'Gd', 'Lu', 'Al', 'Ga']
                b_default = 'Y'
        
        with col2:
            b_cation = st.selectbox("B-cation", b_options, index=b_options.index(b_default) if b_default in b_options else 0)
            
            # Define dopant options - all available 3+ dopants
            dopant_options = ['Sc', 'In', 'Y', 'Yb', 'Gd', 'Er', 'Dy', 'Ho', 'Tm', 'Lu', 'Al', 'Ga']
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
        try:
            X_pred['A_enc'] = model_data['le_A'].transform([a_cation])[0] if a_cation in model_data['le_A'].classes_ else -1
        except:
            X_pred['A_enc'] = -1
        
        try:
            X_pred['B_enc'] = model_data['le_B'].transform([b_cation])[0] if b_cation in model_data['le_B'].classes_ else -1
        except:
            X_pred['B_enc'] = -1
        
        try:
            X_pred['D_enc'] = model_data['le_D'].transform([dopant])[0] if dopant in model_data['le_D'].classes_ else -1
        except:
            X_pred['D_enc'] = -1
        
        # Ensure all required features are present
        for col in model_data['feature_names']:
            if col not in X_pred.columns:
                X_pred[col] = 0
        
        X_pred = X_pred[model_data['feature_names']]
        
        # Scale features
        X_pred_scaled = model_data['scaler'].transform(X_pred)
        
        # Make predictions with multiple models
        pred_H_xgb = model_data['models']['xgb_H'].predict(X_pred_scaled)[0]
        pred_S_xgb = model_data['models']['xgb_S'].predict(X_pred_scaled)[0]
        
        # ElasticNet predictions
        pred_H_elastic = model_data['models']['elastic_H'].predict(X_pred_scaled)[0]
        pred_S_elastic = model_data['models']['elastic_S'].predict(X_pred_scaled)[0]
        
        # Ensemble prediction (average)
        pred_H = np.mean([pred_H_xgb, pred_H_elastic])
        pred_S = np.mean([pred_S_xgb, pred_S_elastic])
        
        # Calculate prediction uncertainty (based on model disagreement)
        H_std = np.std([pred_H_xgb, pred_H_elastic])
        S_std = np.std([pred_S_xgb, pred_S_elastic])
        
        # Display predictions
        st.subheader("Predicted Hydration Parameters")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "ΔH (kJ mol⁻¹)",
                f"{pred_H:.1f} ± {H_std:.1f}",
                delta=None
            )
        with col2:
            st.metric(
                "ΔS (J mol⁻¹ K⁻¹)",
                f"{pred_S:.1f} ± {S_std:.1f}",
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
        with col4:
            # Calculate hydration equilibrium constant at 600°C
            T = 600 + 273.15
            R = 0.008314  # kJ/mol·K
            K_hyd = np.exp(-(pred_H - T*pred_S/1000)/(R*T))
            st.metric(
                "K_hyd (600°C)",
                f"{K_hyd:.2e}",
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
                'Distance': distances[0][i],
                'Reference': df_features.iloc[idx].get('reference', 'N/A')
            })
        
        st.dataframe(pd.DataFrame(similar_data), use_container_width=True)
        
        # Model comparison
        st.subheader("Model Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**XGBoost Prediction**")
            st.markdown(f"ΔH = {pred_H_xgb:.1f} kJ/mol")
            st.markdown(f"ΔS = {pred_S_xgb:.1f} J/mol·K")
        
        with col2:
            st.markdown("**ElasticNet Prediction**")
            st.markdown(f"ΔH = {pred_H_elastic:.1f} kJ/mol")
            st.markdown(f"ΔS = {pred_S_elastic:.1f} J/mol·K")
    
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
                f"{model_data['cv_H_mean']:.3f} ± {model_data['cv_H_std']:.3f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "ΔS Model - R² (CV)",
                f"{model_data['cv_S_mean']:.3f} ± {model_data['cv_S_std']:.3f}",
                delta=None
            )
        
        # Feature importance (XGBoost)
        st.subheader("XGBoost Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        importance_df = model_data['feature_importance']
        top20 = importance_df.head(20)
        
        colors = ['darkred' if 'chi' in f or 'electro' in f else 
                 'darkblue' if 'r_' in f or 'radius' in f else 
                 'darkgreen' if 'content' in f else
                 'purple' for f in top20['feature']]
        
        ax.barh(range(len(top20)), top20['importance'], color=colors, edgecolor='black')
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20['feature'])
        ax.set_xlabel('Importance')
        ax.set_title('XGBoost Feature Importance for ΔH Prediction')
        ax.invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkred', label='Electronegativity'),
            Patch(facecolor='darkblue', label='Ionic Radius'),
            Patch(facecolor='darkgreen', label='Composition'),
            Patch(facecolor='purple', label='Categorical')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        st.pyplot(fig)
        plt.close()
        
        # ElasticNet coefficients (interpretable model)
        st.subheader("ElasticNet Coefficients (Interpretable Linear Model)")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        coef_df = model_data['elastic_coef'].head(15)
        
        colors = ['red' if c < 0 else 'blue' for c in coef_df['coefficient']]
        ax.barh(range(len(coef_df)), coef_df['coefficient'], 
                color=colors, edgecolor='black', alpha=0.7)
        ax.set_yticks(range(len(coef_df)))
        ax.set_yticklabels(coef_df['feature'])
        ax.set_xlabel('Coefficient (effect on ΔH)')
        ax.set_title('ElasticNet Coefficients (more negative → stronger hydration)')
        ax.axvline(x=0, color='black', linewidth=0.5, linestyle='-')
        ax.invert_yaxis()
        
        st.pyplot(fig)
        plt.close()
        
        # Prediction vs Actual plots
        st.subheader("Prediction vs Actual (Training Data)")
        
        # Get predictions on training data
        train_pred_H = model_data['models']['xgb_H'].predict(model_data['X_train'])
        train_pred_S = model_data['models']['xgb_S'].predict(model_data['X_train'])
        
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
            ax.grid(True, alpha=0.3)
            
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
            ax.grid(True, alpha=0.3)
            
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
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(train_pred_S, residuals_S, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
            ax.set_xlabel('Predicted ΔS (J mol⁻¹ K⁻¹)')
            ax.set_ylabel('Residuals')
            ax.set_title(f'ΔS Residuals (MAE = {mean_absolute_error(model_data["y_train_S"], train_pred_S):.1f})')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        # Learning curves
        st.subheader("Learning Curves")
        
        # Generate learning curve data
        train_sizes = np.linspace(0.3, 1.0, 5)
        train_scores_H = []
        val_scores_H = []
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for size in train_sizes:
            n_samples = int(len(model_data['X_train']) * size)
            scores = []
            for train_idx, val_idx in kf.split(model_data['X_train'][:n_samples]):
                X_tr, X_val = model_data['X_train'][train_idx], model_data['X_train'][val_idx]
                y_tr, y_val = model_data['y_train_H'].iloc[train_idx], model_data['y_train_H'].iloc[val_idx]
                
                model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
                model.fit(X_tr, y_tr)
                
                train_scores_H.append(r2_score(y_tr, model.predict(X_tr)))
                val_scores_H.append(r2_score(y_val, model.predict(X_val)))
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, np.mean(train_scores_H), 'o-', label='Training score', color='blue')
        ax.plot(train_sizes, np.mean(val_scores_H), 'o-', label='Validation score', color='red')
        ax.set_xlabel('Training set size')
        ax.set_ylabel('R² Score')
        ax.set_title('Learning Curves for ΔH Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close()
    
    # =========================================================================
    # Page 5: SHAP Analysis
    # =========================================================================
    elif page == "📊 SHAP Analysis":
        st.header("📊 SHAP Analysis for Model Interpretability")
        
        if model_data is None or model_data['shap_values_H'] is None:
            st.warning("SHAP analysis not available. Model may be too small or training failed.")
            return
        
        st.markdown("""
        SHAP (SHapley Additive exPlanations) values show how each feature contributes to the prediction.
        - **Red** = high feature value, **Blue** = low feature value
        - **Positive SHAP** = increases ΔH (less negative → weaker hydration)
        - **Negative SHAP** = decreases ΔH (more negative → stronger hydration)
        """)
        
        # SHAP summary plot
        st.subheader("SHAP Summary Plot - ΔH")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for SHAP summary
        shap_values = model_data['shap_values_H']
        X_display = model_data['X_train_df']
        
        # Calculate mean absolute SHAP values
        shap_summary = model_data['shap_summary']
        
        # Create beeswarm plot manually
        top_features = shap_summary.head(10)['feature'].values
        
        y_pos = np.arange(len(top_features))
        for i, feat in enumerate(top_features):
            feat_idx = list(X_display.columns).index(feat)
            shap_vals = shap_values[:, feat_idx]
            feat_vals = X_display.iloc[:, feat_idx]
            
            # Normalize feature values for coloring
            feat_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-10)
            
            # Add jitter for beeswarm effect
            jitter = np.random.normal(0, 0.05, len(shap_vals))
            y_jitter = i + jitter
            
            scatter = ax.scatter(shap_vals, y_jitter, c=feat_norm, 
                               cmap='coolwarm', alpha=0.6, s=20,
                               edgecolors='black', linewidth=0.3)
            
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.set_xlabel('SHAP value (impact on ΔH)')
        ax.set_title('SHAP Summary: Feature Impact on ΔH Prediction')
        
        plt.colorbar(scatter, ax=ax, label='Feature value')
        st.pyplot(fig)
        plt.close()
        
        # SHAP bar plot
        st.subheader("Mean |SHAP| Values")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        top_shap = shap_summary.head(15)
        ax.barh(range(len(top_shap)), top_shap['mean_abs_shap'], 
                color='steelblue', edgecolor='black')
        ax.set_yticks(range(len(top_shap)))
        ax.set_yticklabels(top_shap['feature'])
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title('Average Impact on Model Output Magnitude')
        ax.invert_yaxis()
        
        st.pyplot(fig)
        plt.close()
        
        # Partial Dependence Plots
        st.subheader("Partial Dependence Plots")
        
        st.markdown("""
        Partial dependence plots show how ΔH changes when varying one feature while keeping others constant.
        """)
        
        feature_for_pdp = st.selectbox(
            "Select feature for partial dependence plot",
            ['content', 'r_B_avg', 't_Goldschmidt', 'chi_diff', 'r_D']
        )
        
        if feature_for_pdp in X_display.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Calculate partial dependence
            feature_idx = list(X_display.columns).index(feature_for_pdp)
            feature_vals = X_display.iloc[:, feature_idx]
            
            # Create grid
            grid = np.linspace(feature_vals.min(), feature_vals.max(), 50)
            
            # For each grid point, average predictions with that feature value
            pdp_values = []
            for val in grid:
                X_temp = X_display.copy()
                X_temp.iloc[:, feature_idx] = val
                X_temp_scaled = model_data['scaler'].transform(X_temp[model_data['feature_names']])
                preds = model_data['models']['xgb_H'].predict(X_temp_scaled)
                pdp_values.append(preds.mean())
            
            ax.plot(grid, pdp_values, 'b-', linewidth=2)
            ax.fill_between(grid, 
                           np.array(pdp_values) - np.std(pdp_values),
                           np.array(pdp_values) + np.std(pdp_values),
                           alpha=0.2, color='blue')
            
            # Add rug plot of actual values
            ax.plot(feature_vals, [pdp_values[0]]*len(feature_vals), 
                   '|', color='red', alpha=0.5, markersize=10)
            
            ax.set_xlabel(feature_for_pdp.replace('_', ' ').title())
            ax.set_ylabel('Partial dependence of ΔH (kJ/mol)')
            ax.set_title(f'Partial Dependence Plot: {feature_for_pdp}')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
    
    # =========================================================================
    # Page 6: About
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
        
        ### New Features in Version 2.0
        
        - **Enhanced correlation plots**: Group-specific trends, 2D hydration maps
        - **SHAP analysis**: Model interpretability showing feature impacts
        - **Multiple ML models**: XGBoost, ElasticNet, ensemble predictions
        - **Partial dependence plots**: Understand individual feature effects
        - **Coverage heatmaps**: Visualize data distribution
        - **Progress indicators**: Real-time feedback during computations
        
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
        
        Multiple models are trained on experimental data:
        - **XGBoost**: High accuracy, captures non-linear relationships
        - **ElasticNet**: Linear model with regularization, interpretable
        - **Ensemble**: Combined predictions for robustness
        
        Features:
        - 5-fold cross-validation
        - Feature scaling and categorical encoding
        - SHAP values for interpretability
        - Partial dependence plots
        - Uncertainty estimation
        
        ### How to Cite
        
        If you use this tool in your research, please cite the original data sources.
        
        ### Version History
        
        - **2.0** (March 2025): Added SHAP analysis, enhanced correlations, multiple models
        - **1.0** (January 2025): Initial release with basic predictions
        
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
            "[21]": "Kreuer, K.D. (2001) Aspects of the formation and mobility of protonic charge carriers and the stability of perovskite-type oxides. Solid State Ionics",
            "[46]": "Multiple sources on BaTiO₃-based proton conductors",
            "[51]": "Journal of Materials Chemistry A - BaZrO₃ hydration",
            "[53]": "Solid State Ionics - cerate perovskites",
            "[110]": "Physical Chemistry Chemical Physics - hydration thermodynamics",
            "[111]": "Solid State Ionics (various) - comparative studies",
            "[112]": "Journal of Materials Chemistry - Sc-doped systems",
            "[113]": "Chemistry of Materials - Y-doped BaZrO₃",
            "[119]": "Journal of Power Sources - BaSnO₃-based conductors",
            "[120]": "International Journal of Hydrogen Energy - Sn-based perovskites",
            "[121]": "Acta Materialia - hydration mechanisms",
            "[122]": "Journal of the Electrochemical Society - La-based conductors",
            "[123]": "Solid State Ionics - LaScO₃ systems",
            "[124]": "Chemistry of Materials - Yb-doped La-based perovskites",
            "[127,128]": "Multiple sources on complex perovskites"
        }
        
        ref_df = pd.DataFrame(list(refs.items()), columns=['Reference', 'Source'])
        st.dataframe(ref_df, use_container_width=True, height=300)

# =============================================================================
# Run the app
# =============================================================================
if __name__ == "__main__":
    main()
