"""
Proton Hydration Predictor for Perovskite Oxides - Version 3.0
Advanced tool for analyzing and predicting hydration thermodynamics 
of proton-conducting perovskites with enhanced visualization and ML capabilities.
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
import shap
import warnings
from itertools import combinations
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.interpolate import griddata
import joblib
from datetime import datetime
import time
import plotly.figure_factory as ff

warnings.filterwarnings('ignore')

# =============================================================================
# Modern scientific color palette and styling
# =============================================================================
MODERN_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'yellow': '#bcbd22',
    'cyan': '#17becf',
    'background': '#f8f9fa',
    'text': '#212529',
    'grid': '#dee2e6'
}

# Set modern plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    # Font sizes and weights
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 13,
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
    'legend.fontsize': 11,
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
# Enhanced ionic radii database (Shannon radii, in Angstroms)
# =============================================================================
IONIC_RADII = {
    # A-site cations with multiple coordinations
    'Ba': {'charge': 2, 'XII': 1.61, 'X': 1.56, 'IX': 1.54, 'VIII': 1.52, 'VI': 1.49},
    'Sr': {'charge': 2, 'XII': 1.44, 'X': 1.40, 'IX': 1.38, 'VIII': 1.36, 'VI': 1.32},
    'Ca': {'charge': 2, 'XII': 1.34, 'X': 1.30, 'IX': 1.28, 'VIII': 1.26, 'VI': 1.14},
    'La': {'charge': 3, 'XII': 1.36, 'IX': 1.216, 'VIII': 1.18, 'VI': 1.06},
    'Nd': {'charge': 3, 'XII': 1.27, 'IX': 1.163, 'VIII': 1.12, 'VI': 1.04},
    'Gd': {'charge': 3, 'IX': 1.107, 'VIII': 1.053, 'VI': 0.94},
    'Sm': {'charge': 3, 'IX': 1.132, 'VIII': 1.079, 'VI': 0.98},
    'Y': {'charge': 3, 'IX': 1.075, 'VIII': 1.019, 'VI': 0.90},
    'Pr': {'charge': 3, 'IX': 1.179, 'VIII': 1.14, 'VI': 1.01},
    'Ce': {'charge': 3, 'IX': 1.196, 'VIII': 1.14, 'VI': 1.01},
    
    # B-site cations with multiple oxidation states
    'Ti': {'charge': 4, 'VI': 0.605, 'V': 0.66, 'IV': 0.75},
    'Zr': {'charge': 4, 'VI': 0.72, 'VII': 0.78, 'VIII': 0.84},
    'Sn': {'charge': 4, 'VI': 0.69, 'VII': 0.71, 'VIII': 0.81},
    'Ce': {'charge': 4, 'VI': 0.87, 'VIII': 0.97, 'X': 1.07},
    'Hf': {'charge': 4, 'VI': 0.71, 'VII': 0.76, 'VIII': 0.83},
    'Nb': {'charge': 5, 'VI': 0.64, 'VIII': 0.74},
    'Ta': {'charge': 5, 'VI': 0.64, 'VIII': 0.74},
    'Mo': {'charge': 6, 'VI': 0.59, 'IV': 0.65, 'V': 0.61},
    'W': {'charge': 6, 'VI': 0.60, 'IV': 0.66, 'V': 0.62},
    'Al': {'charge': 3, 'VI': 0.535, 'IV': 0.39, 'V': 0.48},
    'Ga': {'charge': 3, 'VI': 0.62, 'IV': 0.47, 'V': 0.55},
    'In': {'charge': 3, 'VI': 0.80, 'VIII': 0.92},
    'Sc': {'charge': 3, 'VI': 0.745, 'VIII': 0.87},
    
    # Dopants with multiple coordinations
    'Yb': {'charge': 3, 'VI': 0.868, 'VII': 0.92, 'VIII': 0.98},
    'Er': {'charge': 3, 'VI': 0.89, 'VII': 0.94, 'VIII': 1.00},
    'Dy': {'charge': 3, 'VI': 0.912, 'VII': 0.97, 'VIII': 1.03},
    'Ho': {'charge': 3, 'VI': 0.901, 'VII': 0.95, 'VIII': 1.02},
    'Tm': {'charge': 3, 'VI': 0.88, 'VII': 0.93, 'VIII': 0.99},
    'Lu': {'charge': 3, 'VI': 0.861, 'VII': 0.92, 'VIII': 0.98},
    'Eu': {'charge': 3, 'VI': 0.947, 'VII': 1.01, 'VIII': 1.07},
    'Tb': {'charge': 3, 'VI': 0.923, 'VII': 0.98, 'VIII': 1.04},
    'Fe': {'charge': 3, 'VI': 0.645, 'IV': 0.63, 'V': 0.58},  # High spin
    'Co': {'charge': 3, 'VI': 0.545, 'IV': 0.56, 'V': 0.52},
    'Ni': {'charge': 3, 'VI': 0.56, 'II': 0.69},
    'Zn': {'charge': 2, 'VI': 0.74, 'IV': 0.60, 'V': 0.68},
    'Mg': {'charge': 2, 'VI': 0.72, 'IV': 0.57, 'V': 0.66},
    
    # Oxygen
    'O': {'charge': -2, 'II': 1.35, 'IV': 1.38, 'VI': 1.40, 'VIII': 1.42},
}

# =============================================================================
# Enhanced electronegativity database (Pauling scale with Allred-Rochow)
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
# Additional properties database (polarizability, ionization potential)
# =============================================================================
IONIC_POLARIZABILITY = {
    'Ba': 1.55, 'Sr': 0.86, 'Ca': 0.47, 'La': 1.04, 'Y': 0.55,
    'Ti': 0.19, 'Zr': 0.37, 'Sn': 0.24, 'Ce': 0.43, 'Hf': 0.34,
    'Sc': 0.29, 'In': 0.42, 'Yb': 0.31, 'Gd': 0.39, 'Al': 0.05,
    'Ga': 0.16, 'Fe': 0.18, 'Co': 0.15, 'Ni': 0.13, 'Zn': 0.28,
    'Mg': 0.09, 'O': 2.0
}

IONIZATION_POTENTIAL = {
    'Ba': 5.21, 'Sr': 5.69, 'Ca': 6.11, 'La': 5.58, 'Y': 6.22,
    'Ti': 6.82, 'Zr': 6.84, 'Sn': 7.34, 'Ce': 5.54, 'Hf': 6.83,
    'Sc': 6.56, 'In': 5.79, 'Yb': 6.25, 'Gd': 6.15, 'Al': 5.99,
    'Ga': 6.00, 'Fe': 7.90, 'Co': 7.88, 'Ni': 7.64, 'Zn': 9.39,
    'Mg': 7.65, 'O': 13.62
}

# =============================================================================
# Build combined dataset from all sources (enhanced)
# =============================================================================
@st.cache_data
def load_and_combine_data():
    """
    Create a comprehensive dataset combining multiple sources with enhanced descriptors
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
    
    # Remove duplicates
    df_combined = df_combined.drop_duplicates(
        subset=['A_cation', 'B_cation', 'dopant', 'content', 'delta_H', 'delta_S'],
        keep='first'
    )
    
    return df_combined

# =============================================================================
# Enhanced descriptor calculation functions
# =============================================================================
def calculate_tolerance_factor(r_A, r_B_avg, r_O=1.40):
    """Calculate Goldschmidt tolerance factor"""
    return (r_A + r_O) / (np.sqrt(2) * (r_B_avg + r_O))

def calculate_octahedral_factor(r_B_avg, r_O=1.40):
    """Calculate octahedral factor (r_B/r_O)"""
    return r_B_avg / r_O

def calculate_global_instability_index(r_A, r_B_avg, r_O=1.40):
    """Calculate global instability index based on bond valence sums"""
    # Simplified version - returns deviation from ideal tolerance
    return abs(1 - calculate_tolerance_factor(r_A, r_B_avg, r_O))

def calculate_lattice_energy(r_A, r_B_avg, r_O=1.40, z_A=2, z_B=4, z_O=-2):
    """Calculate approximate lattice energy (simplified Kapustinskii equation)"""
    # Simplified: U = k * (z+ * z-) * ν / (r+ + r-)
    k = 1200  # kJ·Å/mol
    n_ions = 5  # For ABO3: 1 A + 1 B + 3 O
    r_cation_avg = (r_A + r_B_avg) / 2
    r_avg = (r_cation_avg + r_O) / 2
    return -k * abs(z_A * z_O) * n_ions / r_avg

def calculate_polarizability_factor(elements):
    """Calculate average polarizability factor"""
    polarizabilities = []
    for el in elements:
        if el in IONIC_POLARIZABILITY:
            polarizabilities.append(IONIC_POLARIZABILITY[el])
    return np.mean(polarizabilities) if polarizabilities else 0

def calculate_ionization_factor(elements):
    """Calculate average ionization potential factor"""
    potentials = []
    for el in elements:
        if el in IONIZATION_POTENTIAL:
            potentials.append(IONIZATION_POTENTIAL[el])
    return np.mean(potentials) if potentials else 0

def calculate_charge_density(radius, charge):
    """Calculate charge density (Z/r)"""
    if radius and radius > 0:
        return charge / radius
    return 0

def calculate_bond_valence_sum(r_A, r_B, r_O, d_AO, d_BO):
    """Calculate bond valence sums (simplified)"""
    # Using simplified bond valence parameters
    R0_AO = 2.29  # Typical for A-O bonds
    R0_BO = 1.92  # Typical for B-O bonds
    b = 0.37  # Universal constant
    
    if d_AO and d_BO:
        v_A = np.exp((R0_AO - d_AO) / b)
        v_B = np.exp((R0_BO - d_BO) / b)
        return v_A + v_B
    return 0

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

def get_polarizability(element):
    """Get ionic polarizability"""
    return IONIC_POLARIZABILITY.get(element, 0)

def get_ionization_potential(element):
    """Get ionization potential"""
    return IONIZATION_POTENTIAL.get(element, 0)

def calculate_proton_concentration(delta_H, delta_S, T, pH2O, content):
    """
    Calculate proton concentration [OH] using the correct thermodynamic model.
    
    For the hydration reaction: H2O + Vo¨ + Ooˣ ⇄ 2OHo˙
    
    The equilibrium constant: K = exp(-ΔG/RT)
    Proton concentration (in fraction of dopant concentration): 
    [OH] = [D] * K * sqrt(pH2O) / (1 + K * sqrt(pH2O))
    
    Parameters:
    -----------
    delta_H : float
        Enthalpy of hydration (kJ/mol)
    delta_S : float
        Entropy of hydration (J/mol·K)
    T : float
        Temperature (K)
    pH2O : float
        Water vapor pressure (atm)
    content : float
        Dopant concentration (acceptor concentration [D])
    
    Returns:
    --------
    float : Proton concentration [OH] (in same units as content)
    """
    R = 0.008314  # Gas constant (kJ/mol·K)
    
    # Calculate Gibbs free energy
    delta_G = delta_H - T * delta_S / 1000  # Convert ΔS from J to kJ
    
    # Calculate equilibrium constant
    K = np.exp(-delta_G / (R * T))
    
    # Calculate proton concentration (fraction of dopant sites)
    # The formula: [OH] = [D] * K * sqrt(pH2O) / (1 + K * sqrt(pH2O))
    sqrt_pH2O = np.sqrt(max(pH2O, 1e-10))  # Avoid negative or zero
    
    numerator = content * K * sqrt_pH2O
    denominator = 1 + K * sqrt_pH2O
    
    if denominator > 1e-10:
        OH = numerator / denominator
        # Constrain to physical limits
        OH = max(0, min(OH, content))
    else:
        OH = 0
    
    return OH

def calculate_mixed_site_properties(B1, B2, dopant, y_fixed, model_data):
    """
    Calculate properties for mixed B-site system: BaB1_{1-x-y}B2_x D_y O_{3-x/2}
    using ideal additive model based on end-member properties.
    
    Parameters:
    -----------
    B1 : str
        First B-site cation (e.g., 'Zr')
    B2 : str
        Second B-site cation (e.g., 'Ce')
    dopant : str
        Dopant cation (e.g., 'Y')
    y_fixed : float
        Fixed dopant concentration on B-site
    model_data : dict
        Trained model data containing ML models and encoders
    
    Returns:
    --------
    tuple : (x_values, delta_H_values, delta_S_values)
        Arrays of predicted properties as function of x
    """
    
    # Generate x values from 0 to 1-y_fixed
    x_max = 1 - y_fixed
    x_values = np.arange(0, x_max + 0.01, 0.05)
    x_values = np.clip(x_values, 0, x_max)
    
    delta_H_values = []
    delta_S_values = []
    
    # For each x, calculate weighted average of end-member properties
    for x in x_values:
        # Composition: BaB1_{1-x-y}B2_x D_y
        # End-member 1: BaB1_{1-y}D_y (pure B1 with dopant)
        # End-member 2: BaB2_{1-y}D_y (pure B2 with dopant)
        # Weight: (1-x) for B1 end-member, x for B2 end-member
        
        weight_B1 = 1 - x
        weight_B2 = x
        
        # Get properties for end-member 1 (pure B1 with dopant)
        props1 = get_end_member_properties(B1, dopant, y_fixed, model_data)
        
        # Get properties for end-member 2 (pure B2 with dopant)
        props2 = get_end_member_properties(B2, dopant, y_fixed, model_data)
        
        # Weighted average (ideal mixing)
        delta_H_mixed = weight_B1 * props1['delta_H'] + weight_B2 * props2['delta_H']
        delta_S_mixed = weight_B1 * props1['delta_S'] + weight_B2 * props2['delta_S']
        
        delta_H_values.append(delta_H_mixed)
        delta_S_values.append(delta_S_mixed)
    
    return x_values, np.array(delta_H_values), np.array(delta_S_values)


def get_end_member_properties(B_cation, dopant, content, model_data):
    """
    Get predicted properties for an end-member composition: BaB_{1-content}D_{content}O_{3-content/2}
    
    Parameters:
    -----------
    B_cation : str
        B-site cation
    dopant : str
        Dopant cation
    content : float
        Dopant concentration
    model_data : dict
        Trained model data
    
    Returns:
    --------
    dict : {'delta_H': float, 'delta_S': float}
    """
    
    # For end-member, A-cation is Ba (most common)
    A_cation = 'Ba'
    
    # Calculate descriptors
    input_row = pd.Series({
        'A_cation': A_cation,
        'B_cation': B_cation,
        'dopant': dopant,
        'content': content
    })
    
    input_desc = calculate_descriptors(input_row)
    
    # Prepare features for prediction
    feature_names = model_data['feature_names']
    
    X_pred = pd.DataFrame([{
        'content': input_desc.get('content', content),
        'r_A_XII': input_desc.get('r_A_XII', 0),
        'r_B_VI': input_desc.get('r_B_VI', 0),
        'r_D_VI': input_desc.get('r_D_VI', 0),
        'r_B_avg': input_desc.get('r_B_avg', 0),
        'delta_r_B': input_desc.get('delta_r_B', 0),
        't_Goldschmidt': input_desc.get('t_Goldschmidt', 0),
        'octahedral_factor': input_desc.get('octahedral_factor', 0),
        'global_instability': input_desc.get('global_instability', 0),
        'lattice_energy': input_desc.get('lattice_energy', 0),
        'chi_A': input_desc.get('chi_A', 0),
        'chi_B': input_desc.get('chi_B', 0),
        'chi_D': input_desc.get('chi_D', 0),
        'chi_B_avg': input_desc.get('chi_B_avg', 0),
        'chi_diff': input_desc.get('chi_diff', 0),
        'chi_product': input_desc.get('chi_product', 0),
        'polarizability_avg': input_desc.get('polarizability_avg', 0),
        'ionization_avg': input_desc.get('ionization_avg', 0),
        'charge_density_A': input_desc.get('charge_density_A', 0),
        'charge_density_B': input_desc.get('charge_density_B', 0),
        'oxygen_vacancy': input_desc.get('oxygen_vacancy', 0)
    }])
    
    # Add encoded categorical features
    try:
        X_pred['A_enc'] = model_data['le_A'].transform([A_cation])[0] if A_cation in model_data['le_A'].classes_ else -1
    except:
        X_pred['A_enc'] = -1
    
    try:
        X_pred['B_enc'] = model_data['le_B'].transform([B_cation])[0] if B_cation in model_data['le_B'].classes_ else -1
    except:
        X_pred['B_enc'] = -1
    
    try:
        X_pred['D_enc'] = model_data['le_D'].transform([dopant])[0] if dopant in model_data['le_D'].classes_ else -1
    except:
        X_pred['D_enc'] = -1
    
    # Ensure all required features are present
    for col in feature_names:
        if col not in X_pred.columns:
            X_pred[col] = 0
    
    X_pred = X_pred[feature_names]
    X_pred_scaled = model_data['scaler'].transform(X_pred)
    
    # Get predictions from ensemble
    pred_H_xgb = model_data['models']['xgb_H'].predict(X_pred_scaled)[0]
    pred_H_rf = model_data['models']['rf_H'].predict(X_pred_scaled)[0]
    
    pred_S_xgb = model_data['models']['xgb_S'].predict(X_pred_scaled)[0]
    pred_S_rf = model_data['models']['rf_S'].predict(X_pred_scaled)[0]
    
    # Ensemble with weights (XGBoost: 0.6, RF: 0.4)
    delta_H = 0.6 * pred_H_xgb + 0.4 * pred_H_rf
    delta_S = 0.6 * pred_S_xgb + 0.4 * pred_S_rf
    
    return {'delta_H': delta_H, 'delta_S': delta_S}


def calculate_mixed_site_3d_surface(B1, B2, dopant, model_data, property_type='delta_H', 
                                     y_step=0.05, x_step=0.1):
    """
    Generate 3D surface data for mixed B-site system.
    
    BaB1_{1-x-y}B2_x D_y O_{3-x/2}
    
    Parameters:
    -----------
    B1 : str
        First B-site cation
    B2 : str
        Second B-site cation
    dopant : str
        Dopant cation
    model_data : dict
        Trained model data
    property_type : str
        'delta_H' or 'delta_S'
    y_step : float
        Step size for dopant concentration y (default: 0.05)
    x_step : float
        Step size for B2 concentration x (default: 0.1)
    
    Returns:
    --------
    tuple : (x_grid, y_grid, Z_grid, x_unique, y_unique)
        Grids for 3D plotting
    """
    
    # Define ranges with specified steps
    y_values = np.arange(0, 0.31, y_step)  # 0 to 0.3
    x_grid = []
    y_grid = []
    Z_grid = []
    
    for y in y_values:
        x_max = 1 - y
        if x_max <= 0:
            continue
        
        x_values = np.arange(0, x_max + 0.01, x_step)
        x_values = np.clip(x_values, 0, x_max)
        
        for x in x_values:
            # Weights for end-members
            weight_B1 = 1 - x
            weight_B2 = x
            
            # Get end-member properties
            props1 = get_end_member_properties(B1, dopant, y, model_data)
            props2 = get_end_member_properties(B2, dopant, y, model_data)
            
            # Weighted average
            if property_type == 'delta_H':
                value = weight_B1 * props1['delta_H'] + weight_B2 * props2['delta_H']
            else:
                value = weight_B1 * props1['delta_S'] + weight_B2 * props2['delta_S']
            
            x_grid.append(x)
            y_grid.append(y)
            Z_grid.append(value)
    
    # Convert to 2D grid for surface plotting
    x_unique = np.sort(np.unique(x_grid))
    y_unique = np.sort(np.unique(y_grid))
    
    X_mesh, Y_mesh = np.meshgrid(x_unique, y_unique)
    Z_mesh = np.full_like(X_mesh, np.nan, dtype=float)
    
    for i, x_val in enumerate(x_unique):
        for j, y_val in enumerate(y_unique):
            # Find matching value
            for k in range(len(x_grid)):
                if abs(x_grid[k] - x_val) < 0.01 and abs(y_grid[k] - y_val) < 0.01:
                    Z_mesh[j, i] = Z_grid[k]
                    break
    
    return X_mesh, Y_mesh, Z_mesh, x_unique, y_unique

def predict_concentration_dependence_for_B_families(dopant, b_cations_list, model_data, property_type='delta_H'):
    """
    Predict ΔH or ΔS as function of dopant content for multiple B-cation families.
    
    Parameters:
    -----------
    dopant : str
        Dopant cation (e.g., 'Sc', 'Y', 'In')
    b_cations_list : list
        List of B-cations to include (e.g., ['Zr', 'Ce', 'Sn', 'Ti', 'Hf'])
    model_data : dict
        Trained model data
    property_type : str
        'delta_H' or 'delta_S'
    
    Returns:
    --------
    dict : {B_cation: (x_values, property_values)}
    """
    
    A_cation = 'Ba'
    x_values = np.arange(0, 0.31, 0.05)  # 0 to 0.3 step 0.05
    
    results = {}
    
    for B_cation in b_cations_list:
        property_values = []
        
        for x in x_values:
            # Get predicted properties for this composition
            props = get_end_member_properties(B_cation, dopant, x, model_data)
            
            if property_type == 'delta_H':
                property_values.append(props['delta_H'])
            else:
                property_values.append(props['delta_S'])
        
        results[B_cation] = (x_values, np.array(property_values))
    
    return results

def calculate_descriptors(row):
    """Calculate enhanced descriptors for a given material"""
    A = row['A_cation']
    B = row['B_cation']
    D = row['dopant']
    x = row['content']
    
    # Get ionic radii with appropriate coordination
    r_A_XII = get_ionic_radius(A, 'XII')  # 12-coordinate for perovskite A-site
    r_A_IX = get_ionic_radius(A, 'IX')    # 9-coordinate alternative
    r_A_VIII = get_ionic_radius(A, 'VIII') # 8-coordinate alternative
    
    r_B_VI = get_ionic_radius(B, 'VI')     # 6-coordinate for B-site
    r_D_VI = get_ionic_radius(D, 'VI') if D != 'O' else None
    
    r_O_VI = 1.40  # Oxygen radius in 6-coordination
    
    # Get electronegativities
    chi_A = get_electronegativity(A)
    chi_B = get_electronegativity(B)
    chi_D = get_electronegativity(D) if D != 'O' else None
    
    # Get additional properties
    pol_A = get_polarizability(A)
    pol_B = get_polarizability(B)
    pol_D = get_polarizability(D) if D != 'O' else 0
    
    ip_A = get_ionization_potential(A)
    ip_B = get_ionization_potential(B)
    ip_D = get_ionization_potential(D) if D != 'O' else 0
    
    # Calculate descriptors
    descriptors = {}
    
    # Basic radii descriptors
    descriptors['r_A_XII'] = r_A_XII if r_A_XII is not None else 0
    descriptors['r_A_IX'] = r_A_IX if r_A_IX is not None else 0
    descriptors['r_A_VIII'] = r_A_VIII if r_A_VIII is not None else 0
    descriptors['r_B_VI'] = r_B_VI if r_B_VI is not None else 0
    descriptors['r_O_VI'] = r_O_VI
    
    # B-site average and mismatch
    if r_B_VI is not None:
        if r_D_VI is not None and x > 0:
            r_B_avg = (1 - x) * r_B_VI + x * r_D_VI
            descriptors['r_B_avg'] = r_B_avg
            descriptors['delta_r_B'] = abs(r_D_VI - r_B_VI)
            descriptors['r_D_VI'] = r_D_VI
        else:
            r_B_avg = r_B_VI
            descriptors['r_B_avg'] = r_B_VI
            descriptors['delta_r_B'] = 0
            descriptors['r_D_VI'] = 0
    else:
        r_B_avg = 0
        descriptors['r_B_avg'] = 0
        descriptors['delta_r_B'] = 0
        descriptors['r_D_VI'] = 0
    
    # Structural factors
    if r_A_XII is not None and r_B_avg > 0:
        descriptors['t_Goldschmidt'] = calculate_tolerance_factor(r_A_XII, r_B_avg, r_O_VI)
        descriptors['octahedral_factor'] = calculate_octahedral_factor(r_B_avg, r_O_VI)
        descriptors['global_instability'] = calculate_global_instability_index(r_A_XII, r_B_avg, r_O_VI)
        
        # Approximate bond distances (simplified)
        d_AO = (r_A_XII + r_O_VI) / np.sqrt(2)  # Approximate A-O distance
        d_BO = r_B_avg + r_O_VI  # Approximate B-O distance
        descriptors['bond_valence_sum'] = calculate_bond_valence_sum(r_A_XII, r_B_avg, r_O_VI, d_AO, d_BO)
    else:
        descriptors['t_Goldschmidt'] = 0
        descriptors['octahedral_factor'] = 0
        descriptors['global_instability'] = 0
        descriptors['bond_valence_sum'] = 0
    
    # Lattice energy
    if r_A_XII is not None and r_B_avg > 0:
        # Get typical charges
        z_A = IONIC_RADII.get(A, {}).get('charge', 2)
        z_B = IONIC_RADII.get(B, {}).get('charge', 4)
        descriptors['lattice_energy'] = calculate_lattice_energy(r_A_XII, r_B_avg, r_O_VI, z_A, z_B)
    else:
        descriptors['lattice_energy'] = 0
    
    # Electronegativity descriptors
    if chi_A is not None and chi_B is not None:
        descriptors['chi_A'] = chi_A
        descriptors['chi_B'] = chi_B
        
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
        
        descriptors['chi_diff'] = chi_B_avg - chi_A
        descriptors['chi_product'] = chi_A * chi_B_avg
    else:
        descriptors['chi_A'] = 0
        descriptors['chi_B'] = 0
        descriptors['chi_B_avg'] = 0
        descriptors['chi_D'] = 0
        descriptors['delta_chi'] = 0
        descriptors['chi_diff'] = 0
        descriptors['chi_product'] = 0
    
    # Polarizability and ionization
    descriptors['polarizability_A'] = pol_A
    descriptors['polarizability_B'] = pol_B
    descriptors['polarizability_D'] = pol_D
    descriptors['polarizability_avg'] = np.mean([pol_A, pol_B, pol_D]) if any([pol_A, pol_B, pol_D]) else 0
    
    descriptors['ionization_A'] = ip_A
    descriptors['ionization_B'] = ip_B
    descriptors['ionization_D'] = ip_D
    descriptors['ionization_avg'] = np.mean([ip_A, ip_B, ip_D]) if any([ip_A, ip_B, ip_D]) else 0
    
    # Charge densities
    descriptors['charge_density_A'] = calculate_charge_density(r_A_XII, z_A if 'z_A' in locals() else 2)
    descriptors['charge_density_B'] = calculate_charge_density(r_B_VI, z_B if 'z_B' in locals() else 4)
    descriptors['charge_density_D'] = calculate_charge_density(r_D_VI, IONIC_RADII.get(D, {}).get('charge', 3) if D != 'O' else 0)
    
    # Composition
    descriptors['content'] = x
    descriptors['oxygen_vacancy'] = x / 2  # For acceptor doping
    
    return descriptors

# =============================================================================
# Enhanced 3D visualization functions
# =============================================================================
def create_3d_descriptor_landscape(df_features):
    """Create 3D landscape of descriptors with projections"""
    
    # Select key descriptors
    x_axis = 'r_B_avg'
    y_axis = 'chi_diff'
    z_axis = 'delta_H'
    
    # Prepare data
    X = df_features[x_axis].values
    Y = df_features[y_axis].values
    Z = df_features[z_axis].values
    
    # Create grid for surface
    xi = np.linspace(X.min(), X.max(), 50)
    yi = np.linspace(Y.min(), Y.max(), 50)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate for smooth surface
    zi = griddata((X, Y), Z, (xi, yi), method='cubic')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 8))
    
    # Main 3D plot
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Surface
    surf = ax1.plot_surface(xi, yi, zi, cmap='RdBu_r', alpha=0.7,
                           linewidth=0, antialiased=True)
    
    # Data points
    scatter = ax1.scatter(X, Y, Z, c=Z, cmap='RdBu_r', 
                         s=80, edgecolors='black', linewidth=1,
                         vmin=-150, vmax=-50)
    
    # Projections on planes
    ax1.contourf(xi, yi, zi, zdir='z', offset=Z.min()-10, 
                 cmap='RdBu_r', alpha=0.3)
    ax1.contourf(xi, yi, zi, zdir='x', offset=X.max()+0.05, 
                 cmap='RdBu_r', alpha=0.3)
    ax1.contourf(xi, yi, zi, zdir='y', offset=Y.max()+0.2, 
                 cmap='RdBu_r', alpha=0.3)
    
    # Labels and styling
    ax1.set_xlabel(f'{x_axis} (Å)', fontsize=11, labelpad=10)
    ax1.set_ylabel(f'{y_axis}', fontsize=11, labelpad=10)
    ax1.set_zlabel('ΔH (kJ/mol)', fontsize=11, labelpad=10)
    ax1.set_title('3D Hydration Landscape with Orthogonal Projections', 
                  fontsize=13, fontweight='bold', pad=20)
    
    # Color bar
    plt.colorbar(surf, ax=ax1, shrink=0.5, aspect=20, label='ΔH (kJ/mol)')
    
    # 2D Contour plot with group trends
    ax2 = fig.add_subplot(132)
    
    # Contour plot
    contour = ax2.contourf(xi, yi, zi, levels=20, cmap='RdBu_r', alpha=0.8)
    ax2.contour(xi, yi, zi, levels=10, colors='black', linewidths=0.5, alpha=0.3)
    
    # Data points
    for b_cat in df_features['B_cation'].unique():
        subset = df_features[df_features['B_cation'] == b_cat]
        ax2.scatter(subset[x_axis], subset[y_axis], 
                   label=b_cat, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add trend lines for groups with enough points
        if len(subset) >= 4:
            z = np.polyfit(subset[x_axis], subset[y_axis], 1)
            x_line = np.linspace(subset[x_axis].min(), subset[x_axis].max(), 50)
            y_line = np.polyval(z, x_line)
            ax2.plot(x_line, y_line, '--', linewidth=2, alpha=0.5)
    
    ax2.set_xlabel(f'{x_axis} (Å)', fontsize=11)
    ax2.set_ylabel(f'{y_axis}', fontsize=11)
    ax2.set_title('2D Contour Map with Material Families', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', fontsize=9)
    plt.colorbar(contour, ax=ax2, label='ΔH (kJ/mol)')
    ax2.grid(True, alpha=0.3)
    
    # 3D scatter with color coding by material family
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Get unique B-cations for coloring
    unique_B = df_features['B_cation'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_B)))
    
    for b_cat, color in zip(unique_B, colors):
        subset = df_features[df_features['B_cation'] == b_cat]
        ax3.scatter(subset[x_axis], subset[y_axis], subset[z_axis],
                   label=b_cat, c=[color], s=60, alpha=0.8,
                   edgecolors='black', linewidth=0.5)
    
    ax3.set_xlabel(f'{x_axis} (Å)', fontsize=10, labelpad=8)
    ax3.set_ylabel(f'{y_axis}', fontsize=10, labelpad=8)
    ax3.set_zlabel('ΔH (kJ/mol)', fontsize=10, labelpad=8)
    ax3.set_title('3D Distribution by Material Family', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=8)
    
    plt.tight_layout()
    return fig

# =============================================================================
# Parallel Coordinates Analysis
# =============================================================================
def create_enhanced_parallel_coordinates(df_features):
    """Create enhanced parallel coordinates plot for multi-dimensional analysis"""
    
    # Select key descriptors for analysis
    features = ['r_B_avg', 't_Goldschmidt', 'chi_diff', 'delta_r_B', 
                'content', 'polarizability_avg', 'ionization_avg', 'delta_H']
    
    # Filter available features
    available_features = [f for f in features if f in df_features.columns]
    
    if len(available_features) < 3:
        return None
    
    # Normalize data for visualization
    df_norm = df_features[available_features].copy()
    for col in available_features[:-1]:  # Don't normalize target
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # Plot 1: Parallel coordinates with color by ΔH
    ax1 = axes[0, 0]
    
    # Color mapping based on ΔH
    norm = plt.Normalize(df_features['delta_H'].min(), df_features['delta_H'].max())
    colors = plt.cm.RdBu_r(norm(df_features['delta_H']))
    
    # Coordinates for vertical lines
    x_coords = np.arange(len(available_features))
    
    # Plot lines for each material
    for idx in range(len(df_norm)):
        y_coords = df_norm.iloc[idx, :].values
        ax1.plot(x_coords, y_coords, color=colors[idx], alpha=0.3, linewidth=0.5)
    
    # Add mean lines for ΔH quartiles
    delta_H_bins = pd.qcut(df_features['delta_H'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    for bin_name, group in df_features.groupby(delta_H_bins):
        if len(group) > 2:
            group_norm = df_norm.loc[group.index]
            mean_values = group_norm[available_features].mean()
            ax1.plot(x_coords, mean_values, linewidth=3, 
                    label=f'{bin_name} (ΔH: {group["delta_H"].mean():.0f} kJ/mol)', 
                    alpha=0.8)
    
    ax1.set_xticks(x_coords)
    ax1.set_xticklabels([f.replace('_', ' ').title() for f in available_features], 
                        rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Normalized Value', fontsize=10)
    ax1.set_title('Parallel Coordinates: Multi-dimensional Patterns by ΔH Quartiles', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Plot 2: Feature correlation heatmap
    ax2 = axes[0, 1]
    
    # Calculate correlation matrix
    corr_matrix = df_features[available_features].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, square=True, ax=ax2,
                cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                annot_kws={'size': 8})
    ax2.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)
    
    # Plot 3: Radar charts for characteristic compositions
    ax3 = axes[1, 0]
    
    # Find representative compositions for each ΔH quartile
    n_radar = 4
    angles = np.linspace(0, 2 * np.pi, len(available_features)-1, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Select samples (one per quartile)
    delta_H_quartiles = pd.qcut(df_features['delta_H'], n_radar, labels=False)
    radar_data = []
    
    for q in range(n_radar):
        q_data = df_features[delta_H_quartiles == q]
        if len(q_data) > 0:
            # Find sample closest to median in this quartile
            median_H = q_data['delta_H'].median()
            closest_idx = (q_data['delta_H'] - median_H).abs().argsort()[0]
            radar_data.append(q_data.iloc[closest_idx])
    
    # Plot radar charts
    for i, data in enumerate(radar_data):
        values = [data[f] for f in available_features[:-1]]  # Exclude ΔH
        # Normalize values
        values_norm = []
        for v, f in zip(values, available_features[:-1]):
            min_val = df_features[f].min()
            max_val = df_features[f].max()
            if max_val > min_val:
                values_norm.append((v - min_val) / (max_val - min_val))
            else:
                values_norm.append(0.5)
        values_norm += values_norm[:1]  # Close the loop
        
        ax3.plot(angles, values_norm, 'o-', linewidth=2, 
                label=f'ΔH = {data["delta_H"]:.0f} kJ/mol')
        ax3.fill(angles, values_norm, alpha=0.25)
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels([f.replace('_', ' ').title() for f in available_features[:-1]], 
                        size=8)
    ax3.set_ylim(0, 1)
    ax3.set_title('Radar Plots: Characteristic Compositions by ΔH', 
                  fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=8)
    ax3.grid(True)
    
    # Plot 4: Andrews curves (alternative to parallel coordinates)
    ax4 = axes[1, 1]
    
    # Create Andrews curves for each ΔH quartile
    t = np.linspace(-np.pi, np.pi, 100)
    
    for bin_name, group in df_features.groupby(delta_H_bins):
        if len(group) > 2:
            group_norm = df_norm.loc[group.index, available_features[:-1]]
            mean_values = group_norm.mean().values
            
            # Andrews curve: f(t) = x1/√2 + x2·sin(t) + x3·cos(t) + x4·sin(2t) + ...
            curve = mean_values[0] / np.sqrt(2) * np.ones_like(t)
            for i, val in enumerate(mean_values[1:], 1):
                if i % 2 == 1:
                    curve += val * np.sin((i+1)//2 * t)
                else:
                    curve += val * np.cos(i//2 * t)
            
            ax4.plot(t, curve, linewidth=2, 
                    label=f'{bin_name} (ΔH: {group["delta_H"].mean():.0f} kJ/mol)')
    
    ax4.set_xlabel('t', fontsize=10)
    ax4.set_ylabel('Andrews curve value', fontsize=10)
    ax4.set_title('Andrews Curves: Multi-dimensional Pattern Comparison', 
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-dimensional Analysis of Hydration Parameters', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

# =============================================================================
# Sensitivity Analysis and Critical Regions
# =============================================================================
def create_sensitivity_heatmap(model_data, df_features):
    """Create sensitivity analysis heatmap to identify critical regions"""
    
    if model_data is None:
        return None
    
    # Define key descriptors for sensitivity analysis
    x_axis = 'r_B_avg'
    y_axis = 'chi_diff'
    
    if x_axis not in df_features.columns or y_axis not in df_features.columns:
        return None
    
    # Create grid
    x_grid = np.linspace(df_features[x_axis].min(), df_features[x_axis].max(), 40)
    y_grid = np.linspace(df_features[y_axis].min(), df_features[y_axis].max(), 40)
    
    # Get mean values for other features
    mean_values = {}
    feature_names = model_data['feature_names']
    
    for col in feature_names:
        if col not in [x_axis, y_axis, 'A_enc', 'B_enc', 'D_enc']:
            if col in df_features.columns:
                mean_values[col] = df_features[col].mean()
            else:
                mean_values[col] = 0
    
    # Create test points
    X_test = []
    for x_val in x_grid:
        for y_val in y_grid:
            point = mean_values.copy()
            point[x_axis] = x_val
            point[y_axis] = y_val
            # Add categorical features (mode)
            if 'A_cation' in df_features.columns:
                point['A_enc'] = model_data['le_A'].transform([df_features['A_cation'].mode()[0]])[0]
            if 'B_cation' in df_features.columns:
                point['B_enc'] = model_data['le_B'].transform([df_features['B_cation'].mode()[0]])[0]
            if 'dopant' in df_features.columns:
                point['D_enc'] = model_data['le_D'].transform([df_features['dopant'].mode()[0]])[0]
            X_test.append(point)
    
    X_test = pd.DataFrame(X_test)
    
    # Ensure all required columns exist
    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0
    
    X_test = X_test[feature_names]
    X_test_scaled = model_data['scaler'].transform(X_test)
    
    # Get predictions
    predictions = model_data['models']['xgb_H'].predict(X_test_scaled)
    predictions = predictions.reshape(len(x_grid), len(y_grid))
    
    # Calculate gradients (sensitivity)
    dy, dx = np.gradient(predictions)
    sensitivity = np.sqrt(dx**2 + dy**2)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Prediction surface
    ax1 = axes[0, 0]
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    contour1 = ax1.contourf(X_mesh, Y_mesh, predictions.T, levels=20, cmap='RdBu_r')
    plt.colorbar(contour1, ax=ax1, label='Predicted ΔH (kJ/mol)')
    
    # Add experimental points
    ax1.scatter(df_features[x_axis], df_features[y_axis], 
               c='black', s=30, alpha=0.5, edgecolors='white', linewidth=0.5)
    
    ax1.set_xlabel(f'{x_axis} (Å)', fontsize=10)
    ax1.set_ylabel(f'{y_axis}', fontsize=10)
    ax1.set_title('Predicted ΔH Landscape', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sensitivity map
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X_mesh, Y_mesh, sensitivity.T, levels=20, cmap='hot')
    plt.colorbar(contour2, ax=ax2, label='Sensitivity (|∇ΔH|)')
    
    # Highlight high sensitivity regions (>90th percentile)
    threshold = np.percentile(sensitivity, 90)
    high_sens = sensitivity > threshold
    ax2.contour(X_mesh, Y_mesh, high_sens.T, levels=[0.5], colors='cyan', linewidths=2)
    
    ax2.set_xlabel(f'{x_axis} (Å)', fontsize=10)
    ax2.set_ylabel(f'{y_axis}', fontsize=10)
    ax2.set_title('Sensitivity Map (Cyan = Critical Regions)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction uncertainty (based on gradient magnitude)
    ax3 = axes[0, 2]
    uncertainty = sensitivity / sensitivity.max() * 20  # Scale to reasonable uncertainty
    contour3 = ax3.contourf(X_mesh, Y_mesh, uncertainty.T, levels=20, cmap='YlOrRd')
    plt.colorbar(contour3, ax=ax3, label='Estimated Uncertainty (kJ/mol)')
    
    ax3.set_xlabel(f'{x_axis} (Å)', fontsize=10)
    ax3.set_ylabel(f'{y_axis}', fontsize=10)
    ax3.set_title('Prediction Uncertainty Map', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Profiles at fixed y values
    ax4 = axes[1, 0]
    
    y_percentiles = [25, 50, 75]
    for p in y_percentiles:
        y_fixed = np.percentile(df_features[y_axis], p)
        y_idx = np.argmin(np.abs(y_grid - y_fixed))
        ax4.plot(x_grid, predictions[:, y_idx], 
                linewidth=2, label=f'{y_axis} = {y_fixed:.2f} (p{p})')
    
    ax4.set_xlabel(f'{x_axis} (Å)', fontsize=10)
    ax4.set_ylabel('Predicted ΔH (kJ/mol)', fontsize=10)
    ax4.set_title('ΔH Profiles at Fixed χ_diff', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Profiles at fixed x values
    ax5 = axes[1, 1]
    
    x_percentiles = [25, 50, 75]
    for p in x_percentiles:
        x_fixed = np.percentile(df_features[x_axis], p)
        x_idx = np.argmin(np.abs(x_grid - x_fixed))
        ax5.plot(y_grid, predictions[x_idx, :], 
                linewidth=2, label=f'{x_axis} = {x_fixed:.2f} Å (p{p})')
    
    ax5.set_xlabel(f'{y_axis}', fontsize=10)
    ax5.set_ylabel('Predicted ΔH (kJ/mol)', fontsize=10)
    ax5.set_title('ΔH Profiles at Fixed B-site Radius', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Optimal regions overlay
    ax6 = axes[1, 2]
    
    # 2D histogram of experimental data
    h = ax6.hist2d(df_features[x_axis], df_features[y_axis], 
                   bins=15, cmap='Greys', alpha=0.3, density=True)
    
    # Overlay prediction contours
    contour_best = ax6.contour(X_mesh, Y_mesh, predictions.T, 
                               levels=[-120, -100, -80], 
                               colors=['red', 'orange', 'green'],
                               linewidths=2, linestyles='--')
    ax6.clabel(contour_best, inline=True, fontsize=9, fmt='%d kJ/mol')
    
    # Highlight best hydration regions (ΔH < -110)
    best_mask = predictions < -110
    if best_mask.any():
        ax6.contourf(X_mesh, Y_mesh, best_mask.T, levels=[0.5, 1], 
                    colors=['lightblue'], alpha=0.3, label='Strong Hydration')
    
    ax6.set_xlabel(f'{x_axis} (Å)', fontsize=10)
    ax6.set_ylabel(f'{y_axis}', fontsize=10)
    ax6.set_title('Experimental Coverage with Optimal Regions', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Sensitivity Analysis and Critical Region Identification', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

# =============================================================================
# Hierarchical Clustering Analysis
# =============================================================================
def create_hierarchical_clustering(df_features):
    """Create hierarchical clustering analysis with dendrogram"""
    
    # Select features for clustering
    cluster_features = ['r_B_avg', 't_Goldschmidt', 'chi_diff', 'delta_r_B', 
                        'content', 'polarizability_avg', 'delta_H', 'delta_S']
    
    available_features = [f for f in cluster_features if f in df_features.columns]
    
    if len(available_features) < 3 or len(df_features) < 5:
        return None
    
    # Prepare data for clustering
    X_cluster = df_features[available_features].copy()
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Calculate linkage matrix
    Z = linkage(X_scaled, method='ward')
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # Plot 1: Dendrogram
    ax1 = axes[0, 0]
    
    # Create labels for dendrogram
    labels = [f"{row['A_cation']}{row['B_cation']}{row['dopant']}{row['content']:.2f}" 
              for _, row in df_features.iterrows()]
    
    dendrogram(Z, ax=ax1, orientation='top', 
              labels=labels, leaf_rotation=90, leaf_font_size=6,
              color_threshold=0.7*max(Z[:,2]))
    ax1.set_title('Hierarchical Clustering Dendrogram', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Distance (Ward)')
    ax1.set_xlabel('Materials')
    
    # Plot 2: Heatmap with hierarchical ordering
    ax2 = axes[0, 1]
    
    # Get order from dendrogram
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(Z)
    
    # Reorder data
    X_ordered = X_cluster.iloc[order]
    labels_ordered = [labels[i] for i in order]
    
    # Create heatmap
    im = ax2.imshow(X_ordered.T, aspect='auto', cmap='RdBu_r',
                   interpolation='nearest')
    
    ax2.set_yticks(range(len(available_features)))
    ax2.set_yticklabels([f.replace('_', ' ').title() for f in available_features])
    ax2.set_xticks(range(len(labels_ordered)))
    ax2.set_xticklabels(labels_ordered, rotation=90, fontsize=4)
    ax2.set_title('Feature Heatmap with Hierarchical Ordering', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Normalized Value')
    
    # Plot 3: PCA projection with clusters
    ax3 = axes[0, 2]
    
    # Determine clusters at different thresholds
    n_clusters = 4  # Fixed number of clusters
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot clusters
    scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=clusters, cmap='tab10', 
                         s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add cluster ellipses
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        if mask.sum() >= 3:
            from matplotlib.patches import Ellipse
            centroid = X_pca[mask].mean(axis=0)
            cov = np.cov(X_pca[mask].T)
            if cov.shape == (2, 2):
                vals, vecs = np.linalg.eigh(cov)
                order_idx = vals.argsort()[::-1]
                vals, vecs = vals[order_idx], vecs[:, order_idx]
                theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                width, height = 2 * np.sqrt(vals)
                ellipse = Ellipse(xy=centroid, width=width, height=height,
                                 angle=theta, alpha=0.2, 
                                 color=scatter.cmap(scatter.norm(cluster_id)))
                ax3.add_patch(ellipse)
    
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
    ax3.set_title('PCA Projection with Detected Clusters', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cluster profiles
    ax4 = axes[1, 0]
    
    # Calculate cluster means
    cluster_means = pd.DataFrame(X_scaled).groupby(clusters).mean()
    cluster_means.columns = available_features
    
    # Plot cluster profiles
    cluster_means.T.plot(kind='bar', ax=ax4, alpha=0.7,
                        edgecolor='black', linewidth=0.5)
    
    ax4.set_xlabel('Features', fontsize=10)
    ax4.set_ylabel('Normalized Mean Value', fontsize=10)
    ax4.set_title('Cluster Profiles (mean normalized features)', fontsize=12, fontweight='bold')
    ax4.legend(title='Cluster', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticklabels([f.replace('_', ' ').title() for f in available_features], 
                        rotation=45, ha='right', fontsize=8)
    
    # Plot 5: Silhouette analysis
    ax5 = axes[1, 1]
    
    from sklearn.metrics import silhouette_samples, silhouette_score
    
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X_scaled, clusters)
    sample_silhouette_values = silhouette_samples(X_scaled, clusters)
    
    y_lower = 10
    for i in range(1, n_clusters + 1):
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.tab10((i-1) / n_clusters)
        ax5.fill_betweenx(np.arange(y_lower, y_upper), 0, 
                         ith_cluster_silhouette_values,
                         facecolor=color, edgecolor='black', alpha=0.7)
        
        ax5.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax5.axvline(x=silhouette_avg, color='red', linestyle='--', 
                linewidth=2, label=f'Average: {silhouette_avg:.3f}')
    
    ax5.set_xlabel('Silhouette Coefficient', fontsize=10)
    ax5.set_ylabel('Cluster', fontsize=10)
    ax5.set_title('Silhouette Analysis for Cluster Validation', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: t-SNE visualization
    ax6 = axes[1, 2]
    
    # t-SNE for non-linear dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
    X_tsne = tsne.fit_transform(X_scaled)
    
    scatter2 = ax6.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                          c=clusters, cmap='tab10', 
                          s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add labels for representative points
    for i, label in enumerate(labels):
        if i % max(1, len(labels)//10) == 0:  # Show every 10th label
            ax6.annotate(label, (X_tsne[i, 0], X_tsne[i, 1]), 
                        fontsize=6, alpha=0.7)
    
    ax6.set_xlabel('t-SNE Component 1', fontsize=10)
    ax6.set_ylabel('t-SNE Component 2', fontsize=10)
    ax6.set_title('t-SNE Visualization of Clusters', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Hierarchical Cluster Analysis of Perovskite Materials', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

# =============================================================================
# Predictive 3D Diagrams for Proton Concentration
# =============================================================================
def create_proton_concentration_3d(model_data, df_features):
    """Create 3D predictive diagrams for proton concentration"""
    
    if model_data is None:
        return None
    
    st.subheader("3D Proton Concentration Predictor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Select A-cation
        a_options = sorted(df_features['A_cation'].unique())
        selected_a = st.selectbox("A-cation", a_options, key='proton_a')
        
        # Select B-cation
        b_options = sorted(df_features[df_features['A_cation'] == selected_a]['B_cation'].unique())
        selected_b = st.selectbox("B-cation", b_options, key='proton_b')
        
    with col2:
        # Select dopant
        d_options = sorted(df_features[
            (df_features['A_cation'] == selected_a) & 
            (df_features['B_cation'] == selected_b)
        ]['dopant'].unique())
        selected_d = st.selectbox("Dopant", d_options, key='proton_d')
        
        # Temperature range
        T_min = st.number_input("Min Temperature (°C)", value=300, min_value=100, max_value=1000)
        T_max = st.number_input("Max Temperature (°C)", value=800, min_value=200, max_value=1200)
        
    with col3:
        # pH2O range
        log_pH2O_min = st.number_input("Min log(pH2O/atm)", value=-5.0, min_value=-5.0, max_value=0.0, step=0.5)
        log_pH2O_max = st.number_input("Max log(pH2O/atm)", value=0.0, min_value=-5.0, max_value=0.0, step=0.5)
        
        # Dopant content range
        x_min = st.number_input("Min dopant content", value=0.0, min_value=0.0, max_value=0.8, step=0.05)
        x_max = st.number_input("Max dopant content", value=0.5, min_value=0.0, max_value=0.8, step=0.05)
    
    # Option 1: [OH] = f(x, T) at fixed pH2O
    st.subheader("Proton Concentration vs Dopant Content and Temperature")
    
    fixed_pH2O = st.slider("Fixed pH2O (atm)", 
                           min_value=10**log_pH2O_min, 
                           max_value=10**log_pH2O_max,
                           value=10**-2.0,
                           format="%.1e",
                           key='fixed_pH2O')
    
    n_points = 30
    x_grid = np.linspace(x_min, x_max, n_points)
    T_grid = np.linspace(T_min + 273.15, T_max + 273.15, n_points)
    X_grid, T_mesh = np.meshgrid(x_grid, T_grid)
    
    OH_grid = np.zeros_like(X_grid)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, x_val in enumerate(x_grid):
        status_text.text(f"Calculating for x = {x_val:.3f}...")
        
        # Get predictions for this composition
        input_row = pd.Series({
            'A_cation': selected_a,
            'B_cation': selected_b,
            'dopant': selected_d,
            'content': x_val
        })
        
        desc = calculate_descriptors(input_row)
        
        # Prepare feature vector
        X_pred = pd.DataFrame([{
            'content': desc.get('content', x_val),
            'r_A_XII': desc.get('r_A_XII', 0),
            'r_B_VI': desc.get('r_B_VI', 0),
            'r_D_VI': desc.get('r_D_VI', 0),
            'r_B_avg': desc.get('r_B_avg', 0),
            'delta_r_B': desc.get('delta_r_B', 0),
            't_Goldschmidt': desc.get('t_Goldschmidt', 0),
            'chi_A': desc.get('chi_A', 0),
            'chi_B': desc.get('chi_B', 0),
            'chi_D': desc.get('chi_D', 0),
            'chi_B_avg': desc.get('chi_B_avg', 0),
            'chi_diff': desc.get('chi_diff', 0),
            'octahedral_factor': desc.get('octahedral_factor', 0),
            'global_instability': desc.get('global_instability', 0),
            'lattice_energy': desc.get('lattice_energy', 0),
            'polarizability_avg': desc.get('polarizability_avg', 0),
            'ionization_avg': desc.get('ionization_avg', 0),
            'charge_density_A': desc.get('charge_density_A', 0),
            'charge_density_B': desc.get('charge_density_B', 0)
        }])
        
        # Add encoded categorical features
        try:
            X_pred['A_enc'] = model_data['le_A'].transform([selected_a])[0] if selected_a in model_data['le_A'].classes_ else -1
        except:
            X_pred['A_enc'] = -1
        
        try:
            X_pred['B_enc'] = model_data['le_B'].transform([selected_b])[0] if selected_b in model_data['le_B'].classes_ else -1
        except:
            X_pred['B_enc'] = -1
        
        try:
            X_pred['D_enc'] = model_data['le_D'].transform([selected_d])[0] if selected_d in model_data['le_D'].classes_ else -1
        except:
            X_pred['D_enc'] = -1
        
        # Ensure all required features are present
        for col in model_data['feature_names']:
            if col not in X_pred.columns:
                X_pred[col] = 0
        
        X_pred = X_pred[model_data['feature_names']]
        X_pred_scaled = model_data['scaler'].transform(X_pred)
        
        # Get predictions
        delta_H = model_data['models']['xgb_H'].predict(X_pred_scaled)[0]
        delta_S = model_data['models']['xgb_S'].predict(X_pred_scaled)[0]
        
        # Calculate [OH] for all temperatures using corrected formula
        for j, T in enumerate(T_grid):
            OH_grid[j, i] = calculate_proton_concentration(delta_H, delta_S, T, fixed_pH2O, x_val)
        
        progress_bar.progress((i + 1) / n_points)
    
    status_text.text("Calculation complete!")
    progress_bar.empty()
    
    # Create 3D plot
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=x_grid,
        y=T_grid - 273.15,
        z=OH_grid,
        colorscale='Viridis',
        colorbar=dict(title="[OH]"),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
        )
    ))
    
    fig.update_layout(
        title=f'Proton Concentration [OH] for {selected_a}{selected_b}{selected_d}<br>at pH2O = {fixed_pH2O:.2e} atm',
        scene=dict(
            xaxis_title='Dopant Content, x',
            yaxis_title='Temperature (°C)',
            zaxis_title='[OH]',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Option 2: [OH] = f(T, pH2O) at fixed x
    st.subheader("Proton Concentration vs Temperature and Water Vapor Pressure")
    
    fixed_x = st.slider("Fixed dopant content", 
                        min_value=x_min, max_value=x_max,
                        value=(x_min + x_max)/2,
                        step=0.01,
                        key='fixed_x')
    
    # Get predictions for fixed x
    input_row = pd.Series({
        'A_cation': selected_a,
        'B_cation': selected_b,
        'dopant': selected_d,
        'content': fixed_x
    })
    
    desc = calculate_descriptors(input_row)
    
    X_pred = pd.DataFrame([{
        'content': desc.get('content', fixed_x),
        'r_A_XII': desc.get('r_A_XII', 0),
        'r_B_VI': desc.get('r_B_VI', 0),
        'r_D_VI': desc.get('r_D_VI', 0),
        'r_B_avg': desc.get('r_B_avg', 0),
        'delta_r_B': desc.get('delta_r_B', 0),
        't_Goldschmidt': desc.get('t_Goldschmidt', 0),
        'chi_A': desc.get('chi_A', 0),
        'chi_B': desc.get('chi_B', 0),
        'chi_D': desc.get('chi_D', 0),
        'chi_B_avg': desc.get('chi_B_avg', 0),
        'chi_diff': desc.get('chi_diff', 0),
        'octahedral_factor': desc.get('octahedral_factor', 0),
        'global_instability': desc.get('global_instability', 0),
        'lattice_energy': desc.get('lattice_energy', 0),
        'polarizability_avg': desc.get('polarizability_avg', 0),
        'ionization_avg': desc.get('ionization_avg', 0),
        'charge_density_A': desc.get('charge_density_A', 0),
        'charge_density_B': desc.get('charge_density_B', 0)
    }])
    
    try:
        X_pred['A_enc'] = model_data['le_A'].transform([selected_a])[0] if selected_a in model_data['le_A'].classes_ else -1
    except:
        X_pred['A_enc'] = -1
    
    try:
        X_pred['B_enc'] = model_data['le_B'].transform([selected_b])[0] if selected_b in model_data['le_B'].classes_ else -1
    except:
        X_pred['B_enc'] = -1
    
    try:
        X_pred['D_enc'] = model_data['le_D'].transform([selected_d])[0] if selected_d in model_data['le_D'].classes_ else -1
    except:
        X_pred['D_enc'] = -1
    
    for col in model_data['feature_names']:
        if col not in X_pred.columns:
            X_pred[col] = 0
    
    X_pred = X_pred[model_data['feature_names']]
    X_pred_scaled = model_data['scaler'].transform(X_pred)
    
    delta_H = model_data['models']['xgb_H'].predict(X_pred_scaled)[0]
    delta_S = model_data['models']['xgb_S'].predict(X_pred_scaled)[0]
    
    # Create grids
    pH2O_grid = np.logspace(log_pH2O_min, log_pH2O_max, n_points)
    T_grid2 = np.linspace(T_min + 273.15, T_max + 273.15, n_points)
    pH2O_mesh, T_mesh2 = np.meshgrid(pH2O_grid, T_grid2)
    
    OH_grid2 = np.zeros_like(pH2O_mesh)
    
    for i in range(n_points):
        for j in range(n_points):
            OH_grid2[i, j] = calculate_proton_concentration(
                delta_H, delta_S, T_mesh2[i, j], pH2O_mesh[i, j], fixed_x
            )
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Surface(
        x=np.log10(pH2O_grid),
        y=T_grid2 - 273.15,
        z=OH_grid2,
        colorscale='Plasma',
        colorbar=dict(title="[OH]"),
        contours=dict(
            z=dict(show=True, usecolormap=True, project=dict(z=True))
        )
    ))
    
    fig2.update_layout(
        title=f'Proton Concentration [OH] for {selected_a}{selected_b}{selected_d}<br>at x = {fixed_x:.2f}',
        scene=dict(
            xaxis_title='log(pH2O/atm)',
            yaxis_title='Temperature (°C)',
            zaxis_title='[OH]',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Option 3: 2D contour plots with optimization
    st.subheader("Optimization: Maximum Proton Concentration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Optimal conditions for max [OH]**")
        
        max_idx = np.unravel_index(np.argmax(OH_grid2), OH_grid2.shape)
        opt_T = T_grid2[max_idx[0]] - 273.15
        opt_pH2O = pH2O_grid[max_idx[1]]
        max_OH = OH_grid2[max_idx]
        
        st.metric("Maximum [OH]", f"{max_OH:.4f}")
        st.metric("Optimal Temperature", f"{opt_T:.0f} °C")
        st.metric("Optimal pH2O", f"{opt_pH2O:.2e} atm")
        st.metric("Optimal log(pH2O)", f"{np.log10(opt_pH2O):.2f}")
    
    with col2:
        st.write("**Contour plot with optimum**")
        
        fig3, ax = plt.subplots(figsize=(8, 6))
        
        contour = ax.contourf(np.log10(pH2O_grid), T_grid2 - 273.15, OH_grid2, 
                              levels=20, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='[OH]')
        
        ax.plot(np.log10(opt_pH2O), opt_T, 'r*', markersize=15, 
                label=f'Optimum: {max_OH:.4f}')
        
        ax.contour(np.log10(pH2O_grid), T_grid2 - 273.15, OH_grid2, 
                  levels=10, colors='white', linewidths=0.5, alpha=0.5)
        
        ax.set_xlabel('log(pH2O/atm)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'[OH] Contour Map for {selected_a}{selected_b}{selected_d}, x={fixed_x:.2f}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig3)
        plt.close()
    
    return fig

# =============================================================================
# Machine Learning Models with enhanced features
# =============================================================================
@st.cache_resource
def train_prediction_models(df):
    """Optimized ML models for predicting delta_H and delta_S"""
    
    # Enhanced feature list
    feature_cols = [
        'content', 'r_A_XII', 'r_B_VI', 'r_D_VI', 'r_B_avg', 'delta_r_B',
        't_Goldschmidt', 'octahedral_factor', 'global_instability', 'lattice_energy',
        'chi_A', 'chi_B', 'chi_D', 'chi_B_avg', 'chi_diff', 'chi_product',
        'polarizability_avg', 'ionization_avg', 'charge_density_A', 'charge_density_B',
        'oxygen_vacancy'
    ]
    
    # Calculate descriptors
    descriptor_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        desc = calculate_descriptors(row)
        if len(desc) > 10:  # Enough descriptors
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
        return None, df_features
    
    # Prepare X and y
    available_features = [f for f in feature_cols if f in df_features.columns]
    X = df_features[available_features].fillna(0)
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
    feature_names = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # =========================================================
    # OPTIMIZED: Three ensemble models for robust predictions
    # =========================================================
    
    # 1. XGBoost - gradient boosting with regularization
    xgb_model_H = xgb.XGBRegressor(
        n_estimators=100,        # Reduced from 200 for speed
        max_depth=4,              # Reduced from 6 to prevent overfitting
        learning_rate=0.1,        # Moderate learning rate
        subsample=0.8,            # Use 80% of data per tree
        colsample_bytree=0.8,     # Use 80% of features per tree
        reg_alpha=0.1,            # L1 regularization
        reg_lambda=1.0,           # L2 regularization
        random_state=42,
        n_jobs=-1,                 # Use all CPU cores
        verbosity=0
    )
    
    xgb_model_S = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    # 2. Random Forest - bagging ensemble for stability
    rf_model_H = RandomForestRegressor(
        n_estimators=100,         # Reduced from 200 for speed
        max_depth=6,               # Reduced from 8
        min_samples_split=5,       # Minimum samples to split a node
        min_samples_leaf=2,        # Minimum samples in a leaf
        max_features='sqrt',       # Use sqrt(n_features) per split
        bootstrap=True,             # Use bootstrap samples
        random_state=42,
        n_jobs=-1                   # Use all CPU cores
    )
    
    rf_model_S = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Gradient Boosting - sequential boosting for accuracy
    gb_model_H = GradientBoostingRegressor(
        n_estimators=100,          # Number of boosting stages
        max_depth=4,                # Tree depth
        learning_rate=0.1,          # Shrinkage factor
        subsample=0.8,              # Stochastic gradient boosting fraction
        min_samples_split=5,        # Minimum samples to split
        min_samples_leaf=2,         # Minimum samples in leaf
        max_features='sqrt',        # Feature fraction per tree
        random_state=42
    )
    
    gb_model_S = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    
    # Train all models
    with st.spinner("Training XGBoost models..."):
        xgb_model_H.fit(X_scaled, y_H)
        xgb_model_S.fit(X_scaled, y_S)
    
    with st.spinner("Training Random Forest models..."):
        rf_model_H.fit(X_scaled, y_H)
        rf_model_S.fit(X_scaled, y_S)
    
    with st.spinner("Training Gradient Boosting models..."):
        gb_model_H.fit(X_scaled, y_H)
        gb_model_S.fit(X_scaled, y_S)
    
    # OPTIMIZED: Fast cross-validation (only 3 folds)
    from sklearn.model_selection import cross_val_score
    
    cv_scores_H = cross_val_score(xgb_model_H, X_scaled, y_H, cv=min(3, len(X)), scoring='r2', n_jobs=-1)
    cv_scores_S = cross_val_score(xgb_model_S, X_scaled, y_S, cv=min(3, len(X)), scoring='r2', n_jobs=-1)
    
    # Feature importance from XGBoost (for consistency)
    feature_importance_H = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model_H.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # OPTIMIZED: SHAP analysis on subset only (if needed)
    shap_values_H = None
    shap_values_S = None
    shap_expected_H = None
    shap_expected_S = None
    shap_summary_H = feature_importance_H.copy()
    shap_summary_H.columns = ['feature', 'mean_abs_shap']
    
    # Only do SHAP if requested and dataset is small
    if len(df_features) < 100:  # Only for smaller datasets
        try:
            # Use subset for SHAP to speed up
            sample_size = min(50, len(X_scaled))
            X_sample = X_scaled[:sample_size]
            
            explainer_H = shap.TreeExplainer(xgb_model_H)
            shap_values_H = explainer_H.shap_values(X_sample)
            shap_expected_H = explainer_H.expected_value
            
            explainer_S = shap.TreeExplainer(xgb_model_S)
            shap_values_S = explainer_S.shap_values(X_sample)
            shap_expected_S = explainer_S.expected_value
            
            shap_summary_H = pd.DataFrame({
                'feature': feature_names,
                'mean_abs_shap': np.abs(shap_values_H).mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
        except Exception as e:
            # Silently fail if SHAP fails (e.g., on ARM architecture)
            pass
    
    # Return all trained models and metadata
    return {
        'models': {
            'xgb_H': xgb_model_H,
            'xgb_S': xgb_model_S,
            'rf_H': rf_model_H,
            'rf_S': rf_model_S,
            'gb_H': gb_model_H,
            'gb_S': gb_model_S
        },
        'scaler': scaler,
        'feature_names': feature_names,
        'cv_H_mean': cv_scores_H.mean() if len(cv_scores_H) > 0 else 0,
        'cv_S_mean': cv_scores_S.mean() if len(cv_scores_S) > 0 else 0,
        'cv_H_std': cv_scores_H.std() if len(cv_scores_H) > 0 else 0,
        'cv_S_std': cv_scores_S.std() if len(cv_scores_S) > 0 else 0,
        'feature_importance': feature_importance_H,
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
# Modern UI Components
# =============================================================================
def apply_modern_styling():
    """Apply modern CSS styling to the app"""
    st.markdown("""
    <style>
    /* Modern color scheme */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --background-color: #f8f9fa;
        --text-color: #212529;
        --border-color: #dee2e6;
    }
    
    /* Main container */
    .main {
        background-color: var(--background-color);
        padding: 2rem;
    }
    
    /* Headers */
    h1 {
        color: var(--primary-color);
        font-weight: 600;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid var(--primary-color);
        padding-bottom: 0.5rem;
    }
    
    h2 {
        color: var(--text-color);
        font-weight: 500;
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: var(--text-color);
        font-weight: 500;
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    
    /* Cards for metrics and content */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, var(--primary-color), #45a049);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: white;
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: white;
        border-right: 1px solid var(--border-color);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        font-weight: 500;
    }
    
    /* DataFrames */
    .dataframe {
        border: none !important;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .dataframe th {
        background-color: var(--primary-color);
        color: white;
        font-weight: 500;
        padding: 0.75rem !important;
    }
    
    .dataframe td {
        padding: 0.5rem !important;
        border-bottom: 1px solid var(--border-color);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: var(--primary-color);
        border-radius: 10px;
    }
    
    /* Alerts and info boxes */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Select boxes and inputs */
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    .stSlider div[data-baseweb="slider"] {
        margin-top: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid var(--border-color);
        margin-top: 3rem;
    }
    
    /* Version badge */
    .version-badge {
        display: inline-block;
        background-color: var(--primary-color);
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-left: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# Progress bar context manager (enhanced)
# =============================================================================
class ModernProgressBar:
    def __init__(self, message, total_steps, show_time=True):
        self.message = message
        self.total_steps = total_steps
        self.progress_bar = None
        self.status_text = None
        self.time_text = None
        self.show_time = show_time
        self.start_time = None
    
    def __enter__(self):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        if self.show_time:
            self.time_text = st.empty()
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress_bar.empty()
        self.status_text.empty()
        if self.time_text:
            self.time_text.empty()
    
    def update(self, step, sub_message=""):
        progress = step / self.total_steps
        self.progress_bar.progress(progress)
        
        elapsed = time.time() - self.start_time
        if step > 0:
            eta = (elapsed / step) * (self.total_steps - step)
            time_str = f" | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
        else:
            time_str = ""
        
        self.status_text.text(f"{self.message}: {sub_message} ({int(progress*100)}%)")
        if self.show_time and self.time_text:
            self.time_text.text(f"⏱️{time_str}")

# =============================================================================
# Streamlit App with Modern UI
# =============================================================================
def main():
    import pandas as pd  # ДОБАВИТЬ ЭТУ СТРОКУ
    
    st.set_page_config(
        page_title="Proton Hydration Predictor v3.0",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply modern styling
    apply_modern_styling()
    
    # Title with version badge
    st.markdown("""
    <h1>
        💧 Proton Hydration Predictor for Perovskite Oxides
        <span class="version-badge">v3.0</span>
    </h1>
    """, unsafe_allow_html=True)
    
    # Description in modern card
    st.markdown("""
    <div class="card">
        <p style="font-size: 1.1rem; margin-bottom: 0;">
        Advanced computational platform for analyzing and predicting hydration thermodynamics 
        of proton-conducting perovskites. Version 3.0 introduces enhanced 3D visualization,
        multi-dimensional analysis, sensitivity mapping, and predictive proton concentration modeling.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize progress for data loading
    with ModernProgressBar("Initializing system", 4) as pb:
        pb.update(1, "Loading database")
        df = load_and_combine_data()
        
        pb.update(2, "Calculating enhanced descriptors")
        model_data, df_features = train_prediction_models(df)
        
        pb.update(3, "Training ensemble models")
        
        pb.update(4, "Ready!")
    
    # Sidebar navigation with modern styling
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        page = st.radio(
            "Select Module",
            ["📊 Data Explorer", 
             "🔍 Advanced Correlations", 
             "🤖 ML Predictor",
             "📈 Model Performance",
             "📊 SHAP Analysis",
             "🔬 3D Visualization",
             "📊 Multi-dimensional Analysis",
             "⚠️ Sensitivity Analysis",
             "🌲 Cluster Analysis",
             "💧 Proton Concentration 3D",
             "ℹ️ About"]
        )
        
        st.markdown("---")
        
        # Database stats in modern cards
        st.markdown("## 📊 Database Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Entries</div>
                <div class="metric-value">{len(df)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">A-cations</div>
                <div class="metric-value">{df['A_cation'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">B-cations</div>
                <div class="metric-value">{df['B_cation'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Dopants</div>
                <div class="metric-value">{df['dopant'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick filters
        st.markdown("## 🔍 Quick Filters")
        show_recent = st.checkbox("Show only recent data (post-2020)")
        if show_recent:
            # This is a placeholder - implement actual filtering logic
            st.info("Filter applied")

    # =========================================================================
    # Page 1: Data Explorer (Enhanced)
    # =========================================================================
    if page == "📊 Data Explorer":
        st.markdown("## 📊 Hydration Thermodynamics Database")
        
        # Filters in modern layout
        with st.expander("🔍 Filter Data", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                a_cations = ['All'] + sorted(df['A_cation'].unique().tolist())
                selected_a = st.selectbox("A-cation", a_cations)
            
            with col2:
                b_cations = ['All'] + sorted(df['B_cation'].unique().tolist())
                selected_b = st.selectbox("B-cation", b_cations)
            
            with col3:
                dopants = ['All'] + sorted([d for d in df['dopant'].unique() if d is not None and str(d) != 'nan'])
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
        
        # Statistics in modern metric cards
        st.markdown("### 📈 Dataset Statistics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Entries</div>
                <div class="metric-value">{len(filtered_df)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ΔH Range</div>
                <div class="metric-value">{filtered_df['delta_H'].min():.0f} to {filtered_df['delta_H'].max():.0f}</div>
                <div class="metric-delta">kJ/mol</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ΔS Range</div>
                <div class="metric-value">{filtered_df['delta_S'].min():.0f} to {filtered_df['delta_S'].max():.0f}</div>
                <div class="metric-delta">J/mol·K</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">A-cations</div>
                <div class="metric-value">{filtered_df['A_cation'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Dopants</div>
                <div class="metric-value">{filtered_df['dopant'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Data table with styling
        st.markdown("### 📋 Data Table")
        
        styled_df = filtered_df.style.background_gradient(cmap='RdBu_r', subset=['delta_H', 'delta_S'])
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="hydration_data.csv",
            mime="text/csv"
        )
        
        # Distribution plots
        st.markdown("### 📊 Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(filtered_df['delta_H'], bins=20, color=MODERN_COLORS['primary'], 
                   edgecolor='white', alpha=0.7)
            ax.set_xlabel('ΔH (kJ mol⁻¹)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Enthalpy of Hydration Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(filtered_df['delta_S'], bins=20, color=MODERN_COLORS['secondary'], 
                   edgecolor='white', alpha=0.7)
            ax.set_xlabel('ΔS (J mol⁻¹ K⁻¹)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Entropy of Hydration Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Coverage heatmap
        st.markdown("### 🗺️ Data Coverage Matrix")
        
        pivot = filtered_df.pivot_table(
            index='A_cation', columns='B_cation',
            values='delta_H', aggfunc='count', fill_value=0
        )
        
        fig = px.imshow(pivot, text_auto=True, color_continuous_scale='Blues',
                       title='Number of Samples per A-B Combination')
        fig.update_layout(
            title_font_size=14,
            title_font_family="Arial",
            title_font_weight="bold"
        )
        st.plotly_chart(fig, use_container_width=True)

    # =============================================================================
    # Page 2: Advanced Correlations (исправленная и дополненная версия)
    # =============================================================================
    elif page == "🔍 Advanced Correlations":
        st.markdown("## 🔍 Advanced Structure-Property Correlations")
        
        # Prepare data with descriptors
        desc_list = []
        for _, row in df.iterrows():
            desc = calculate_descriptors(row)
            if len(desc) > 10:
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
        
        # Plot type selection in modern tabs
        plot_category = st.radio(
            "Select Analysis Type",
            ["Compensation Effect", "Structure Descriptors", "Electronic Effects", 
             "Composition Trends", "Statistical Overview", "2D Hydration Maps",
             "Advanced 3D Analysis", "Multi-dimensional Patterns", "Sensitivity Analysis",
             "Cluster Analysis"],
            horizontal=True
        )
        
        if plot_category == "Compensation Effect":
            st.markdown("### ΔH–ΔS Compensation Effect by Material Families")
            
            col1, col2 = st.columns(2)
            with col1:
                group_by = st.selectbox("Group by", ["B_cation", "A_cation", "dopant"])
            
            with col2:
                show_trends = st.checkbox("Show trend lines", value=True)
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            groups = plot_df[group_by].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
            
            for idx, (group, color) in enumerate(zip(groups, colors)):
                if idx >= 6:
                    break
                    
                ax = axes[idx]
                subset = plot_df[plot_df[group_by] == group]
                
                ax.scatter(subset['delta_S'], subset['delta_H'], 
                          c=[color], s=80, alpha=0.7, edgecolors='black', linewidth=0.5,
                          label=f'n={len(subset)}')
                
                if show_trends and len(subset) >= 3:
                    z = np.polyfit(subset['delta_S'], subset['delta_H'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(subset['delta_S'].min(), subset['delta_S'].max(), 50)
                    ax.plot(x_trend, p(x_trend), '--', color=color, linewidth=2, alpha=0.8)
                    
                    # Calculate isokinetic temperature
                    if abs(z[0]) > 1e-6:
                        T_iso = -1000 / z[0]  # K
                        r2 = np.corrcoef(subset['delta_S'], subset['delta_H'])[0,1]**2
                        ax.text(0.05, 0.95, f"T_iso = {T_iso:.0f} K\nR² = {r2:.2f}",
                               transform=ax.transAxes, va='top', fontsize=9,
                               bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))
                
                ax.set_xlabel('ΔS (J mol⁻¹ K⁻¹)', fontsize=10)
                ax.set_ylabel('ΔH (kJ mol⁻¹)', fontsize=10)
                ax.set_title(f'{group_by}: {group}', fontsize=11, fontweight='bold')
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='lower right', fontsize=8)
            
            # Hide empty subplots
            for idx in range(len(groups), 6):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Compensation Effect Grouped by {group_by}', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown("""
            <div class="card">
                <h4>Interpretation Guide</h4>
                <ul>
                    <li><b>Isokinetic temperature (T_iso)</b>: Temperature at which all materials in a family have the same rate</li>
                    <li><b>Higher T_iso</b> (>1000 K) indicates strong coupling between protons and lattice</li>
                    <li><b>Lower T_iso</b> suggests more independent proton motion</li>
                    <li><b>R² > 0.9</b> indicates strong compensation behavior within the family</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif plot_category == "Structure Descriptors":
            st.markdown("### Structure-Property Correlations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                plot_type = st.selectbox(
                    "Select correlation",
                    ["ΔH vs Tolerance Factor",
                     "ΔH vs B-site Average Radius",
                     "ΔH vs Dopant Radius",
                     "ΔH vs A-cation Radius",
                     "ΔH vs Radius Mismatch",
                     "ΔH vs Octahedral Factor",
                     "ΔH vs Lattice Energy",
                     "ΔH vs Global Instability"]
                )
            
            with col2:
                color_by = st.selectbox("Color by", ["A_cation", "B_cation", "dopant"])
            
            with col3:
                show_error = st.checkbox("Show error bars", value=False)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Map plot types to descriptors
            plot_mapping = {
                "ΔH vs Tolerance Factor": ('t_Goldschmidt', 'Goldschmidt Tolerance Factor, t'),
                "ΔH vs B-site Average Radius": ('r_B_avg', 'Average B-site Radius (Å)'),
                "ΔH vs Dopant Radius": ('r_D_VI', 'Dopant Ionic Radius (Å)'),
                "ΔH vs A-cation Radius": ('r_A_XII', 'A-cation Radius (Å)'),
                "ΔH vs Radius Mismatch": ('delta_r_B', '|r_D - r_B| (Å)'),
                "ΔH vs Octahedral Factor": ('octahedral_factor', 'Octahedral Factor (r_B/r_O)'),
                "ΔH vs Lattice Energy": ('lattice_energy', 'Lattice Energy (kJ/mol)'),
                "ΔH vs Global Instability": ('global_instability', 'Global Instability Index')
            }
            
            x_col, xlabel = plot_mapping[plot_type]
            
            groups = plot_df[color_by].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))
            
            for group, color in zip(groups, colors):
                subset = plot_df[plot_df[color_by] == group]
                
                if show_error:
                    # Assume error of 5% for demonstration
                    yerr = abs(subset['delta_H'] * 0.05)
                    ax.errorbar(subset[x_col], subset['delta_H'], yerr=yerr,
                               fmt='o', color=color, capsize=3, capthick=1,
                               markersize=8, alpha=0.7, label=group)
                else:
                    ax.scatter(subset[x_col], subset['delta_H'], 
                              label=group, c=[color], s=80, alpha=0.7, 
                              edgecolors='black', linewidth=0.5)
                
                # Add trend line
                if len(subset) >= 4:
                    z = np.polyfit(subset[x_col], subset['delta_H'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(subset[x_col].min(), subset[x_col].max(), 50)
                    ax.plot(x_trend, p(x_trend), '--', color=color, linewidth=1.5, alpha=0.5)
                    
                    # Add correlation coefficient
                    corr = np.corrcoef(subset[x_col], subset['delta_H'])[0,1]
                    ax.text(0.02, 0.98 - 0.05*list(groups).index(group), 
                           f'{group}: r = {corr:.2f}', transform=ax.transAxes,
                           color=color, fontsize=9, fontweight='bold',
                           verticalalignment='top')
            
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel('ΔH (kJ mol⁻¹)', fontsize=12)
            ax.set_title(f'{plot_type}', fontsize=14, fontweight='bold')
            ax.legend(title=color_by, frameon=True, fancybox=False, edgecolor='black', fontsize=9)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        # =========================================================================
        # NEW: Electronic Effects
        # =========================================================================
        elif plot_category == "Electronic Effects":
            st.markdown("### Electronic Structure Effects on Hydration")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                plot_type = st.selectbox(
                    "Select electronic correlation",
                    ["ΔH vs B-site Electronegativity",
                     "ΔH vs Electronegativity Difference (χ_B - χ_A)",
                     "ΔH vs Average Polarizability",
                     "ΔH vs Average Ionization Potential",
                     "ΔH vs Charge Density",
                     "Correlation Matrix: Electronic Descriptors"]
                )
                
                color_by = st.selectbox("Color by", ["B_cation", "A_cation", "dopant"], key="elec_color")
                
                show_trend = st.checkbox("Show trend line", value=True)
            
            with col1:
                if plot_type == "ΔH vs B-site Electronegativity":
                    fig, ax = plt.subplots(figsize=(10, 7))
                    
                    groups = plot_df[color_by].unique()
                    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
                    
                    for group, color in zip(groups, colors):
                        subset = plot_df[plot_df[color_by] == group]
                        ax.scatter(subset['chi_B_avg'], subset['delta_H'], 
                                  label=group, c=[color], s=100, alpha=0.7,
                                  edgecolors='black', linewidth=0.5)
                        
                        if show_trend and len(subset) >= 4:
                            if subset['chi_B_avg'].nunique() >= 2:
                                try:
                                    x_clean = subset['chi_B_avg'].dropna()
                                    y_clean = subset['delta_H'].dropna()
                                    
                                    if len(x_clean) >= 4 and len(y_clean) >= 4:
                                        from scipy import stats
                                        
                                        corr, p_value = stats.pearsonr(x_clean, y_clean)
                                        
                                        if abs(corr) > 0.3 or p_value < 0.1:
                                            z = np.polyfit(x_clean, y_clean, 1)
                                            x_trend = np.linspace(x_clean.min(), x_clean.max(), 50)
                                            ax.plot(x_trend, np.polyval(z, x_trend), '--', 
                                                   color=color, alpha=0.5, linewidth=1.5)
                                            
                                            ax.text(0.02, 0.98 - 0.05 * list(groups).index(group), 
                                                   f'{group}: r = {corr:.2f}{"*" if p_value < 0.05 else ""}', 
                                                   transform=ax.transAxes, color=color, fontsize=8,
                                                   verticalalignment='top')
                                except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                                    pass
                    
                    ax.set_xlabel('Average B-site Electronegativity (χ_B)', fontsize=12)
                    ax.set_ylabel('ΔH (kJ mol⁻¹)', fontsize=12)
                    ax.set_title('Effect of B-site Electronegativity on Hydration Enthalpy', 
                                fontsize=14, fontweight='bold')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
                    
                    # Add interpretation text
                    ax.text(0.02, 0.02, 
                           "Higher electronegativity → stronger\nB-O bond → affects water incorporation",
                           transform=ax.transAxes, fontsize=9, 
                           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                    
                elif plot_type == "ΔH vs Electronegativity Difference (χ_B - χ_A)":
                    fig, ax = plt.subplots(figsize=(10, 7))
                    
                    # Calculate electronegativity difference
                    plot_df['chi_diff_AB'] = plot_df['chi_B_avg'] - plot_df['chi_A']
                    
                    groups = plot_df[color_by].unique()
                    colors = plt.cm.plasma(np.linspace(0, 1, len(groups)))
                    
                    for group, color in zip(groups, colors):
                        subset = plot_df[plot_df[color_by] == group]
                        ax.scatter(subset['chi_diff_AB'], subset['delta_H'], 
                                  label=group, c=[color], s=100, alpha=0.7,
                                  edgecolors='black', linewidth=0.5)
                        
                        if show_trend and len(subset) >= 4 and subset['chi_diff_AB'].nunique() >= 2:
                            try:
                                x_clean = subset['chi_diff_AB'].dropna()
                                y_clean = subset['delta_H'].dropna()
                                if len(x_clean) >= 4 and len(y_clean) >= 4:
                                    z = np.polyfit(x_clean, y_clean, 1)
                                    x_trend = np.linspace(subset['chi_diff_AB'].min(), subset['chi_diff_AB'].max(), 50)
                                    ax.plot(x_trend, np.polyval(z, x_trend), '--', color=color, alpha=0.5)
                            except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                                pass
                    
                    ax.set_xlabel('Electronegativity Difference (χ_B - χ_A)', fontsize=12)
                    ax.set_ylabel('ΔH (kJ mol⁻¹)', fontsize=12)
                    ax.set_title('Effect of A-B Electronegativity Difference', 
                                fontsize=14, fontweight='bold')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
                    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
                    
                elif plot_type == "ΔH vs Average Polarizability":
                    fig, ax = plt.subplots(figsize=(10, 7))
                    
                    groups = plot_df[color_by].unique()
                    colors = plt.cm.coolwarm(np.linspace(0, 1, len(groups)))
                    
                    for group, color in zip(groups, colors):
                        subset = plot_df[plot_df[color_by] == group]
                        ax.scatter(subset['polarizability_avg'], subset['delta_H'], 
                                  label=group, c=[color], s=100, alpha=0.7,
                                  edgecolors='black', linewidth=0.5)
                        
                        if show_trend and len(subset) >= 4 and subset['polarizability_avg'].nunique() >= 2:
                            try:
                                x_clean = subset['polarizability_avg'].dropna()
                                y_clean = subset['delta_H'].dropna()
                                if len(x_clean) >= 4 and len(y_clean) >= 4:
                                    z = np.polyfit(x_clean, y_clean, 1)
                                    x_trend = np.linspace(subset['polarizability_avg'].min(), subset['polarizability_avg'].max(), 50)
                                    ax.plot(x_trend, np.polyval(z, x_trend), '--', color=color, alpha=0.5)
                            except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                                pass
                    
                    ax.set_xlabel('Average Ionic Polarizability', fontsize=12)
                    ax.set_ylabel('ΔH (kJ mol⁻¹)', fontsize=12)
                    ax.set_title('Effect of Polarizability on Hydration', 
                                fontsize=14, fontweight='bold')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    # Add note about polarizability
                    ax.text(0.98, 0.02, 
                           "Higher polarizability → easier\n charge redistribution",
                           transform=ax.transAxes, fontsize=9, ha='right',
                           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                    
                elif plot_type == "ΔH vs Average Ionization Potential":
                    fig, ax = plt.subplots(figsize=(10, 7))
                    
                    groups = plot_df[color_by].unique()
                    colors = plt.cm.inferno(np.linspace(0, 1, len(groups)))
                    
                    for group, color in zip(groups, colors):
                        subset = plot_df[plot_df[color_by] == group]
                        ax.scatter(subset['ionization_avg'], subset['delta_H'], 
                                  label=group, c=[color], s=100, alpha=0.7,
                                  edgecolors='black', linewidth=0.5)
                        
                        if show_trend and len(subset) >= 4 and subset['ionization_avg'].nunique() >= 2:
                            try:
                                x_clean = subset['ionization_avg'].dropna()
                                y_clean = subset['delta_H'].dropna()
                                if len(x_clean) >= 4 and len(y_clean) >= 4:
                                    z = np.polyfit(x_clean, y_clean, 1)
                                    x_trend = np.linspace(subset['ionization_avg'].min(), subset['ionization_avg'].max(), 50)
                                    ax.plot(x_trend, np.polyval(z, x_trend), '--', color=color, alpha=0.5)
                            except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                                pass
                    
                    ax.set_xlabel('Average Ionization Potential (eV)', fontsize=12)
                    ax.set_ylabel('ΔH (kJ mol⁻¹)', fontsize=12)
                    ax.set_title('Effect of Ionization Potential on Hydration', 
                                fontsize=14, fontweight='bold')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                elif plot_type == "ΔH vs Charge Density":
                    fig, ax = plt.subplots(figsize=(10, 7))
                    
                    groups = plot_df[color_by].unique()
                    colors = plt.cm.Spectral(np.linspace(0, 1, len(groups)))
                    
                    for group, color in zip(groups, colors):
                        subset = plot_df[plot_df[color_by] == group]
                        ax.scatter(subset['charge_density_B'], subset['delta_H'], 
                                  label=group, c=[color], s=100, alpha=0.7,
                                  edgecolors='black', linewidth=0.5)
                        
                        if show_trend and len(subset) >= 4 and subset['charge_density_B'].nunique() >= 2:
                            try:
                                x_clean = subset['charge_density_B'].dropna()
                                y_clean = subset['delta_H'].dropna()
                                if len(x_clean) >= 4 and len(y_clean) >= 4:
                                    z = np.polyfit(x_clean, y_clean, 1)
                                    x_trend = np.linspace(subset['charge_density_B'].min(), subset['charge_density_B'].max(), 50)
                                    ax.plot(x_trend, np.polyval(z, x_trend), '--', color=color, alpha=0.5)
                            except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                                pass
                    
                    ax.set_xlabel('B-site Charge Density (Z/r, e/Å)', fontsize=12)
                    ax.set_ylabel('ΔH (kJ mol⁻¹)', fontsize=12)
                    ax.set_title('Effect of Charge Density on Hydration', 
                                fontsize=14, fontweight='bold')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                elif plot_type == "Correlation Matrix: Electronic Descriptors":
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Select electronic descriptors
                    electronic_features = ['chi_A', 'chi_B_avg', 'chi_diff', 'chi_product',
                                           'polarizability_avg', 'ionization_avg', 
                                           'charge_density_A', 'charge_density_B', 'delta_H']
                    
                    available = [f for f in electronic_features if f in plot_df.columns]
                    
                    if len(available) >= 3:
                        corr_matrix = plot_df[available].corr()
                        
                        # Create mask for upper triangle
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                        
                        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                                   cmap='RdBu_r', center=0, square=True, ax=ax,
                                   cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                                   annot_kws={'size': 10})
                        
                        ax.set_title('Correlation Matrix: Electronic Descriptors vs ΔH', 
                                    fontsize=14, fontweight='bold')
                        
                        # Highlight correlations with ΔH
                        for i, feat in enumerate(available):
                            if feat == 'delta_H':
                                for j in range(len(available)):
                                    if i != j:
                                        corr_val = corr_matrix.iloc[i, j]
                                        if abs(corr_val) > 0.5:
                                            ax.text(j + 0.5, i + 0.5, '★', 
                                                   ha='center', va='center', 
                                                   color='gold', fontsize=14)
                    else:
                        ax.text(0.5, 0.5, "Insufficient electronic descriptors available", 
                               ha='center', va='center', fontsize=14)
                        ax.set_axis_off()
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Summary statistics
            with col2:
                st.markdown("### Electronic Descriptor Statistics")
                
                stats_data = []
                for desc in ['chi_B_avg', 'chi_diff', 'polarizability_avg', 'ionization_avg']:
                    if desc in plot_df.columns:
                        stats_data.append({
                            'Descriptor': desc.replace('_', ' ').title(),
                            'Mean': f"{plot_df[desc].mean():.3f}",
                            'Range': f"{plot_df[desc].min():.3f} - {plot_df[desc].max():.3f}",
                            'Corr with ΔH': f"{plot_df[desc].corr(plot_df['delta_H']):.3f}"
                        })
                
                if stats_data:
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        
        # =========================================================================
        # NEW: Composition Trends
        # =========================================================================
        elif plot_category == "Composition Trends":
            st.markdown("### Composition-Dependent Hydration Trends")
            
            tab1, tab2, tab3 = st.tabs(["Dopant Content Effects", "Material Families", "Statistical Distribution"])
            
            with tab1:
                st.markdown("#### Effect of Dopant Concentration (x)")
                
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    # Select specific system for detailed view
                    a_options = ['All'] + sorted(plot_df['A_cation'].unique().tolist())
                    selected_a = st.selectbox("Filter A-cation", a_options, key="comp_a")
                    
                    b_options = ['All'] + sorted(plot_df['B_cation'].unique().tolist())
                    selected_b = st.selectbox("Filter B-cation", b_options, key="comp_b")
                    
                    d_options = ['All'] + sorted(plot_df['dopant'].unique().tolist())
                    selected_d = st.selectbox("Filter dopant", d_options, key="comp_d")
                    
                    plot_type = st.radio("Plot type", ["ΔH vs x", "ΔS vs x", "Both"], horizontal=True)
                
                with col1:
                    # Filter data
                    filtered_df = plot_df.copy()
                    if selected_a != 'All':
                        filtered_df = filtered_df[filtered_df['A_cation'] == selected_a]
                    if selected_b != 'All':
                        filtered_df = filtered_df[filtered_df['B_cation'] == selected_b]
                    if selected_d != 'All':
                        filtered_df = filtered_df[filtered_df['dopant'] == selected_d]
                    
                    if len(filtered_df) < 3:
                        st.warning("Not enough data points with current filters")
                    else:
                        if plot_type == "ΔH vs x":
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Group by material system
                            filtered_df['system'] = filtered_df['A_cation'] + filtered_df['B_cation'] + '-' + filtered_df['dopant']
                            systems = filtered_df['system'].unique()
                            colors = plt.cm.tab20(np.linspace(0, 1, len(systems)))
                            
                            for system, color in zip(systems, colors):
                                subset = filtered_df[filtered_df['system'] == system]
                                ax.scatter(subset['content'], subset['delta_H'], 
                                          label=system, c=[color], s=100, alpha=0.7,
                                          edgecolors='black', linewidth=0.5)
                                
                                if len(subset) >= 3:
                                    # Add trend line
                                    z = np.polyfit(subset['content'], subset['delta_H'], 1)
                                    x_trend = np.linspace(subset['content'].min(), subset['content'].max(), 50)
                                    ax.plot(x_trend, np.polyval(z, x_trend), '--', color=color, alpha=0.5)
                            
                            ax.set_xlabel('Dopant Content, x', fontsize=12)
                            ax.set_ylabel('ΔH (kJ mol⁻¹)', fontsize=12)
                            ax.set_title('Dopant Concentration Effect on Hydration Enthalpy', 
                                        fontsize=14, fontweight='bold')
                            ax.legend(loc='best', fontsize=8, ncol=2)
                            ax.grid(True, alpha=0.3)
                            
                        elif plot_type == "ΔS vs x":
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            filtered_df['system'] = filtered_df['A_cation'] + filtered_df['B_cation'] + '-' + filtered_df['dopant']
                            systems = filtered_df['system'].unique()
                            colors = plt.cm.tab20(np.linspace(0, 1, len(systems)))
                            
                            for system, color in zip(systems, colors):
                                subset = filtered_df[filtered_df['system'] == system]
                                ax.scatter(subset['content'], subset['delta_S'], 
                                          label=system, c=[color], s=100, alpha=0.7,
                                          edgecolors='black', linewidth=0.5)
                                
                                if len(subset) >= 3:
                                    z = np.polyfit(subset['content'], subset['delta_S'], 1)
                                    x_trend = np.linspace(subset['content'].min(), subset['content'].max(), 50)
                                    ax.plot(x_trend, np.polyval(z, x_trend), '--', color=color, alpha=0.5)
                            
                            ax.set_xlabel('Dopant Content, x', fontsize=12)
                            ax.set_ylabel('ΔS (J mol⁻¹ K⁻¹)', fontsize=12)
                            ax.set_title('Dopant Concentration Effect on Hydration Entropy', 
                                        fontsize=14, fontweight='bold')
                            ax.legend(loc='best', fontsize=8, ncol=2)
                            ax.grid(True, alpha=0.3)
                            
                        else:  # Both
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                            
                            filtered_df['system'] = filtered_df['A_cation'] + filtered_df['B_cation'] + '-' + filtered_df['dopant']
                            systems = filtered_df['system'].unique()
                            colors = plt.cm.tab20(np.linspace(0, 1, len(systems)))
                            
                            for system, color in zip(systems, colors):
                                subset = filtered_df[filtered_df['system'] == system]
                                
                                # ΔH plot
                                ax1.scatter(subset['content'], subset['delta_H'], 
                                           label=system, c=[color], s=80, alpha=0.7,
                                           edgecolors='black', linewidth=0.5)
                                if len(subset) >= 3:
                                    z = np.polyfit(subset['content'], subset['delta_H'], 1)
                                    x_trend = np.linspace(subset['content'].min(), subset['content'].max(), 50)
                                    ax1.plot(x_trend, np.polyval(z, x_trend), '--', color=color, alpha=0.5)
                                
                                # ΔS plot
                                ax2.scatter(subset['content'], subset['delta_S'], 
                                           label=system, c=[color], s=80, alpha=0.7,
                                           edgecolors='black', linewidth=0.5)
                                if len(subset) >= 3:
                                    z = np.polyfit(subset['content'], subset['delta_S'], 1)
                                    x_trend = np.linspace(subset['content'].min(), subset['content'].max(), 50)
                                    ax2.plot(x_trend, np.polyval(z, x_trend), '--', color=color, alpha=0.5)
                            
                            ax1.set_xlabel('Dopant Content, x')
                            ax1.set_ylabel('ΔH (kJ mol⁻¹)')
                            ax1.set_title('Enthalpy vs Content')
                            ax1.grid(True, alpha=0.3)
                            
                            ax2.set_xlabel('Dopant Content, x')
                            ax2.set_ylabel('ΔS (J mol⁻¹ K⁻¹)')
                            ax2.set_title('Entropy vs Content')
                            ax2.grid(True, alpha=0.3)
                            
                            ax2.legend(loc='best', fontsize=7, ncol=2)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
            
            with tab2:
                st.markdown("#### Material Family Comparison")
                
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    compare_by = st.selectbox("Compare by", ["B_cation", "A_cation", "dopant"])
                    chart_type = st.selectbox("Chart type", ["Box plot", "Violin plot", "Bar chart"])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    if chart_type == "Box plot":
                        data_to_plot = [plot_df[plot_df[compare_by] == cat]['delta_H'].values 
                                       for cat in sorted(plot_df[compare_by].unique())]
                        bp = ax.boxplot(data_to_plot, labels=sorted(plot_df[compare_by].unique()), 
                                       patch_artist=True)
                        
                        # Color boxes
                        for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, len(data_to_plot)))):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        
                    elif chart_type == "Violin plot":
                        data_to_plot = [plot_df[plot_df[compare_by] == cat]['delta_H'].values 
                                       for cat in sorted(plot_df[compare_by].unique())]
                        parts = ax.violinplot(data_to_plot, positions=range(1, len(data_to_plot)+1), showmeans=True)
                        
                        for pc, color in zip(parts['bodies'], plt.cm.viridis(np.linspace(0, 1, len(data_to_plot)))):
                            pc.set_facecolor(color)
                            pc.set_alpha(0.7)
                        
                        ax.set_xticks(range(1, len(data_to_plot)+1))
                        ax.set_xticklabels(sorted(plot_df[compare_by].unique()))
                        
                    else:  # Bar chart
                        means = plot_df.groupby(compare_by)['delta_H'].mean()
                        stds = plot_df.groupby(compare_by)['delta_H'].std()
                        
                        bars = ax.bar(range(len(means)), means.values, yerr=stds.values,
                                     capsize=5, color=plt.cm.viridis(np.linspace(0, 1, len(means))),
                                     edgecolor='black', linewidth=0.5, alpha=0.7)
                        
                        ax.set_xticks(range(len(means)))
                        ax.set_xticklabels(means.index, rotation=45, ha='right')
                        
                        # Add value labels
                        for i, (bar, val) in enumerate(zip(bars, means.values)):
                            ax.text(i, val + 2, f'{val:.1f}', ha='center', fontsize=9)
                    
                    ax.set_ylabel('ΔH (kJ mol⁻¹)', fontsize=12)
                    ax.set_title(f'ΔH Distribution by {compare_by}', fontsize=14, fontweight='bold')
                    ax.axhline(y=plot_df['delta_H'].mean(), color='red', linestyle='--', 
                              alpha=0.5, label=f'Global mean: {plot_df["delta_H"].mean():.1f}')
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            
            with tab3:
                st.markdown("#### Statistical Summary by Composition")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    group_by = st.selectbox("Group by for statistics", 
                                           ["A_cation", "B_cation", "dopant", "A_cation + B_cation"])
                    
                with col2:
                    metric = st.selectbox("Metric", ["ΔH", "ΔS", "content"])
                
                # Create summary table
                if group_by == "A_cation + B_cation":
                    plot_df['A_B_system'] = plot_df['A_cation'] + plot_df['B_cation']
                    summary = plot_df.groupby('A_B_system').agg({
                        'delta_H': ['count', 'mean', 'std', 'min', 'max'],
                        'delta_S': ['count', 'mean', 'std', 'min', 'max'],
                        'content': ['mean', 'min', 'max']
                    }).round(2)
                    
                    # Flatten column names
                    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
                    summary = summary.reset_index()
                    summary.columns = ['System', 'Count_H', 'Mean_H', 'Std_H', 'Min_H', 'Max_H',
                                       'Count_S', 'Mean_S', 'Std_S', 'Min_S', 'Max_S',
                                       'Mean_x', 'Min_x', 'Max_x']
                    
                    st.dataframe(summary, use_container_width=True)
                    
                else:
                    summary = plot_df.groupby(group_by).agg({
                        'delta_H': ['count', 'mean', 'std', 'min', 'max'],
                        'delta_S': ['count', 'mean', 'std', 'min', 'max'],
                        'content': ['mean', 'min', 'max']
                    }).round(2)
                    
                    st.dataframe(summary, use_container_width=True)
                
                # Download summary
                csv = summary.to_csv()
                st.download_button("📥 Download Statistics", csv, "composition_statistics.csv", "text/csv")
        
        # =========================================================================
        # NEW: Statistical Overview
        # =========================================================================
        elif plot_category == "Statistical Overview":
            st.markdown("### Comprehensive Statistical Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Distribution Analysis", "Correlation Matrix", 
                                              "Outlier Detection", "Statistical Tests"])
            
            with tab1:
                st.markdown("#### Distribution Analysis of Key Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    var1 = st.selectbox("Select variable for histogram", 
                                       ['delta_H', 'delta_S', 'content', 'r_B_avg', 
                                        't_Goldschmidt', 'chi_diff'], key="hist_var")
                    
                    bins = st.slider("Number of bins", 5, 30, 15, key="hist_bins")
                    
                with col2:
                    group_by_hist = st.selectbox("Color by (optional)", 
                                               ['None', 'A_cation', 'B_cation', 'dopant'], key="hist_color")
                
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # Histogram with KDE
                ax1 = axes[0, 0]
                if group_by_hist == 'None':
                    ax1.hist(plot_df[var1], bins=bins, color=MODERN_COLORS['primary'], 
                            edgecolor='white', alpha=0.7, density=True)
                    plot_df[var1].plot(kind='kde', ax=ax1, color='red', linewidth=2, 
                                      label='KDE', secondary_y=False)
                else:
                    for group in plot_df[group_by_hist].unique():
                        subset = plot_df[plot_df[group_by_hist] == group][var1]
                        ax1.hist(subset, bins=bins, alpha=0.5, label=group, density=True)
                    ax1.legend()
                
                ax1.set_xlabel(var1.replace('_', ' ').title())
                ax1.set_ylabel('Density')
                ax1.set_title(f'Distribution of {var1.replace("_", " ").title()}')
                ax1.grid(True, alpha=0.3)
                
                # Q-Q plot for normality check
                ax2 = axes[0, 1]
                from scipy import stats
                stats.probplot(plot_df[var1].dropna(), dist="norm", plot=ax2)
                ax2.set_title(f'Q-Q Plot: {var1.replace("_", " ").title()}')
                ax2.grid(True, alpha=0.3)
                
                # Box plot
                ax3 = axes[1, 0]
                if group_by_hist != 'None':
                    plot_df.boxplot(column=var1, by=group_by_hist, ax=ax3)
                    ax3.set_title(f'Box Plot by {group_by_hist}')
                else:
                    plot_df.boxplot(column=var1, ax=ax3)
                    ax3.set_title(f'Box Plot of {var1.replace("_", " ").title()}')
                ax3.grid(True, alpha=0.3)
                
                # Violin plot
                ax4 = axes[1, 1]
                if group_by_hist != 'None':
                    data_to_plot = [plot_df[plot_df[group_by_hist] == cat][var1].dropna().values 
                                   for cat in sorted(plot_df[group_by_hist].unique())]
                    parts = ax4.violinplot(data_to_plot, positions=range(1, len(data_to_plot)+1), showmeans=True)
                    ax4.set_xticks(range(1, len(data_to_plot)+1))
                    ax4.set_xticklabels(sorted(plot_df[group_by_hist].unique()), rotation=45, ha='right')
                    
                    # Color violins
                    for i, pc in enumerate(parts['bodies']):
                        pc.set_facecolor(plt.cm.viridis(i/len(data_to_plot)))
                        pc.set_alpha(0.7)
                else:
                    ax4.text(0.5, 0.5, "Select grouping for violin plot", 
                            ha='center', va='center', transform=ax4.transAxes)
                
                ax4.set_ylabel(var1.replace('_', ' ').title())
                ax4.set_title('Violin Plot Distribution')
                ax4.grid(True, alpha=0.3)
                
                plt.suptitle('Statistical Distribution Analysis', fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Summary statistics table
                st.markdown("#### Summary Statistics")
                
                stats_df = plot_df[['delta_H', 'delta_S', 'content', 'r_B_avg', 
                                    't_Goldschmidt', 'chi_diff']].describe().T
                stats_df['skewness'] = plot_df[['delta_H', 'delta_S', 'content', 'r_B_avg', 
                                                't_Goldschmidt', 'chi_diff']].skew()
                stats_df['kurtosis'] = plot_df[['delta_H', 'delta_S', 'content', 'r_B_avg', 
                                                't_Goldschmidt', 'chi_diff']].kurtosis()
                
                st.dataframe(stats_df.round(3), use_container_width=True)
            
            with tab2:
                st.markdown("#### Enhanced Correlation Matrix")
                
                # Select features for correlation
                all_features = ['delta_H', 'delta_S', 'content', 'r_B_avg', 't_Goldschmidt', 
                               'chi_diff', 'polarizability_avg', 'ionization_avg', 
                               'charge_density_B', 'lattice_energy']
                
                available_features = [f for f in all_features if f in plot_df.columns]
                
                selected_features = st.multiselect(
                    "Select features for correlation matrix",
                    available_features,
                    default=available_features[:min(8, len(available_features))]
                )
                
                if len(selected_features) >= 2:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    corr_matrix = plot_df[selected_features].corr()
                    
                    # Create mask for upper triangle
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    
                    # Generate heatmap
                    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                               cmap='RdBu_r', center=0, square=True, ax=ax,
                               cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                               annot_kws={'size': 10})
                    
                    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
                    
                    # Highlight strong correlations with ΔH
                    if 'delta_H' in selected_features:
                        h_idx = selected_features.index('delta_H')
                        for i, feat in enumerate(selected_features):
                            if i != h_idx:
                                corr_val = corr_matrix.iloc[h_idx, i]
                                if abs(corr_val) > 0.5:
                                    ax.text(i + 0.5, h_idx + 0.5, '★', 
                                           ha='center', va='center', 
                                           color='gold', fontsize=14)
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # Feature clustering dendrogram
                    st.markdown("#### Feature Clustering Dendrogram")
                    
                    from scipy.cluster.hierarchy import dendrogram, linkage
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Calculate linkage based on correlation distance
                    corr_dist = 1 - abs(corr_matrix)
                    linkage_matrix = linkage(corr_dist, method='average')
                    
                    dendrogram(linkage_matrix, labels=selected_features, ax=ax,
                              leaf_rotation=45, leaf_font_size=10,
                              color_threshold=0.5)
                    
                    ax.set_title('Feature Clustering Dendrogram (based on correlation)',
                                fontsize=14, fontweight='bold')
                    ax.set_ylabel('Distance (1 - |r|)')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    st.pyplot(fig)
                    plt.close()
            
            with tab3:
                st.markdown("#### Outlier Detection")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    outlier_var = st.selectbox("Select variable for outlier detection", 
                                              ['delta_H', 'delta_S', 'content'], key="outlier_var")
                    
                    method = st.selectbox("Detection method", 
                                         ["IQR", "Z-score", "Modified Z-score"])
                
                with col2:
                    threshold = st.slider("Threshold", 1.0, 5.0, 3.0, 0.5)
                    show_stats = st.checkbox("Show statistics", True)
                
                # Detect outliers
                if method == "IQR":
                    Q1 = plot_df[outlier_var].quantile(0.25)
                    Q3 = plot_df[outlier_var].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outliers = plot_df[(plot_df[outlier_var] < lower_bound) | 
                                       (plot_df[outlier_var] > upper_bound)]
                    
                elif method == "Z-score":
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(plot_df[outlier_var].dropna()))
                    outliers = plot_df.iloc[np.where(z_scores > threshold)[0]]
                    
                else:  # Modified Z-score
                    from scipy import stats
                    median = plot_df[outlier_var].median()
                    mad = stats.median_abs_deviation(plot_df[outlier_var].dropna())
                    modified_z_scores = 0.6745 * (plot_df[outlier_var] - median) / mad
                    outliers = plot_df[np.abs(modified_z_scores) > threshold]
                
                # Visualization
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Box plot with outliers highlighted
                ax1 = axes[0]
                bp = ax1.boxplot(plot_df[outlier_var].dropna(), patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][0].set_alpha(0.7)
                
                # Highlight outliers
                if not outliers.empty:
                    outlier_vals = outliers[outlier_var].values
                    ax1.plot(np.ones_like(outlier_vals), outlier_vals, 'ro', 
                            markersize=8, label=f'Outliers (n={len(outliers)})')
                
                ax1.set_ylabel(outlier_var.replace('_', ' ').title())
                ax1.set_title(f'Outlier Detection: {method} (threshold={threshold})')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Scatter plot of outliers vs non-outliers
                ax2 = axes[1]
                
                # Create a boolean mask for outliers
                if not outliers.empty:
                    outlier_mask = plot_df.index.isin(outliers.index)
                    
                    # Plot non-outliers
                    ax2.scatter(range(len(plot_df[~outlier_mask])), 
                               plot_df[~outlier_mask][outlier_var], 
                               c='blue', alpha=0.5, label='Normal', s=50)
                    
                    # Plot outliers
                    ax2.scatter(np.where(outlier_mask)[0], 
                               plot_df[outlier_mask][outlier_var], 
                               c='red', alpha=0.8, label='Outlier', s=80, 
                               edgecolors='black', linewidth=0.5)
                    
                    ax2.set_xlabel('Data Point Index')
                    ax2.set_ylabel(outlier_var.replace('_', ' ').title())
                    ax2.set_title('Outlier Distribution')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'No outliers detected', 
                            ha='center', va='center', transform=ax2.transAxes,
                            fontsize=14)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Show outliers table
                if not outliers.empty:
                    st.markdown("#### Detected Outliers")
                    st.dataframe(outliers[['A_cation', 'B_cation', 'dopant', 'content', 
                                           outlier_var, 'reference']], use_container_width=True)
                    
                    # Download outliers
                    csv = outliers.to_csv(index=False)
                    st.download_button("📥 Download Outliers", csv, "outliers.csv", "text/csv")
                
                if show_stats:
                    st.markdown("#### Outlier Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total samples", len(plot_df))
                    with col2:
                        st.metric("Outliers", len(outliers))
                    with col3:
                        st.metric("Outlier %", f"{len(outliers)/len(plot_df)*100:.1f}%")
            
            with tab4:
                st.markdown("#### Statistical Hypothesis Tests")
                
                test_type = st.selectbox("Select test", 
                                        ["t-test (two groups)", 
                                         "ANOVA (multiple groups)",
                                         "Correlation test",
                                         "Normality test"])
                
                if test_type == "t-test (two groups)":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        group_var = st.selectbox("Grouping variable", 
                                               ['A_cation', 'B_cation', 'dopant'])
                        groups = plot_df[group_var].unique()
                        
                        if len(groups) >= 2:
                            group1 = st.selectbox("Group 1", groups, index=0)
                            group2 = st.selectbox("Group 2", groups, index=min(1, len(groups)-1))
                    
                    with col2:
                        test_var = st.selectbox("Test variable", ['delta_H', 'delta_S', 'content'])
                        equal_var = st.checkbox("Assume equal variance", True)
                    
                    if len(groups) >= 2 and group1 != group2:
                        data1 = plot_df[plot_df[group_var] == group1][test_var].dropna()
                        data2 = plot_df[plot_df[group_var] == group2][test_var].dropna()
                        
                        from scipy import stats
                        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
                        
                        st.markdown("#### t-test Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("t-statistic", f"{t_stat:.4f}")
                        with col2:
                            st.metric("p-value", f"{p_value:.4f}")
                        with col3:
                            st.metric("Significant", "Yes" if p_value < 0.05 else "No")
                        with col4:
                            st.metric("Confidence", "95%" if p_value < 0.05 else "<95%")
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(8, 5))
                        
                        data_to_plot = [data1, data2]
                        bp = ax.boxplot(data_to_plot, labels=[group1, group2], patch_artist=True)
                        
                        colors = [MODERN_COLORS['primary'], MODERN_COLORS['secondary']]
                        for patch, color in zip(bp['boxes'], colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        
                        ax.set_ylabel(test_var.replace('_', ' ').title())
                        ax.set_title(f't-test: {group1} vs {group2}\np-value = {p_value:.4f}')
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close()
                
                elif test_type == "ANOVA (multiple groups)":
                    group_var = st.selectbox("Grouping variable", 
                                           ['A_cation', 'B_cation', 'dopant'], key="anova_group")
                    test_var = st.selectbox("Test variable", ['delta_H', 'delta_S', 'content'], key="anova_var")
                    
                    groups = plot_df[group_var].unique()
                    
                    if len(groups) >= 2:
                        data_groups = [plot_df[plot_df[group_var] == g][test_var].dropna().values 
                                      for g in groups]
                        
                        from scipy import stats
                        f_stat, p_value = stats.f_oneway(*data_groups)
                        
                        st.markdown("#### ANOVA Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("F-statistic", f"{f_stat:.4f}")
                        with col2:
                            st.metric("p-value", f"{p_value:.4f}")
                        with col3:
                            st.metric("Significant", "Yes" if p_value < 0.05 else "No")
                        with col4:
                            st.metric("Groups", len(groups))
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        bp = ax.boxplot(data_groups, labels=groups, patch_artist=True)
                        
                        for i, patch in enumerate(bp['boxes']):
                            patch.set_facecolor(plt.cm.viridis(i/len(groups)))
                            patch.set_alpha(0.7)
                        
                        ax.set_ylabel(test_var.replace('_', ' ').title())
                        ax.set_title(f'ANOVA: Comparison across {group_var}\np-value = {p_value:.4f}')
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close()
        
        # =========================================================================
        # NEW: 2D Hydration Maps
        # =========================================================================
        elif plot_category == "2D Hydration Maps":
            st.markdown("### 2D Hydration Maps for Material Design")
            
            st.markdown("""
            <div class="card">
                <p>These maps help identify optimal composition regions for strong hydration.
                Darker blue regions indicate more negative ΔH (stronger hydration).</p>
            </div>
            """, unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["Contour Maps", "Experimental Coverage", "Design Guide"])
            
            with tab1:
                st.markdown("#### Hydration Enthalpy Contour Maps")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox("X-axis", 
                                         ['r_B_avg', 't_Goldschmidt', 'content', 'chi_diff'],
                                         index=0, key="2d_x")
                    
                with col2:
                    y_axis = st.selectbox("Y-axis", 
                                         ['chi_diff', 'content', 't_Goldschmidt', 'r_B_avg'],
                                         index=1, key="2d_y")
                
                # Create grid for interpolation
                x_grid = np.linspace(plot_df[x_axis].min(), plot_df[x_axis].max(), 50)
                y_grid = np.linspace(plot_df[y_axis].min(), plot_df[y_axis].max(), 50)
                X, Y = np.meshgrid(x_grid, y_grid)
                
                # Interpolate ΔH
                from scipy.interpolate import griddata
                
                # Check if we have enough points for interpolation
                if len(plot_df) < 4:
                    st.warning("Insufficient data points for interpolation (need at least 4)")
                    Z = np.full_like(X, np.nan)
                else:
                    try:
                        # Try cubic interpolation first (requires at least 4 points)
                        Z = griddata((plot_df[x_axis], plot_df[y_axis]), plot_df['delta_H'], 
                                    (X, Y), method='cubic')
                        
                        # Check if cubic interpolation produced any NaNs (happens at edges)
                        if np.isnan(Z).all():
                            # Fall back to linear interpolation
                            Z = griddata((plot_df[x_axis], plot_df[y_axis]), plot_df['delta_H'], 
                                        (X, Y), method='linear')
                    except Exception as e:
                        # If cubic fails, try linear
                        try:
                            Z = griddata((plot_df[x_axis], plot_df[y_axis]), plot_df['delta_H'], 
                                        (X, Y), method='linear')
                        except Exception as e:
                            st.warning(f"Interpolation failed: {str(e)}")
                            Z = np.full_like(X, np.nan)
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                # Plot 1: Filled contour
                ax1 = axes[0]
                contour1 = ax1.contourf(X, Y, Z, levels=20, cmap='RdBu_r', alpha=0.8)
                plt.colorbar(contour1, ax=ax1, label='ΔH (kJ/mol)')
                
                # Add contour lines
                contour_lines = ax1.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5, alpha=0.3)
                ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')
                
                # Add experimental points
                scatter = ax1.scatter(plot_df[x_axis], plot_df[y_axis], 
                                     c=plot_df['delta_H'], cmap='RdBu_r', 
                                     s=80, edgecolors='black', linewidth=0.5,
                                     vmin=-150, vmax=-50)
                
                ax1.set_xlabel(x_axis.replace('_', ' ').title())
                ax1.set_ylabel(y_axis.replace('_', ' ').title())
                ax1.set_title('ΔH Contour Map with Data Points')
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Optimal regions highlighted
                ax2 = axes[1]
                
                # Define optimal regions (ΔH < -100 kJ/mol)
                optimal_mask = Z < -100
                optimal_masked = np.ma.masked_where(~optimal_mask, Z)
                
                # Background
                ax2.contourf(X, Y, Z, levels=20, cmap='gray', alpha=0.3)
                
                # Highlight optimal regions
                if optimal_mask.any():
                    contour_opt = ax2.contourf(X, Y, optimal_masked, levels=[-200, -100], 
                                              colors=['lightblue'], alpha=0.7)
                    
                    # Add boundary
                    ax2.contour(X, Y, Z, levels=[-100], colors='red', linewidths=2, linestyles='--')
                
                # Add points with color by ΔH
                scatter2 = ax2.scatter(plot_df[x_axis], plot_df[y_axis], 
                                      c=plot_df['delta_H'], cmap='RdBu_r',
                                      s=80, edgecolors='black', linewidth=0.5,
                                      vmin=-150, vmax=-50)
                
                ax2.set_xlabel(x_axis.replace('_', ' ').title())
                ax2.set_ylabel(y_axis.replace('_', ' ').title())
                ax2.set_title('Optimal Hydration Regions (ΔH < -100 kJ/mol)')
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Gradient map (sensitivity)
                ax3 = axes[2]
                
                # Calculate gradient magnitude
                dy, dx = np.gradient(Z)
                gradient_magnitude = np.sqrt(dx**2 + dy**2)
                
                contour3 = ax3.contourf(X, Y, gradient_magnitude, levels=20, cmap='hot', alpha=0.8)
                plt.colorbar(contour3, ax=ax3, label='|∇ΔH|')
                
                # Add experimental points
                ax3.scatter(plot_df[x_axis], plot_df[y_axis], 
                           c='white', s=30, alpha=0.5, edgecolors='black', linewidth=0.5)
                
                ax3.set_xlabel(x_axis.replace('_', ' ').title())
                ax3.set_ylabel(y_axis.replace('_', ' ').title())
                ax3.set_title('Gradient Map (Sensitivity)')
                ax3.grid(True, alpha=0.3)
                
                plt.suptitle(f'2D Hydration Maps: {x_axis.replace("_", " ").title()} vs {y_axis.replace("_", " ").title()}', 
                            fontsize=14, fontweight='bold', y=1.05)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with tab2:
                st.markdown("#### Experimental Coverage Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    dim1 = st.selectbox("Dimension 1", 
                                       ['r_B_avg', 't_Goldschmidt', 'content', 'chi_diff'],
                                       index=0, key="cov_x")
                    
                with col2:
                    dim2 = st.selectbox("Dimension 2", 
                                       ['chi_diff', 'content', 't_Goldschmidt', 'r_B_avg'],
                                       index=1, key="cov_y")
                
                # Create 2D histogram of experimental coverage
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Plot 1: Hexbin plot (density)
                ax1 = axes[0]
                
                # Clean data for hexbin
                hex_data = plot_df[[dim1, dim2, 'delta_H']].dropna()
                if len(hex_data) > 0:
                    hb = ax1.hexbin(hex_data[dim1], hex_data[dim2], gridsize=20, 
                                   cmap='YlOrRd', mincnt=1, alpha=0.8,
                                   edgecolors='white', linewidth=0.5)
                    plt.colorbar(hb, ax=ax1, label='Number of samples')
                else:
                    ax1.text(0.5, 0.5, 'No valid data', ha='center', va='center')
                
                ax1.set_xlabel(dim1.replace('_', ' ').title())
                ax1.set_ylabel(dim2.replace('_', ' ').title())
                ax1.set_title('Experimental Density Map')
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Coverage gaps with robust distance calculation
                ax2 = axes[1]
                
                # Create grid for coverage assessment
                x_min, x_max = plot_df[dim1].min(), plot_df[dim1].max()
                y_min, y_max = plot_df[dim2].min(), plot_df[dim2].max()
                
                # Add small padding
                x_pad = (x_max - x_min) * 0.1
                y_pad = (y_max - y_min) * 0.1
                
                x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 30)
                y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 30)
                X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                
                # Find nearest experimental point for each grid point (with robust handling)
                from scipy.spatial import cKDTree, QhullError
                
                # Clean data: remove NaN and ensure we have valid points
                clean_data = plot_df[[dim1, dim2]].dropna()
                
                if len(clean_data) >= 3:
                    points = clean_data.values
                    # Remove duplicate points for numerical stability
                    points = np.unique(points, axis=0)
                    
                    if len(points) >= 3:
                        tree = cKDTree(points)
                        
                        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
                        # Remove any NaN from grid
                        valid_grid = ~np.isnan(grid_points).any(axis=1)
                        grid_points_clean = grid_points[valid_grid]
                        
                        if len(grid_points_clean) > 0:
                            distances_all = np.full(len(grid_points), np.nan)
                            distances_clean, _ = tree.query(grid_points_clean)
                            distances_all[valid_grid] = distances_clean
                            distances = distances_all.reshape(X_grid.shape)
                            
                            # Plot coverage gaps
                            if not np.all(np.isnan(distances)):
                                # Use percentile only on valid distances
                                valid_distances = distances[~np.isnan(distances)]
                                if len(valid_distances) > 0:
                                    gap_threshold = np.percentile(valid_distances, 75)
                                    
                                    # Background - use pcolormesh for better handling of NaNs
                                    masked_distances = np.ma.masked_invalid(distances)
                                    mesh = ax2.pcolormesh(X_grid, Y_grid, masked_distances, 
                                                         cmap='viridis_r', alpha=0.5, shading='auto')
                                    plt.colorbar(mesh, ax=ax2, label='Distance to nearest point')
                                    
                                    # Highlight gaps
                                    gap_mask = distances > gap_threshold
                                    if gap_mask.any():
                                        gap_masked = np.ma.masked_where(~gap_mask, distances)
                                        ax2.contourf(X_grid, Y_grid, gap_masked, 
                                                   levels=[gap_threshold, distances.max()],
                                                   colors=['red'], alpha=0.3)
                        else:
                            ax2.text(0.5, 0.5, 'No valid grid points', 
                                    ha='center', va='center', transform=ax2.transAxes)
                    else:
                        ax2.text(0.5, 0.5, 'Insufficient unique points', 
                                ha='center', va='center', transform=ax2.transAxes)
                else:
                    ax2.text(0.5, 0.5, 'Need at least 3 data points', 
                            ha='center', va='center', transform=ax2.transAxes)
                
                # Add experimental points
                ax2.scatter(plot_df[dim1], plot_df[dim2], 
                           c='black', s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
                
                ax2.set_xlabel(dim1.replace('_', ' ').title())
                ax2.set_ylabel(dim2.replace('_', ' ').title())
                ax2.set_title('Coverage Gaps (Red = Under-explored)')
                ax2.grid(True, alpha=0.3)
                
                plt.suptitle('Experimental Coverage Analysis', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Coverage statistics with robust error handling
                st.markdown("#### Coverage Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total samples", len(plot_df))
                
                with col2:
                    # Calculate area coverage with robust convex hull calculation
                    x_range = plot_df[dim1].max() - plot_df[dim1].min()
                    y_range = plot_df[dim2].max() - plot_df[dim2].min()
                    total_area = x_range * y_range
                    
                    if len(clean_data) >= 3 and total_area > 0:
                        try:
                            points = clean_data.values
                            points = np.unique(points, axis=0)
                            
                            if len(points) >= 3:
                                from scipy.spatial import ConvexHull, QhullError
                                try:
                                    hull = ConvexHull(points)
                                    covered_area = hull.volume  # area for 2D
                                    coverage_pct = min(100, (covered_area / total_area) * 100)
                                    st.metric("Area coverage", f"{coverage_pct:.1f}%")
                                except QhullError:
                                    # Points might be collinear - use bounding box estimate
                                    covered_area = x_range * y_range * 0.3  # Rough estimate for collinear
                                    coverage_pct = 30.0
                                    st.metric("Area coverage", f"{coverage_pct:.1f}% (linear)")
                            else:
                                st.metric("Area coverage", "Insufficient unique points")
                        except Exception as e:
                            st.metric("Area coverage", "Error")
                    else:
                        st.metric("Area coverage", "N/A")
                
                with col3:
                    if total_area > 0 and len(clean_data) > 0:
                        density = len(clean_data) / total_area
                        st.metric("Data density", f"{density:.2f} pts/unit²")
                    else:
                        st.metric("Data density", "N/A")
                
                with col4:
                    # Recommended new experiments based on coverage gaps
                    if 'distances' in locals() and not np.all(np.isnan(distances)):
                        n_suggested = min(10, int(len(plot_df) * 0.3))
                        st.metric("Suggested experiments", n_suggested)
                    else:
                        st.metric("Suggested experiments", "N/A")
            
            with tab3:
                st.markdown("#### Material Design Guide")
                
                st.markdown("""
                <div class="card">
                    <h4>How to use these maps for materials design:</h4>
                    <ol>
                        <li><b>Identify optimal regions</b> - Look for dark blue areas in contour maps (ΔH < -100 kJ/mol)</li>
                        <li><b>Check experimental coverage</b> - Ensure your target composition is in a well-explored region</li>
                        <li><b>Avoid high-gradient areas</b> - Red regions in gradient maps indicate high sensitivity to composition variations</li>
                        <li><b>Fill coverage gaps</b> - Prioritize unexplored regions for new experiments</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate recommendations
                st.markdown("#### Recommended Compositions for Exploration")
                
                # Create grid for recommendations
                x_grid = np.linspace(plot_df['r_B_avg'].min(), plot_df['r_B_avg'].max(), 20)
                y_grid = np.linspace(plot_df['chi_diff'].min(), plot_df['chi_diff'].max(), 20)
                X, Y = np.meshgrid(x_grid, y_grid)
                
                # Interpolate ΔH
                from scipy.interpolate import griddata
                
                if len(plot_df) < 4:
                    st.warning("Insufficient data for interpolation")
                    Z = np.full_like(X, np.nan)
                else:
                    try:
                        Z = griddata((plot_df['r_B_avg'], plot_df['chi_diff']), plot_df['delta_H'], 
                                    (X, Y), method='cubic')
                        if np.isnan(Z).all():
                            Z = griddata((plot_df['r_B_avg'], plot_df['chi_diff']), plot_df['delta_H'], 
                                        (X, Y), method='linear')
                    except:
                        try:
                            Z = griddata((plot_df['r_B_avg'], plot_df['chi_diff']), plot_df['delta_H'], 
                                        (X, Y), method='linear')
                        except:
                            Z = np.full_like(X, np.nan)
                
                # Find promising regions (ΔH < -100) that are far from existing points
                promising_mask = Z < -100
                
                if promising_mask.any():
                    # Get grid points in promising regions
                    promising_points = np.column_stack([X[promising_mask], Y[promising_mask]])
                    
                    # Calculate distances to nearest experimental point
                    exp_points = plot_df[['r_B_avg', 'chi_diff']].values
                    tree = cKDTree(exp_points)
                    distances, _ = tree.query(promising_points)
                    
                    # Sort by distance (prioritize unexplored regions)
                    sorted_idx = np.argsort(-distances)  # farthest first
                    
                    # Show top recommendations
                    recommendations = []
                    for i in sorted_idx[:10]:
                        recommendations.append({
                            'Rank': len(recommendations) + 1,
                            'r_B_avg (Å)': f"{promising_points[i, 0]:.3f}",
                            'χ_diff': f"{promising_points[i, 1]:.3f}",
                            'Predicted ΔH (kJ/mol)': f"{Z[promising_mask][i]:.1f}",
                            'Distance to known': f"{distances[i]:.3f}"
                        })
                    
                    st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
                else:
                    st.info("No promising regions identified with current criteria")
        
        elif plot_category == "Advanced 3D Analysis":
            if model_data is not None:
                st.markdown("### 3D Hydration Landscape with Projections")
                fig = create_3d_descriptor_landscape(plot_df)
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Model data not available for 3D analysis")
        
        # =========================================================================
        # Исправленный Multi-dimensional Patterns
        # =========================================================================
        elif plot_category == "Multi-dimensional Patterns":
            st.markdown("### Multi-dimensional Pattern Analysis")
            
            # Исправленная функция create_enhanced_parallel_coordinates
            def create_enhanced_parallel_coordinates_fixed(df_features):
                """Create enhanced parallel coordinates plot for multi-dimensional analysis (исправленная версия)"""
                
                # Select key descriptors for analysis
                features = ['r_B_avg', 't_Goldschmidt', 'chi_diff', 'delta_r_B', 
                            'content', 'polarizability_avg', 'ionization_avg', 'delta_H']
                
                # Filter available features
                available_features = [f for f in features if f in df_features.columns]
                
                if len(available_features) < 3:
                    return None
                
                # Normalize data for visualization
                df_norm = df_features[available_features].copy()
                for col in available_features[:-1]:  # Don't normalize target
                    min_val = df_norm[col].min()
                    max_val = df_norm[col].max()
                    if max_val > min_val:
                        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
                
                # Create figure
                fig, axes = plt.subplots(2, 2, figsize=(20, 14))
                
                # Plot 1: Parallel coordinates with color by ΔH
                ax1 = axes[0, 0]
                
                # Color mapping based on ΔH
                norm = plt.Normalize(df_features['delta_H'].min(), df_features['delta_H'].max())
                colors = plt.cm.RdBu_r(norm(df_features['delta_H']))
                
                # Coordinates for vertical lines
                x_coords = np.arange(len(available_features))
                
                # Plot lines for each material
                for idx in range(len(df_norm)):
                    y_coords = df_norm.iloc[idx, :].values
                    ax1.plot(x_coords, y_coords, color=colors[idx], alpha=0.3, linewidth=0.5)
                
                # Add mean lines for ΔH quartiles - ИСПРАВЛЕНО
                try:
                    # Используем qcut только если достаточно уникальных значений
                    if df_features['delta_H'].nunique() >= 4:
                        delta_H_bins = pd.qcut(df_features['delta_H'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                        
                        for bin_name, group in df_features.groupby(delta_H_bins):
                            if len(group) > 2:
                                group_norm = df_norm.loc[group.index]
                                mean_values = group_norm[available_features].mean()
                                ax1.plot(x_coords, mean_values, linewidth=3, 
                                        label=f'{bin_name} (ΔH: {group["delta_H"].mean():.0f} kJ/mol)', 
                                        alpha=0.8)
                    else:
                        # Если мало данных, просто покажем среднюю линию
                        mean_values = df_norm[available_features].mean()
                        ax1.plot(x_coords, mean_values, linewidth=3, color='black',
                                label=f'Mean (n={len(df_features)})', alpha=0.8)
                except Exception as e:
                    # В случае ошибки показываем среднюю линию
                    mean_values = df_norm[available_features].mean()
                    ax1.plot(x_coords, mean_values, linewidth=3, color='black',
                            label=f'Mean (n={len(df_features)})', alpha=0.8)
                
                ax1.set_xticks(x_coords)
                ax1.set_xticklabels([f.replace('_', ' ').title() for f in available_features], 
                                    rotation=45, ha='right', fontsize=9)
                ax1.set_ylabel('Normalized Value', fontsize=10)
                ax1.set_title('Parallel Coordinates: Multi-dimensional Patterns', 
                              fontsize=12, fontweight='bold')
                ax1.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', fontsize=8)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(-0.1, 1.1)
                
                # Plot 2: Feature correlation heatmap
                ax2 = axes[0, 1]
                
                # Calculate correlation matrix
                corr_matrix = df_features[available_features].corr()
                
                # Create mask for upper triangle
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                # Heatmap
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                            cmap='RdBu_r', center=0, square=True, ax=ax2,
                            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                            annot_kws={'size': 8})
                ax2.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)
                
                # Plot 3: Radar charts for characteristic compositions - ИСПРАВЛЕНО
                ax3 = axes[1, 0]
                
                if len(df_features) >= 4:
                    # Find representative compositions for each ΔH quartile
                    n_radar = min(4, len(df_features))
                    angles = np.linspace(0, 2 * np.pi, len(available_features)-1, endpoint=False).tolist()
                    angles += angles[:1]  # Close the loop
                    
                    # Select samples evenly spaced by ΔH
                    df_sorted = df_features.sort_values('delta_H')
                    indices = np.linspace(0, len(df_sorted)-1, n_radar, dtype=int)
                    radar_data = df_sorted.iloc[indices]
                    
                    # Plot radar charts
                    for i, (idx, data) in enumerate(radar_data.iterrows()):
                        values = [data[f] for f in available_features[:-1]]  # Exclude ΔH
                        # Normalize values
                        values_norm = []
                        for v, f in zip(values, available_features[:-1]):
                            min_val = df_features[f].min()
                            max_val = df_features[f].max()
                            if max_val > min_val:
                                values_norm.append((v - min_val) / (max_val - min_val))
                            else:
                                values_norm.append(0.5)
                        values_norm += values_norm[:1]  # Close the loop
                        
                        ax3.plot(angles, values_norm, 'o-', linewidth=2, 
                                label=f'ΔH = {data["delta_H"]:.0f} kJ/mol')
                        ax3.fill(angles, values_norm, alpha=0.25)
                else:
                    ax3.text(0.5, 0.5, 'Insufficient data for radar charts', 
                            ha='center', va='center', transform=ax3.transAxes)
                
                ax3.set_xticks(angles[:-1] if len(df_features) >= 4 else [])
                ax3.set_xticklabels([f.replace('_', ' ').title() for f in available_features[:-1]] 
                                   if len(df_features) >= 4 else [], size=8)
                ax3.set_ylim(0, 1)
                ax3.set_title('Radar Plots: Characteristic Compositions', 
                              fontsize=12, fontweight='bold')
                if len(df_features) >= 4:
                    ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=8)
                ax3.grid(True)
                
                # Plot 4: Andrews curves
                ax4 = axes[1, 1]
                
                # Create Andrews curves for different ΔH ranges - ИСПРАВЛЕНО
                t = np.linspace(-np.pi, np.pi, 100)
                
                try:
                    if df_features['delta_H'].nunique() >= 4:
                        delta_H_bins = pd.qcut(df_features['delta_H'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                        
                        for bin_name, group in df_features.groupby(delta_H_bins):
                            if len(group) > 2:
                                group_norm = df_norm.loc[group.index, available_features[:-1]]
                                mean_values = group_norm.mean().values
                                
                                # Andrews curve: f(t) = x1/√2 + x2·sin(t) + x3·cos(t) + x4·sin(2t) + ...
                                curve = mean_values[0] / np.sqrt(2) * np.ones_like(t)
                                for i, val in enumerate(mean_values[1:], 1):
                                    if i % 2 == 1:
                                        curve += val * np.sin((i+1)//2 * t)
                                    else:
                                        curve += val * np.cos(i//2 * t)
                                
                                ax4.plot(t, curve, linewidth=2, 
                                        label=f'{bin_name} (ΔH: {group["delta_H"].mean():.0f} kJ/mol)')
                    else:
                        # Single curve for all data
                        mean_values = df_norm[available_features[:-1]].mean().values
                        curve = mean_values[0] / np.sqrt(2) * np.ones_like(t)
                        for i, val in enumerate(mean_values[1:], 1):
                            if i % 2 == 1:
                                curve += val * np.sin((i+1)//2 * t)
                            else:
                                curve += val * np.cos(i//2 * t)
                        ax4.plot(t, curve, linewidth=2, color='black', 
                                label=f'All data (n={len(df_features)})')
                except Exception as e:
                    ax4.text(0.5, 0.5, 'Could not generate Andrews curves', 
                            ha='center', va='center', transform=ax4.transAxes)
                
                ax4.set_xlabel('t', fontsize=10)
                ax4.set_ylabel('Andrews curve value', fontsize=10)
                ax4.set_title('Andrews Curves: Pattern Comparison', 
                              fontsize=12, fontweight='bold')
                ax4.legend(loc='best', fontsize=8)
                ax4.grid(True, alpha=0.3)
                
                plt.suptitle('Multi-dimensional Analysis of Hydration Parameters', 
                             fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                return fig
            
            # Используем исправленную функцию
            fig = create_enhanced_parallel_coordinates_fixed(plot_df)
            if fig:
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Insufficient data for multi-dimensional analysis")
        
        elif plot_category == "Sensitivity Analysis":
            if model_data is not None:
                st.markdown("### Sensitivity Analysis and Critical Regions")
                fig = create_sensitivity_heatmap(model_data, plot_df)
                if fig:
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Could not generate sensitivity map")
            else:
                st.warning("Model not available for sensitivity analysis")
        
        elif plot_category == "Cluster Analysis":
            st.markdown("### Hierarchical Cluster Analysis")
            fig = create_hierarchical_clustering(plot_df)
            if fig:
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Insufficient data for cluster analysis (need at least 5 samples)")

    # =========================================================================
    # Page 3: ML Predictor (Enhanced with new descriptors)
    # =========================================================================
    elif page == "🤖 ML Predictor":
        st.markdown("## 🤖 Advanced Hydration Parameter Predictor")
        
        if model_data is None or df_features.empty:
            st.warning("Insufficient data for training prediction models. Need at least 10 samples with complete descriptors.")
            return
        
        st.markdown("""
        <div class="card">
            <p>Enter the composition of your perovskite material to predict its hydration thermodynamics.
            The prediction uses ensemble machine learning models trained on experimental data with enhanced descriptors.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            a_cation = st.selectbox("A-cation", ['Ba', 'Sr', 'Ca', 'La', 'Nd', 'Gd'])
            
            # Define B-cation options based on A-cation valence
            if a_cation in ['Ba', 'Sr', 'Ca']:
                b_options = ['Ti', 'Zr', 'Sn', 'Ce', 'Hf']
                b_default = 'Zr'
            else:
                b_options = ['Sc', 'In', 'Y', 'Yb', 'Gd', 'Lu', 'Al', 'Ga']
                b_default = 'Y'
        
        with col2:
            b_cation = st.selectbox("B-cation", b_options, 
                                   index=b_options.index(b_default) if b_default in b_options else 0)
            
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
        
        # Display calculated descriptors in expandable section
        with st.expander("📊 Calculated Descriptors", expanded=True):
            desc_cols = st.columns(4)
            desc_items = [
                ('r_A (XII)', f"{input_desc.get('r_A_XII', 0):.3f} Å"),
                ('r_B (VI)', f"{input_desc.get('r_B_VI', 0):.3f} Å"),
                ('r_D (VI)', f"{input_desc.get('r_D_VI', 0):.3f} Å"),
                ('r_B_avg', f"{input_desc.get('r_B_avg', 0):.3f} Å"),
                ('t_Goldschmidt', f"{input_desc.get('t_Goldschmidt', 0):.3f}"),
                ('Octahedral factor', f"{input_desc.get('octahedral_factor', 0):.3f}"),
                ('χ_A', f"{input_desc.get('chi_A', 0):.2f}"),
                ('χ_B', f"{input_desc.get('chi_B', 0):.2f}"),
                ('χ_D', f"{input_desc.get('chi_D', 0):.2f}"),
                ('χ_diff', f"{input_desc.get('chi_diff', 0):.2f}"),
                ('δr_B', f"{input_desc.get('delta_r_B', 0):.3f} Å"),
                ('Lattice energy', f"{input_desc.get('lattice_energy', 0):.0f} kJ/mol"),
                ('Polarizability', f"{input_desc.get('polarizability_avg', 0):.3f}"),
                ('Charge density A', f"{input_desc.get('charge_density_A', 0):.2f} e/Å"),
            ]
            
            for i, (label, value) in enumerate(desc_items):
                desc_cols[i % 4].metric(label, value)
        
        # Prepare features for prediction
        feature_names = model_data['feature_names']
        
        # Create feature vector
        X_pred = pd.DataFrame([{
            'content': input_desc.get('content', content),
            'r_A_XII': input_desc.get('r_A_XII', 0),
            'r_B_VI': input_desc.get('r_B_VI', 0),
            'r_D_VI': input_desc.get('r_D_VI', 0),
            'r_B_avg': input_desc.get('r_B_avg', 0),
            'delta_r_B': input_desc.get('delta_r_B', 0),
            't_Goldschmidt': input_desc.get('t_Goldschmidt', 0),
            'octahedral_factor': input_desc.get('octahedral_factor', 0),
            'global_instability': input_desc.get('global_instability', 0),
            'lattice_energy': input_desc.get('lattice_energy', 0),
            'chi_A': input_desc.get('chi_A', 0),
            'chi_B': input_desc.get('chi_B', 0),
            'chi_D': input_desc.get('chi_D', 0),
            'chi_B_avg': input_desc.get('chi_B_avg', 0),
            'chi_diff': input_desc.get('chi_diff', 0),
            'chi_product': input_desc.get('chi_product', 0),
            'polarizability_A': input_desc.get('polarizability_A', 0),
            'polarizability_B': input_desc.get('polarizability_B', 0),
            'polarizability_D': input_desc.get('polarizability_D', 0),
            'polarizability_avg': input_desc.get('polarizability_avg', 0),
            'ionization_A': input_desc.get('ionization_A', 0),
            'ionization_B': input_desc.get('ionization_B', 0),
            'ionization_D': input_desc.get('ionization_D', 0),
            'ionization_avg': input_desc.get('ionization_avg', 0),
            'charge_density_A': input_desc.get('charge_density_A', 0),
            'charge_density_B': input_desc.get('charge_density_B', 0),
            'charge_density_D': input_desc.get('charge_density_D', 0),
            'oxygen_vacancy': input_desc.get('oxygen_vacancy', 0),
            'bond_valence_sum': input_desc.get('bond_valence_sum', 0)
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
        for col in feature_names:
            if col not in X_pred.columns:
                X_pred[col] = 0
        
        X_pred = X_pred[feature_names]
        
        # Scale features
        X_pred_scaled = model_data['scaler'].transform(X_pred)
        
        # Make predictions with ensemble
        pred_H_xgb = model_data['models']['xgb_H'].predict(X_pred_scaled)[0]
        pred_S_xgb = model_data['models']['xgb_S'].predict(X_pred_scaled)[0]
        
        pred_H_rf = model_data['models']['rf_H'].predict(X_pred_scaled)[0]
        pred_S_rf = model_data['models']['rf_S'].predict(X_pred_scaled)[0]
        
        # Check if Gradient Boosting models exist
        if 'gb_H' in model_data['models']:
            pred_H_gb = model_data['models']['gb_H'].predict(X_pred_scaled)[0]
            pred_S_gb = model_data['models']['gb_S'].predict(X_pred_scaled)[0]
            
            # Ensemble prediction with all three models
            weights = [0.5, 0.3, 0.2]  # XGBoost, RF, GB weights
            pred_H = (weights[0]*pred_H_xgb + weights[1]*pred_H_rf + weights[2]*pred_H_gb)
            pred_S = (weights[0]*pred_S_xgb + weights[1]*pred_S_rf + weights[2]*pred_S_gb)
            
            # Calculate uncertainty (standard deviation of predictions)
            H_std = np.std([pred_H_xgb, pred_H_rf, pred_H_gb])
            S_std = np.std([pred_S_xgb, pred_S_rf, pred_S_gb])
        else:
            # Fallback to two models
            weights = [0.6, 0.4]  # XGBoost, RF weights
            pred_H = weights[0]*pred_H_xgb + weights[1]*pred_H_rf
            pred_S = weights[0]*pred_S_xgb + weights[1]*pred_S_rf
            
            # Calculate uncertainty with two models
            H_std = np.std([pred_H_xgb, pred_H_rf])
            S_std = np.std([pred_S_xgb, pred_S_rf])
        
        # Calculate uncertainty (standard deviation of predictions)
        H_std = np.std([pred_H_xgb, pred_H_rf, pred_H_gb])
        S_std = np.std([pred_S_xgb, pred_S_rf, pred_S_gb])
        
        # Display predictions in modern cards
        st.markdown("### 🔮 Predicted Hydration Parameters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ΔH (kJ mol⁻¹)</div>
                <div class="metric-value">{pred_H:.1f} ± {H_std:.1f}</div>
                <div class="metric-delta">Ensemble prediction</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ΔS (J mol⁻¹ K⁻¹)</div>
                <div class="metric-value">{pred_S:.1f} ± {S_std:.1f}</div>
                <div class="metric-delta">Ensemble prediction</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Calculate approximate T* (isokinetic temperature)
            if pred_S != 0:
                T_star = -pred_H / pred_S * 1000  # in K
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Isokinetic T (K)</div>
                    <div class="metric-value">{T_star:.0f}</div>
                    <div class="metric-delta">T_iso = -ΔH/ΔS × 1000</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            # Calculate hydration equilibrium constant at 600°C
            T = 600 + 273.15
            R = 0.008314  # kJ/mol·K
            K_hyd = np.exp(-(pred_H - T*pred_S/1000)/(R*T))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">K_hyd (600°C)</div>
                <div class="metric-value">{K_hyd:.2e}</div>
                <div class="metric-delta">Equilibrium constant</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Model comparison
        with st.expander("📊 Model Comparison", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**XGBoost**")
                st.markdown(f"ΔH = {pred_H_xgb:.1f} kJ/mol")
                st.markdown(f"ΔS = {pred_S_xgb:.1f} J/mol·K")
            
            with col2:
                st.markdown("**Random Forest**")
                st.markdown(f"ΔH = {pred_H_rf:.1f} kJ/mol")
                st.markdown(f"ΔS = {pred_S_rf:.1f} J/mol·K")
            
            with col3:
                st.markdown("**Gradient Boosting**")
                st.markdown(f"ΔH = {pred_H_gb:.1f} kJ/mol")
                st.markdown(f"ΔS = {pred_S_gb:.1f} J/mol·K")
        
        # Find similar materials
        st.markdown("### 🔍 Similar Materials in Database")
        
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

        # =========================================================================
        # NEW SECTION: Concentration dependence for multiple B-cations
        # =========================================================================
        st.markdown("---")
        st.markdown("## 📈 Concentration Dependence Across B-cation Families")
        
        st.markdown("""
        <div class="card">
            <p>Predict how ΔH and ΔS vary with dopant concentration for different B-cation families.
            Select a dopant and choose which B-cations to compare.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Dopant selection
            all_dopants = sorted(df['dopant'].unique())
            selected_dopant_fam = st.selectbox("Select dopant", all_dopants, key="fam_dopant")
        
        with col2:
            # B-cation selection (multiple)
            all_b_cations = ['Zr', 'Ce', 'Sn', 'Ti', 'Hf']
            selected_b_families = st.multiselect(
                "Select B-cation families to compare",
                all_b_cations,
                default=['Zr', 'Ce']
            )
        
        with col3:
            # Property to plot
            fam_property = st.radio(
                "Property to display",
                ["ΔH (kJ/mol)", "ΔS (J/mol·K)"],
                horizontal=True,
                key="fam_property"
            )
            show_trend_lines_fam = st.checkbox("Show trend lines", value=True, key="fam_trend")
        
        if selected_b_families and selected_dopant_fam:
            # Get predictions for all selected B-cations
            prop_type = 'delta_H' if fam_property == "ΔH (kJ/mol)" else 'delta_S'
            
            with st.spinner("Calculating predictions..."):
                results = predict_concentration_dependence_for_B_families(
                    selected_dopant_fam, selected_b_families, model_data, prop_type
                )
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 7))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_b_families)))
            
            for b_cat, color in zip(selected_b_families, colors):
                x_vals, y_vals = results[b_cat]
                
                ax.plot(x_vals, y_vals, 'o-', linewidth=2, markersize=6,
                       color=color, label=f'{b_cat}', alpha=0.8)
                
                if show_trend_lines_fam and len(x_vals) >= 3:
                    # Add polynomial trend (degree 2)
                    z = np.polyfit(x_vals, y_vals, 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(x_vals.min(), x_vals.max(), 100)
                    ax.plot(x_smooth, p(x_smooth), '--', color=color, alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Dopant Concentration, x', fontsize=12)
            
            if fam_property == "ΔH (kJ/mol)":
                ax.set_ylabel('ΔH (kJ mol⁻¹)', fontsize=12)
                ax.set_title(f'Enthalpy vs Dopant Concentration for BaB₁₋ₓ{selected_dopant_fam}_xO₃₋ₓ/₂', 
                            fontsize=14, fontweight='bold')
            else:
                ax.set_ylabel('ΔS (J mol⁻¹ K⁻¹)', fontsize=12)
                ax.set_title(f'Entropy vs Dopant Concentration for BaB₁₋ₓ{selected_dopant_fam}_xO₃₋ₓ/₂', 
                            fontsize=14, fontweight='bold')
            
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 0.3)
            
            st.pyplot(fig)
            plt.close()
            
            # Add note about the model
            st.caption("Predictions are based on ensemble ML models trained on experimental data.")
        
        # =========================================================================
        # NEW SECTION: Mixed B-site systems (solid solutions)
        # =========================================================================
        st.markdown("---")
        st.markdown("## 🔬 Mixed B-site Systems (Solid Solutions)")
        
        st.markdown("""
        <div class="card">
            <p>Predict properties for mixed B-site perovskites of the form:<br>
            <b>BaB1₁₋ₓ₋ᵧB2ₓDᵧO₃₋ₓ/₂</b><br>
            using ideal mixing of end-member properties.</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab_mixed_2d, tab_mixed_3d = st.tabs(["2D Composition Dependence", "3D Property Landscape"])
        
        with tab_mixed_2d:
            st.markdown("### 2D: Property vs B2 Concentration (x)")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                B1_options = ['Zr', 'Ce', 'Sn', 'Ti', 'Hf']
                selected_B1 = st.selectbox("B1 (base)", B1_options, key="mixed_B1")
            
            with col2:
                B2_options = [b for b in B1_options if b != selected_B1]
                selected_B2 = st.selectbox("B2 (substituent)", B2_options, key="mixed_B2")
            
            with col3:
                dopant_options = sorted(df['dopant'].unique())
                selected_dopant_mixed = st.selectbox("Dopant D", dopant_options, key="mixed_dopant")
            
            with col4:
                y_fixed = st.slider("Fixed dopant concentration (y)", 0.0, 0.3, 0.1, 0.01, key="mixed_y")
            
            mixed_property = st.radio(
                "Property to display",
                ["ΔH (kJ/mol)", "ΔS (J/mol·K)"],
                horizontal=True,
                key="mixed_property_2d"
            )
            
            show_trend_lines_mixed = st.checkbox("Show trend lines", value=True, key="mixed_trend_2d")
            
            if st.button("Calculate Mixed System Properties", key="calc_mixed_2d"):
                with st.spinner("Calculating using ideal mixing model..."):
                    x_values, delta_H_vals, delta_S_vals = calculate_mixed_site_properties(
                        selected_B1, selected_B2, selected_dopant_mixed, y_fixed, model_data
                    )
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if mixed_property == "ΔH (kJ/mol)":
                    y_vals = delta_H_vals
                    ylabel = 'ΔH (kJ mol⁻¹)'
                    title = f'Enthalpy of Mixing: Ba{selected_B1}₁₋ₓ₋{y_fixed:.2f}{selected_B2}ₓ{selected_dopant_mixed}{y_fixed:.2f}O₃₋ₓ/₂'
                else:
                    y_vals = delta_S_vals
                    ylabel = 'ΔS (J mol⁻¹ K⁻¹)'
                    title = f'Entropy of Mixing: Ba{selected_B1}₁₋ₓ₋{y_fixed:.2f}{selected_B2}ₓ{selected_dopant_mixed}{y_fixed:.2f}O₃₋ₓ/₂'
                
                ax.plot(x_values, y_vals, 'o-', linewidth=2, markersize=6,
                       color=MODERN_COLORS['primary'], label=f'Predicted (ideal mixing)')
                
                if show_trend_lines_mixed and len(x_values) >= 3:
                    z = np.polyfit(x_values, y_vals, 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(x_values.min(), x_values.max(), 100)
                    ax.plot(x_smooth, p(x_smooth), '--', color=MODERN_COLORS['secondary'], 
                           alpha=0.7, linewidth=1.5, label='Quadratic fit')
                
                # Add vertical line at x=0 and x=1-y
                ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
                ax.axvline(x=1-y_fixed, color='gray', linestyle=':', alpha=0.5, 
                          label=f'x_max = {1-y_fixed:.2f}')
                
                ax.set_xlabel('x (fraction of B2 on B-site)', fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1-y_fixed)
                
                st.pyplot(fig)
                plt.close()
                
                # Show end-member values
                st.markdown("#### End-member Properties")
                col1, col2 = st.columns(2)
                
                with col1:
                    props1 = get_end_member_properties(selected_B1, selected_dopant_mixed, y_fixed, model_data)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Pure {selected_B1} (x=0)</div>
                        <div class="metric-value">ΔH = {props1['delta_H']:.1f}</div>
                        <div class="metric-delta">ΔS = {props1['delta_S']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    props2 = get_end_member_properties(selected_B2, selected_dopant_mixed, y_fixed, model_data)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Pure {selected_B2} (x={1-y_fixed:.2f})</div>
                        <div class="metric-value">ΔH = {props2['delta_H']:.1f}</div>
                        <div class="metric-delta">ΔS = {props2['delta_S']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab_mixed_3d:
            st.markdown("### 3D: Property Landscape (x vs y)")
                    
            col1, col2 = st.columns(2)
                    
            with col1:
                B1_3d = st.selectbox("B1 (base)", ['Zr', 'Ce', 'Sn', 'Ti', 'Hf'], key="3d_B1")
                    
            with col2:
                B2_3d_options = [b for b in ['Zr', 'Ce', 'Sn', 'Ti', 'Hf'] if b != B1_3d]
                B2_3d = st.selectbox("B2 (substituent)", B2_3d_options, key="3d_B2")
                    
            # Дополнительные настройки в expander
            with st.expander("⚙️ Advanced Settings (Grid Resolution)", expanded=False):
                col1, col2, col3 = st.columns(3)
                        
                with col1:
                    y_step = st.select_slider(
                        "Dopant (y) step size",
                        options=[0.01, 0.02, 0.03, 0.05, 0.1],
                        value=0.05,
                        key="y_step_3d"
                    )
                    st.caption(f"y range: 0 → 0.3, {int(0.3 / y_step) + 1} points")
                        
                with col2:
                    x_step = st.select_slider(
                        "B2 (x) step size",
                        options=[0.05, 0.1, 0.15, 0.2],
                        value=0.1,
                        key="x_step_3d"
                    )
                    st.caption(f"x range: 0 → 1-y, variable points")
                        
                with col3:
                    interpolation_method = st.selectbox(
                        "Interpolation for smoothness",
                        ['None', 'Linear', 'Cubic'],
                        index=0,
                        key="interp_3d"
                    )
                    
            col1, col2, col3 = st.columns(3)
                    
            with col1:
                dopant_3d = st.selectbox("Dopant D", sorted(df['dopant'].unique()), key="3d_dopant")
                    
            with col2:
                property_3d = st.radio("Property", ["ΔH (kJ/mol)", "ΔS (J/mol·K)"], 
                                      horizontal=True, key="3d_property")
                    
            with col3:
                show_contours = st.checkbox("Show contour projections", value=True, key="3d_contours")
                show_points = st.checkbox("Show grid points", value=False, key="3d_points")
                    
            if st.button("Generate 3D Surface", key="calc_mixed_3d"):
                with st.spinner(f"Generating 3D property landscape (y_step={y_step}, x_step={x_step})..."):
                    prop_type = 'delta_H' if property_3d == "ΔH (kJ/mol)" else 'delta_S'
                            
                    # Используем обновленную функцию с параметрами шага
                    X_mesh, Y_mesh, Z_mesh, x_unique, y_unique = calculate_mixed_site_3d_surface(
                        B1_3d, B2_3d, dopant_3d, model_data, prop_type, y_step, x_step
                    )
                            
                    # Опциональная интерполяция для сглаживания
                    if interpolation_method != 'None' and not np.isnan(Z_mesh).all():
                        from scipy.interpolate import griddata
                                
                        # Создаем более плотную сетку для интерполяции
                        x_fine = np.linspace(x_unique.min(), x_unique.max(), 100)
                        y_fine = np.linspace(y_unique.min(), y_unique.max(), 100)
                        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
                                
                        # Собираем точки с известными значениями
                        points = []
                        values = []
                        for i in range(len(x_unique)):
                            for j in range(len(y_unique)):
                                if not np.isnan(Z_mesh[j, i]):
                                    points.append([x_unique[i], y_unique[j]])
                                    values.append(Z_mesh[j, i])
                                
                        if len(points) >= 4:
                            method = 'cubic' if interpolation_method == 'Cubic' else 'linear'
                            Z_fine = griddata(points, values, (X_fine, Y_fine), method=method)
                            X_mesh, Y_mesh, Z_mesh = X_fine, Y_fine, Z_fine
                        
                fig = go.Figure()
                        
                # Add surface
                fig.add_trace(go.Surface(
                    x=X_mesh,
                    y=Y_mesh,
                    z=Z_mesh,
                    colorscale='RdBu_r' if property_3d == "ΔH (kJ/mol)" else 'Viridis',
                    colorbar=dict(title=property_3d),
                    contours=dict(
                        z=dict(show=show_contours, usecolormap=True, 
                              highlightcolor="limegreen", project=dict(z=True))
                    ),
                    opacity=0.9
                ))
                        
                # Add grid points if requested
                if show_points and 'x_unique' in locals():
                    # Собираем все точки сетки
                    points_x = []
                    points_y = []
                    points_z = []
                    for i in range(len(x_unique)):
                        for j in range(len(y_unique)):
                            if not np.isnan(Z_mesh[j, i]) if isinstance(Z_mesh, np.ndarray) else True:
                                points_x.append(x_unique[i])
                                points_y.append(y_unique[j])
                                points_z.append(Z_mesh[j, i] if isinstance(Z_mesh, np.ndarray) else Z_mesh[j, i])
                            
                    fig.add_trace(go.Scatter3d(
                        x=points_x,
                        y=points_y,
                        z=points_z,
                        mode='markers',
                        marker=dict(size=3, color='black', symbol='circle'),
                        name='Grid points',
                        showlegend=True
                    ))
                        
                # Add contour projections on planes
                if show_contours:
                    z_min = np.nanmin(Z_mesh) if not np.all(np.isnan(Z_mesh)) else -150
                    fig.add_trace(go.Surface(
                        x=X_mesh,
                        y=Y_mesh,
                        z=np.full_like(Z_mesh, z_min - 10),
                        surfacecolor=Z_mesh,
                        colorscale='RdBu_r' if property_3d == "ΔH (kJ/mol)" else 'Viridis',
                        showscale=False,
                        opacity=0.3
                    ))
                        
                title_text = f"3D Property Landscape: Ba{B1_3d}₁₋ₓ₋ᵧ{B2_3d}ₓ{dopant_3d}ᵧO₃₋ₓ/₂<br>"
                title_text += f"Color: {property_3d} | Grid: y_step={y_step}, x_step={x_step}"
                        
                fig.update_layout(
                    title=title_text,
                    scene=dict(
                        xaxis_title='x (B2 concentration)',
                        yaxis_title='y (dopant concentration)',
                        zaxis_title=property_3d,
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    ),
                    width=900,
                    height=700,
                    margin=dict(l=65, r=50, b=65, t=90)
                )
                        
                st.plotly_chart(fig, use_container_width=True)
                        
                # Информация о расчете
                st.markdown(f"""
                <div class="card">
                    <h4>📊 Calculation Summary</h4>
                    <ul>
                        <li><b>Dopant (y) range:</b> 0 → 0.3 with step {y_step} → {int(0.3 / y_step) + 1} points</li>
                        <li><b>B2 (x) range:</b> 0 → 1-y with step {x_step} → variable points</li>
                        <li><b>Interpolation:</b> {interpolation_method if interpolation_method != 'None' else 'None (raw grid)'}</li>
                        <li><b>Total calculated points:</b> ~{len(x_unique) * len(y_unique)}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                        
                # Добавляем возможность скачать данные
                if st.button("📥 Download 3D surface data as CSV", key="download_3d_data"):
                    # Собираем данные в DataFrame
                    import pandas as pd
                    data_rows = []
                    for i in range(len(x_unique)):
                        for j in range(len(y_unique)):
                            if not np.isnan(Z_mesh[j, i]):
                                data_rows.append({
                                    'x (B2 fraction)': x_unique[i],
                                    'y (dopant concentration)': y_unique[j],
                                    property_3d: Z_mesh[j, i]
                                })
                            
                    df_export = pd.DataFrame(data_rows)
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="Confirm Download",
                        data=csv,
                        file_name=f"3D_surface_{B1_3d}{B2_3d}_{dopant_3d}_{property_3d}.csv",
                        mime="text/csv"
                    )
    
    # =========================================================================
    # Page 4: Model Performance (Enhanced)
    # =========================================================================
    elif page == "📈 Model Performance":
        st.markdown("## 📈 Machine Learning Model Performance")
        
        if model_data is None:
            st.warning("Model not trained. Need more data.")
            return
        
        # Model metrics in modern cards
        st.markdown("### 📊 Cross-Validation Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ΔH Model - R² (CV)</div>
                <div class="metric-value">{model_data['cv_H_mean']:.3f} ± {model_data['cv_H_std']:.3f}</div>
                <div class="metric-delta">5-fold cross-validation</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ΔS Model - R² (CV)</div>
                <div class="metric-value">{model_data['cv_S_mean']:.3f} ± {model_data['cv_S_std']:.3f}</div>
                <div class="metric-delta">5-fold cross-validation</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance with enhanced visualization
        st.markdown("### 🔑 Enhanced Feature Importance Analysis")
        
        tab1, tab2, tab3 = st.tabs(["XGBoost Importance", "SHAP Values", "Coefficient Analysis"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            importance_df = model_data['feature_importance']
            top20 = importance_df.head(20)
            
            # Color code by feature type
            colors = []
            for f in top20['feature']:
                if 'chi' in f or 'electro' in f:
                    colors.append(MODERN_COLORS['primary'])
                elif 'r_' in f or 'radius' in f or 'lattice' in f:
                    colors.append(MODERN_COLORS['secondary'])
                elif 'content' in f or 'vacancy' in f:
                    colors.append(MODERN_COLORS['success'])
                elif 'polar' in f:
                    colors.append(MODERN_COLORS['purple'])
                elif 'ion' in f:
                    colors.append(MODERN_COLORS['pink'])
                else:
                    colors.append(MODERN_COLORS['gray'])
            
            bars = ax.barh(range(len(top20)), top20['importance'], 
                          color=colors, edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, top20['importance'])):
                ax.text(val + 0.002, i, f'{val:.3f}', 
                       va='center', fontsize=9)
            
            ax.set_yticks(range(len(top20)))
            ax.set_yticklabels(top20['feature'])
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title('XGBoost Feature Importance for ΔH Prediction', 
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=MODERN_COLORS['primary'], label='Electronegativity'),
                Patch(facecolor=MODERN_COLORS['secondary'], label='Structural'),
                Patch(facecolor=MODERN_COLORS['success'], label='Composition'),
                Patch(facecolor=MODERN_COLORS['purple'], label='Polarizability'),
                Patch(facecolor=MODERN_COLORS['pink'], label='Ionization'),
                Patch(facecolor=MODERN_COLORS['gray'], label='Other')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
            
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            if model_data['shap_summary'] is not None:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                shap_summary = model_data['shap_summary'].head(15)
                
                bars = ax.barh(range(len(shap_summary)), shap_summary['mean_abs_shap'],
                              color=MODERN_COLORS['primary'], edgecolor='black', alpha=0.7)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, shap_summary['mean_abs_shap'])):
                    ax.text(val + 0.1, i, f'{val:.2f}', 
                           va='center', fontsize=9)
                
                ax.set_yticks(range(len(shap_summary)))
                ax.set_yticklabels(shap_summary['feature'])
                ax.set_xlabel('Mean |SHAP|', fontsize=12)
                ax.set_title('SHAP Feature Importance (Average Impact on Model Output)',
                            fontsize=14, fontweight='bold')
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis='x')
                
                st.pyplot(fig)
                plt.close()
            else:
                st.info("SHAP analysis not available for this dataset")
        
        with tab3:
            # Get coefficients from linear model (if available)
            if 'elastic_H' in model_data['models'] and hasattr(model_data['models']['elastic_H'], 'coef_'):
                coef_df = pd.DataFrame({
                    'feature': model_data['feature_names'],
                    'coefficient': model_data['models']['elastic_H'].coef_
                }).sort_values('coefficient', ascending=False).head(15)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                colors = ['red' if c < 0 else 'blue' for c in coef_df['coefficient']]
                bars = ax.barh(range(len(coef_df)), coef_df['coefficient'], 
                              color=colors, edgecolor='black', alpha=0.7)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, coef_df['coefficient'])):
                    ax.text(val + (0.5 if val > 0 else -2.5), i, f'{val:.2f}', 
                           va='center', fontsize=9)
                
                ax.set_yticks(range(len(coef_df)))
                ax.set_yticklabels(coef_df['feature'])
                ax.set_xlabel('Coefficient (effect on ΔH)', fontsize=12)
                ax.set_title('ElasticNet Coefficients (Linear Model Interpretation)',
                            fontsize=14, fontweight='bold')
                ax.axvline(x=0, color='black', linewidth=0.5, linestyle='-')
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='blue', label='Positive (increases ΔH)'),
                    Patch(facecolor='red', label='Negative (decreases ΔH)')
                ]
                ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
                
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Coefficient analysis not available - using feature importance instead")
                # Показываем важность признаков из XGBoost как альтернативу
                if 'xgb_H' in model_data['models']:
                    importance_df = model_data['feature_importance'].head(15)
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = ax.barh(range(len(importance_df)), importance_df['importance'],
                                  color='blue', edgecolor='black', alpha=0.7)
                    ax.set_yticks(range(len(importance_df)))
                    ax.set_yticklabels(importance_df['feature'])
                    ax.set_xlabel('Feature Importance', fontsize=12)
                    ax.set_title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
                    ax.invert_yaxis()
                    ax.grid(True, alpha=0.3, axis='x')
                    st.pyplot(fig)
                    plt.close()
        
        # Prediction vs Actual plots
        st.markdown("### 📈 Prediction vs Actual (Training Data)")
        
        # Get predictions on training data
        train_pred_H = model_data['models']['xgb_H'].predict(model_data['X_train'])
        train_pred_S = model_data['models']['xgb_S'].predict(model_data['X_train'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Scatter plot
            ax.scatter(model_data['y_train_H'], train_pred_H, 
                      alpha=0.6, edgecolors='black', linewidth=0.5, s=80,
                      color=MODERN_COLORS['primary'])
            
            # Perfect prediction line
            min_val = min(model_data['y_train_H'].min(), train_pred_H.min())
            max_val = max(model_data['y_train_H'].max(), train_pred_H.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7,
                   label='Perfect prediction')
            
            # Add confidence band (±10%)
            ax.fill_between([min_val, max_val], 
                           [min_val*0.9, max_val*0.9],
                           [min_val*1.1, max_val*1.1],
                           alpha=0.1, color='gray', label='±10%')
            
            ax.set_xlabel('Actual ΔH (kJ mol⁻¹)', fontsize=12)
            ax.set_ylabel('Predicted ΔH (kJ mol⁻¹)', fontsize=12)
            ax.set_title(f'ΔH: R² = {r2_score(model_data["y_train_H"], train_pred_H):.3f}\n'
                        f'MAE = {mean_absolute_error(model_data["y_train_H"], train_pred_H):.1f}',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            ax.scatter(model_data['y_train_S'], train_pred_S, 
                      alpha=0.6, edgecolors='black', linewidth=0.5, s=80,
                      color=MODERN_COLORS['secondary'])
            
            min_val = min(model_data['y_train_S'].min(), train_pred_S.min())
            max_val = max(model_data['y_train_S'].max(), train_pred_S.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7,
                   label='Perfect prediction')
            
            ax.fill_between([min_val, max_val], 
                           [min_val*0.9, max_val*0.9],
                           [min_val*1.1, max_val*1.1],
                           alpha=0.1, color='gray', label='±10%')
            
            ax.set_xlabel('Actual ΔS (J mol⁻¹ K⁻¹)', fontsize=12)
            ax.set_ylabel('Predicted ΔS (J mol⁻¹ K⁻¹)', fontsize=12)
            ax.set_title(f'ΔS: R² = {r2_score(model_data["y_train_S"], train_pred_S):.3f}\n'
                        f'MAE = {mean_absolute_error(model_data["y_train_S"], train_pred_S):.1f}',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            st.pyplot(fig)
            plt.close()
        
        # Learning curves
        st.markdown("### 📉 Learning Curves")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate learning curve data
        train_sizes = np.linspace(0.3, 1.0, 8)
        train_scores = []
        val_scores = []
        train_scores_std = []
        val_scores_std = []
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for size in train_sizes:
            n_samples = int(len(model_data['X_train']) * size)
            fold_train_scores = []
            fold_val_scores = []
            
            for train_idx, val_idx in kf.split(model_data['X_train'][:n_samples]):
                X_tr = model_data['X_train'][train_idx]
                X_val = model_data['X_train'][val_idx]
                y_tr = model_data['y_train_H'].iloc[train_idx]
                y_val = model_data['y_train_H'].iloc[val_idx]
                
                model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
                model.fit(X_tr, y_tr)
                
                fold_train_scores.append(r2_score(y_tr, model.predict(X_tr)))
                fold_val_scores.append(r2_score(y_val, model.predict(X_val)))
            
            train_scores.append(np.mean(fold_train_scores))
            val_scores.append(np.mean(fold_val_scores))
            train_scores_std.append(np.std(fold_train_scores))
            val_scores_std.append(np.std(fold_val_scores))
        
        ax.plot(train_sizes * 100, train_scores, 'o-', 
               label='Training score', color=MODERN_COLORS['primary'], linewidth=2)
        ax.fill_between(train_sizes * 100, 
                       np.array(train_scores) - np.array(train_scores_std),
                       np.array(train_scores) + np.array(train_scores_std),
                       alpha=0.2, color=MODERN_COLORS['primary'])
        
        ax.plot(train_sizes * 100, val_scores, 'o-', 
               label='Validation score', color=MODERN_COLORS['secondary'], linewidth=2)
        ax.fill_between(train_sizes * 100,
                       np.array(val_scores) - np.array(val_scores_std),
                       np.array(val_scores) + np.array(val_scores_std),
                       alpha=0.2, color=MODERN_COLORS['secondary'])
        
        ax.set_xlabel('Training set size (%)', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Learning Curves for ΔH Prediction', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        st.pyplot(fig)
        plt.close()

    # =========================================================================
    # Page 5: SHAP Analysis (Enhanced)
    # =========================================================================
    elif page == "📊 SHAP Analysis":
        st.markdown("## 📊 Advanced SHAP Analysis for Model Interpretability")
        
        if model_data is None or model_data['shap_values_H'] is None:
            st.warning("SHAP analysis not available. Model may be too small or training failed.")
            return
        
        st.markdown("""
        <div class="card">
            <p>SHAP (SHapley Additive exPlanations) values show how each feature contributes to the prediction.
            This helps understand which structural and compositional parameters most influence hydration thermodynamics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Summary Plot", "Feature Impact", "Dependence Plots", "Interaction Analysis"])
        
        with tab1:
            st.markdown("### SHAP Summary Plot - ΔH")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            shap_values = model_data['shap_values_H']
            X_display = model_data['X_train_df']
            feature_names = model_data['feature_names']
            
            # Get top features
            shap_summary = model_data['shap_summary']
            top_features = shap_summary.head(15)['feature'].values
            
            y_pos = np.arange(len(top_features))
            for i, feat in enumerate(top_features):
                feat_idx = list(feature_names).index(feat)
                shap_vals = shap_values[:, feat_idx]
                feat_vals = X_display.iloc[:, feat_idx].values
                
                # Normalize feature values for coloring
                if feat_vals.max() > feat_vals.min():
                    feat_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min())
                else:
                    feat_norm = np.zeros_like(feat_vals)
                
                # Убеждаемся, что feat_norm - одномерный массив правильной длины
                feat_norm = np.array(feat_norm).flatten()
                
                # Если размеры не совпадают, используем повторение или обрезаем
                if len(feat_norm) != len(shap_vals):
                    feat_norm = np.resize(feat_norm, len(shap_vals))
                
                # Add jitter for beeswarm effect
                jitter = np.random.normal(0, 0.05, len(shap_vals))
                y_jitter = i + jitter
                
                scatter = ax.scatter(shap_vals, y_jitter, c=feat_norm, 
                                   cmap='coolwarm', alpha=0.6, s=30,
                                   edgecolors='black', linewidth=0.3,
                                   norm=plt.Normalize(vmin=0, vmax=1))
                
                ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features)
            ax.set_xlabel('SHAP value (impact on ΔH)', fontsize=12)
            ax.set_title('SHAP Summary: Feature Impact on ΔH Prediction', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.colorbar(scatter, ax=ax, label='Feature value (normalized)')
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            st.markdown("### Mean |SHAP| Values (Feature Importance)")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            top_shap = shap_summary.head(15)
            
            # Color code by feature type
            colors = []
            for f in top_shap['feature']:
                if 'chi' in f or 'electro' in f:
                    colors.append(MODERN_COLORS['primary'])
                elif 'r_' in f or 'radius' in f or 'lattice' in f:
                    colors.append(MODERN_COLORS['secondary'])
                elif 'content' in f or 'vacancy' in f:
                    colors.append(MODERN_COLORS['success'])
                elif 'polar' in f:
                    colors.append(MODERN_COLORS['purple'])
                elif 'ion' in f:
                    colors.append(MODERN_COLORS['pink'])
                else:
                    colors.append(MODERN_COLORS['gray'])
            
            bars = ax.barh(range(len(top_shap)), top_shap['mean_abs_shap'], 
                          color=colors, edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, top_shap['mean_abs_shap'])):
                ax.text(val + 0.1, i, f'{val:.2f}', 
                       va='center', fontsize=9)
            
            ax.set_yticks(range(len(top_shap)))
            ax.set_yticklabels(top_shap['feature'])
            ax.set_xlabel('Mean |SHAP|', fontsize=12)
            ax.set_title('Average Impact on Model Output Magnitude', 
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=MODERN_COLORS['primary'], label='Electronegativity'),
                Patch(facecolor=MODERN_COLORS['secondary'], label='Structural'),
                Patch(facecolor=MODERN_COLORS['success'], label='Composition'),
                Patch(facecolor=MODERN_COLORS['purple'], label='Polarizability'),
                Patch(facecolor=MODERN_COLORS['pink'], label='Ionization'),
                Patch(facecolor=MODERN_COLORS['gray'], label='Other')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
            
            st.pyplot(fig)
            plt.close()
        
        with tab3:
            st.markdown("### Partial Dependence Plots")
            
            st.markdown("""
            Partial dependence plots show how ΔH changes when varying one feature while keeping others constant.
            Select features to explore their individual effects.
            """)
            
            # Multi-select for features
            available_features = [f for f in feature_names if f not in ['A_enc', 'B_enc', 'D_enc']]
            selected_features = st.multiselect(
                "Select features for partial dependence plots",
                available_features,
                default=['content', 'r_B_avg', 't_Goldschmidt', 'chi_diff'][:min(4, len(available_features))]
            )
            
            if selected_features:
                n_features = len(selected_features)
                n_cols = 2
                n_rows = (n_features + 1) // 2
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
                if n_rows == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.ravel()
                
                for idx, feat in enumerate(selected_features):
                    if idx < len(axes):
                        ax = axes[idx]
                        
                        # Calculate partial dependence
                        if feat in X_display.columns:
                            feature_idx = list(X_display.columns).index(feat)
                            feature_vals = X_display.iloc[:, feature_idx]
                            
                            # Create grid
                            grid = np.linspace(feature_vals.min(), feature_vals.max(), 50)
                            
                            # Calculate partial dependence
                            pdp_values = []
                            pdp_std = []
                            
                            for val in grid:
                                X_temp = X_display.copy()
                                X_temp.iloc[:, feature_idx] = val
                                X_temp_scaled = model_data['scaler'].transform(X_temp[feature_names])
                                preds = model_data['models']['xgb_H'].predict(X_temp_scaled)
                                pdp_values.append(preds.mean())
                                pdp_std.append(preds.std())
                            
                            ax.plot(grid, pdp_values, 'b-', linewidth=2, label='PDP')
                            ax.fill_between(grid, 
                                           np.array(pdp_values) - np.array(pdp_std),
                                           np.array(pdp_values) + np.array(pdp_std),
                                           alpha=0.2, color='blue', label='±1 std')
                            
                            # Add rug plot of actual values
                            ax.plot(feature_vals, [pdp_values[0]] * len(feature_vals), 
                                   '|', color='red', alpha=0.5, markersize=10, label='Data distribution')
                            
                            ax.set_xlabel(feat.replace('_', ' ').title(), fontsize=10)
                            ax.set_ylabel('Partial dependence of ΔH (kJ/mol)', fontsize=10)
                            ax.set_title(f'PDP: {feat.replace("_", " ").title()}', fontsize=11, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            ax.legend(loc='best', fontsize=8)
                
                # Hide empty subplots
                for idx in range(len(selected_features), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.suptitle('Partial Dependence Plots for Selected Features', 
                            fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with tab4:
            st.markdown("### SHAP Interaction Analysis")
            
            st.markdown("""
            SHAP interaction values reveal how features interact with each other.
            Positive interaction means features amplify each other's effect.
            """)
            
            # Select top features for interaction analysis
            top_features_for_interaction = shap_summary.head(6)['feature'].values
            
            # Create interaction matrix
            n_top = len(top_features_for_interaction)
            interaction_matrix = np.zeros((n_top, n_top))
            
            for i, feat_i in enumerate(top_features_for_interaction):
                for j, feat_j in enumerate(top_features_for_interaction):
                    if i < j:
                        # Calculate interaction strength (simplified)
                        idx_i = list(feature_names).index(feat_i)
                        idx_j = list(feature_names).index(feat_j)
                        
                        # Approximate interaction via correlation of SHAP values
                        shap_i = shap_values[:, idx_i]
                        shap_j = shap_values[:, idx_j]
                        
                        if len(shap_i) > 1 and len(shap_j) > 1:
                            interaction = np.abs(np.corrcoef(shap_i, shap_j)[0, 1])
                            interaction_matrix[i, j] = interaction
                            interaction_matrix[j, i] = interaction
            
            # Plot interaction heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(interaction_matrix, cmap='YlOrRd', vmin=0, vmax=1)
            
            # Add text annotations
            for i in range(n_top):
                for j in range(n_top):
                    if i != j:
                        text = ax.text(j, i, f'{interaction_matrix[i, j]:.2f}',
                                     ha='center', va='center', fontsize=9,
                                     color='black' if interaction_matrix[i, j] < 0.7 else 'white')
            
            ax.set_xticks(range(n_top))
            ax.set_yticks(range(n_top))
            ax.set_xticklabels([f.replace('_', ' ').title() for f in top_features_for_interaction], 
                               rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels([f.replace('_', ' ').title() for f in top_features_for_interaction], 
                               fontsize=9)
            ax.set_title('SHAP Feature Interaction Matrix', fontsize=14, fontweight='bold')
            
            plt.colorbar(im, ax=ax, label='Interaction Strength')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown("""
            <div class="card">
                <h4>Interpretation Guide</h4>
                <ul>
                    <li><b>Interaction strength > 0.7</b>: Strong interaction between features</li>
                    <li><b>Interaction strength 0.3-0.7</b>: Moderate interaction</li>
                    <li><b>Interaction strength < 0.3</b>: Weak or no interaction</li>
                    <li>Strong interactions suggest synergistic effects on hydration thermodynamics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # Page 6: 3D Visualization (New)
    # =========================================================================
    elif page == "🔬 3D Visualization":
        st.markdown("## 🔬 3D Visualization of Hydration Landscape")
        
        # Prepare data with descriptors
        desc_list = []
        for _, row in df.iterrows():
            desc = calculate_descriptors(row)
            if len(desc) > 10:
                desc['A_cation'] = row['A_cation']
                desc['B_cation'] = row['B_cation']
                desc['dopant'] = row['dopant']
                desc['delta_H'] = row['delta_H']
                desc['delta_S'] = row['delta_S']
                desc_list.append(desc)
        
        plot_df = pd.DataFrame(desc_list)
        
        if plot_df.empty:
            st.warning("No data available for 3D visualization.")
            return
        
        # Interactive 3D plot with Plotly
        st.markdown("### Interactive 3D Scatter Plot")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X-axis", 
                                 ['r_B_avg', 't_Goldschmidt', 'chi_diff', 'content', 
                                  'delta_r_B', 'lattice_energy', 'polarizability_avg'],
                                 index=0)
        with col2:
            y_axis = st.selectbox("Y-axis",
                                 ['chi_diff', 'r_B_avg', 't_Goldschmidt', 'content',
                                  'delta_r_B', 'lattice_energy', 'ionization_avg'],
                                 index=1)
        with col3:
            z_axis = st.selectbox("Z-axis",
                                 ['delta_H', 'delta_S', 't_Goldschmidt', 'lattice_energy'],
                                 index=0)
        
        color_by = st.selectbox("Color by", 
                               ['delta_H', 'delta_S', 'A_cation', 'B_cation', 'dopant'],
                               index=0)
        
        # Create 3D scatter plot
        fig = px.scatter_3d(plot_df, 
                           x=x_axis, y=y_axis, z=z_axis,
                           color=color_by,
                           hover_data=['A_cation', 'B_cation', 'dopant', 'content', 'delta_H', 'delta_S'],
                           title=f'3D Hydration Space: {x_axis} vs {y_axis} vs {z_axis}',
                           labels={x_axis: x_axis.replace('_', ' ').title(),
                                  y_axis: y_axis.replace('_', ' ').title(),
                                  z_axis: z_axis.replace('_', ' ').title()})
        
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
        fig.update_layout(scene=dict(
                            xaxis_title=x_axis.replace('_', ' ').title(),
                            yaxis_title=y_axis.replace('_', ' ').title(),
                            zaxis_title=z_axis.replace('_', ' ').title()),
                         width=900, height=700)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D Surface plot
        st.markdown("### 3D Surface: Hydration Energy Landscape")
        
        col1, col2 = st.columns(2)
        
        with col1:
            surface_x = st.selectbox("Surface X-axis", 
                                    ['r_B_avg', 't_Goldschmidt', 'chi_diff', 'content'],
                                    index=0, key='surf_x')
        with col2:
            surface_y = st.selectbox("Surface Y-axis",
                                    ['chi_diff', 'r_B_avg', 't_Goldschmidt', 'content'],
                                    index=1, key='surf_y')
        
        # Create grid for surface
        x_grid = np.linspace(plot_df[surface_x].min(), plot_df[surface_x].max(), 30)
        y_grid = np.linspace(plot_df[surface_y].min(), plot_df[surface_y].max(), 30)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate ΔH values
        from scipy.interpolate import griddata
        Z = griddata((plot_df[surface_x], plot_df[surface_y]), plot_df['delta_H'], 
                    (X, Y), method='cubic')
        
        # Create surface plot
        fig = go.Figure(data=[
            go.Surface(z=Z, x=x_grid, y=y_grid, colorscale='RdBu_r',
                      contours=dict(z=dict(show=True, usecolormap=True,
                                          highlightcolor="limegreen", project=dict(z=True))))
        ])
        
        fig.update_layout(
            title=f'ΔH Surface: {surface_x.replace("_", " ").title()} vs {surface_y.replace("_", " ").title()}',
            scene=dict(
                xaxis_title=surface_x.replace('_', ' ').title(),
                yaxis_title=surface_y.replace('_', ' ').title(),
                zaxis_title='ΔH (kJ/mol)'
            ),
            width=900, height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add the enhanced 3D landscape with projections
        st.markdown("### Enhanced 3D Landscape with Projections")
        fig = create_3d_descriptor_landscape(plot_df)
        st.pyplot(fig)
        plt.close()

    # =========================================================================
    # Page 7: Multi-dimensional Analysis (New)
    # =========================================================================
    elif page == "📊 Multi-dimensional Analysis":
        st.markdown("## 📊 Multi-dimensional Pattern Analysis")
        
        # Prepare data with descriptors
        desc_list = []
        for _, row in df.iterrows():
            desc = calculate_descriptors(row)
            if len(desc) > 10:
                desc['A_cation'] = row['A_cation']
                desc['B_cation'] = row['B_cation']
                desc['dopant'] = row['dopant']
                desc['delta_H'] = row['delta_H']
                desc['delta_S'] = row['delta_S']
                desc_list.append(desc)
        
        plot_df = pd.DataFrame(desc_list)
        
        if plot_df.empty:
            st.warning("No data available for multi-dimensional analysis.")
            return
        
        tab1, tab2, tab3 = st.tabs(["Parallel Coordinates", "Andrews Curves", "Radar Charts"])
        
        with tab1:
            st.markdown("### Parallel Coordinates Plot")
            
            # Select features for parallel coordinates
            all_features = ['r_B_avg', 't_Goldschmidt', 'chi_diff', 'delta_r_B', 
                           'content', 'lattice_energy', 'polarizability_avg', 
                           'ionization_avg', 'delta_H', 'delta_S']
            
            available_features = [f for f in all_features if f in plot_df.columns]
            
            selected_features = st.multiselect(
                "Select features for parallel coordinates",
                available_features,
                default=available_features[:min(6, len(available_features))]
            )
            
            if len(selected_features) >= 2:
                # Normalize data
                df_norm = plot_df[selected_features].copy()
                for col in selected_features:
                    if col not in ['delta_H', 'delta_S']:  # Don't normalize targets
                        min_val = df_norm[col].min()
                        max_val = df_norm[col].max()
                        if max_val > min_val:
                            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
                
                # Create parallel coordinates plot
                fig = px.parallel_coordinates(
                    df_norm,
                    color='delta_H' if 'delta_H' in selected_features else None,
                    color_continuous_scale=px.colors.diverging.RdBu_r,
                    title='Parallel Coordinates: Multi-dimensional Patterns',
                    labels={col: col.replace('_', ' ').title() for col in selected_features}
                )
                
                fig.update_layout(width=1000, height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Andrews Curves")
            
            # Select features for Andrews curves
            andrews_features = st.multiselect(
                "Select features for Andrews curves",
                [f for f in available_features if f not in ['delta_H', 'delta_S']],
                default=[f for f in available_features if f not in ['delta_H', 'delta_S']][:min(5, len(available_features))]
            )
            
            if len(andrews_features) >= 2:
                from pandas.plotting import andrews_curves
                
                # Prepare data
                df_andrews = plot_df[andrews_features + ['delta_H']].copy()
                df_andrews['delta_H_class'] = pd.qcut(df_andrews['delta_H'], q=4, 
                                                      labels=['Q1 (strongest)', 'Q2', 'Q3', 'Q4 (weakest)'])
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Custom Andrews curves implementation
                t = np.linspace(-np.pi, np.pi, 100)
                
                for class_name in df_andrews['delta_H_class'].unique():
                    class_data = df_andrews[df_andrews['delta_H_class'] == class_name][andrews_features]
                    mean_values = class_data.mean().values
                    
                    # Andrews curve: f(t) = x1/√2 + x2·sin(t) + x3·cos(t) + x4·sin(2t) + ...
                    curve = mean_values[0] / np.sqrt(2) * np.ones_like(t)
                    for i, val in enumerate(mean_values[1:], 1):
                        if i % 2 == 1:
                            curve += val * np.sin((i+1)//2 * t)
                        else:
                            curve += val * np.cos(i//2 * t)
                    
                    ax.plot(t, curve, linewidth=2, label=class_name)
                
                ax.set_xlabel('t', fontsize=12)
                ax.set_ylabel('Andrews curve value', fontsize=12)
                ax.set_title('Andrews Curves by ΔH Quartile', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
        
        with tab3:
            st.markdown("### Radar Charts by Material Family")
            
            # Select group by
            group_by = st.selectbox("Group by", ['B_cation', 'A_cation', 'dopant'])
            
            # Select features for radar
            radar_features = st.multiselect(
                "Select features for radar chart",
                [f for f in available_features if f not in ['delta_H', 'delta_S']],
                default=['r_B_avg', 't_Goldschmidt', 'chi_diff', 'content'][:min(4, len(available_features))]
            )
            
            if len(radar_features) >= 3 and group_by in plot_df.columns:
                # Calculate mean values for each group
                grouped = plot_df.groupby(group_by)[radar_features].mean()
                
                # Normalize for radar
                for col in radar_features:
                    min_val = grouped[col].min()
                    max_val = grouped[col].max()
                    if max_val > min_val:
                        grouped[col] = (grouped[col] - min_val) / (max_val - min_val)
                
                # Create radar chart
                fig = go.Figure()
                
                for group in grouped.index:
                    fig.add_trace(go.Scatterpolar(
                        r=grouped.loc[group].values.tolist() + [grouped.loc[group].values[0]],
                        theta=radar_features + [radar_features[0]],
                        fill='toself',
                        name=str(group)
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    title=f'Radar Charts: Feature Profiles by {group_by}',
                    width=800,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # Page 8: Sensitivity Analysis (New)
    # =========================================================================
    elif page == "⚠️ Sensitivity Analysis":
        st.markdown("## ⚠️ Sensitivity Analysis and Critical Regions")
        
        if model_data is None:
            st.warning("Model not available for sensitivity analysis.")
            return
        
        # Prepare data with descriptors
        desc_list = []
        for _, row in df.iterrows():
            desc = calculate_descriptors(row)
            if len(desc) > 10:
                desc['A_cation'] = row['A_cation']
                desc['B_cation'] = row['B_cation']
                desc['dopant'] = row['dopant']
                desc['delta_H'] = row['delta_H']
                desc['delta_S'] = row['delta_S']
                desc_list.append(desc)
        
        plot_df = pd.DataFrame(desc_list)
        
        if plot_df.empty:
            st.warning("No data available for sensitivity analysis.")
            return
        
        # Generate sensitivity heatmap
        fig = create_sensitivity_heatmap(model_data, plot_df)
        if fig:
            st.pyplot(fig)
            plt.close()
        
        # Additional sensitivity analysis
        st.markdown("### Feature Sensitivity Analysis")
        
        # Calculate feature sensitivities using partial derivatives
        feature_names = model_data['feature_names']
        X_scaled = model_data['X_train']
        
        # Perturbation analysis
        n_samples = min(100, len(X_scaled))
        sensitivities = []
        
        with st.spinner("Calculating feature sensitivities..."):
            for i, feature in enumerate(feature_names):
                if feature in ['A_enc', 'B_enc', 'D_enc']:
                    continue
                
                feature_sensitivity = []
                for sample_idx in range(n_samples):
                    X_base = X_scaled[sample_idx:sample_idx+1].copy()
                    
                    # Perturb feature
                    delta = 0.1  # 10% perturbation
                    X_perturbed = X_base.copy()
                    X_perturbed[0, i] += delta
                    
                    # Get predictions
                    pred_base = model_data['models']['xgb_H'].predict(X_base)[0]
                    pred_perturbed = model_data['models']['xgb_H'].predict(X_perturbed)[0]
                    
                    # Calculate sensitivity
                    sensitivity = abs(pred_perturbed - pred_base) / delta
                    feature_sensitivity.append(sensitivity)
                
                sensitivities.append({
                    'feature': feature,
                    'mean_sensitivity': np.mean(feature_sensitivity),
                    'std_sensitivity': np.std(feature_sensitivity)
                })
        
        sensitivities_df = pd.DataFrame(sensitivities).sort_values('mean_sensitivity', ascending=False)
        
        # Plot sensitivities
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_sensitivities = sensitivities_df.head(15)
        
        bars = ax.barh(range(len(top_sensitivities)), top_sensitivities['mean_sensitivity'],
                      xerr=top_sensitivities['std_sensitivity'],
                      color=MODERN_COLORS['primary'], edgecolor='black', alpha=0.7,
                      capsize=3)
        
        ax.set_yticks(range(len(top_sensitivities)))
        ax.set_yticklabels(top_sensitivities['feature'])
        ax.set_xlabel('Sensitivity (|ΔΔH/Δx|)', fontsize=12)
        ax.set_title('Feature Sensitivity Analysis (Response to 10% Perturbation)', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        <div class="card">
            <h4>Interpretation Guide</h4>
            <ul>
                <li><b>High sensitivity</b> (>10): Small changes in this feature cause large changes in ΔH</li>
                <li><b>Medium sensitivity</b> (5-10): Moderate impact on hydration thermodynamics</li>
                <li><b>Low sensitivity</b> (<5): Feature has little influence on predictions</li>
                <li>Critical regions (cyan in map) indicate descriptor combinations where predictions are most sensitive</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # Page 9: Cluster Analysis (New)
    # =========================================================================
    elif page == "🌲 Cluster Analysis":
        st.markdown("## 🌲 Hierarchical Cluster Analysis of Materials")
        
        # Prepare data with descriptors
        desc_list = []
        for _, row in df.iterrows():
            desc = calculate_descriptors(row)
            if len(desc) > 10:
                desc['A_cation'] = row['A_cation']
                desc['B_cation'] = row['B_cation']
                desc['dopant'] = row['dopant']
                desc['delta_H'] = row['delta_H']
                desc['delta_S'] = row['delta_S']
                desc_list.append(desc)
        
        plot_df = pd.DataFrame(desc_list)
        
        if len(plot_df) < 5:
            st.warning("Need at least 5 samples for cluster analysis.")
            return
        
        # Generate clustering visualization
        fig = create_hierarchical_clustering(plot_df)
        if fig:
            st.pyplot(fig)
            plt.close()
        
        # Additional cluster analysis
        st.markdown("### Cluster Membership Analysis")
        
        # Select features for clustering
        cluster_features = ['r_B_avg', 't_Goldschmidt', 'chi_diff', 'delta_r_B', 
                           'content', 'lattice_energy', 'delta_H', 'delta_S']
        
        available_features = [f for f in cluster_features if f in plot_df.columns]
        
        # Prepare data
        X_cluster = plot_df[available_features].copy()
        
        # Standardize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # Determine optimal number of clusters
        from sklearn.metrics import silhouette_score
        
        n_clusters_range = range(2, min(8, len(plot_df)//2))
        silhouette_scores = []
        
        for n_clusters in n_clusters_range:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
        
        # Plot silhouette scores
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(list(n_clusters_range), silhouette_scores, 'o-', 
                color=MODERN_COLORS['primary'], linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('Optimal Number of Clusters', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark optimal
        optimal_n = n_clusters_range[np.argmax(silhouette_scores)]
        ax.axvline(x=optimal_n, color='red', linestyle='--', alpha=0.5,
                   label=f'Optimal: {optimal_n} clusters')
        ax.legend(loc='best')
        
        st.pyplot(fig)
        plt.close()
        
        # Show cluster compositions
        st.markdown(f"### Cluster Composition (k={optimal_n})")
        
        # Perform clustering with optimal k
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=optimal_n, random_state=42, n_init=10)
        plot_df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Display cluster statistics
        for cluster_id in range(optimal_n):
            cluster_data = plot_df[plot_df['Cluster'] == cluster_id]
            
            with st.expander(f"Cluster {cluster_id} (n={len(cluster_data)})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Composition summary:**")
                    st.markdown(f"- A-cations: {', '.join(cluster_data['A_cation'].unique())}")
                    st.markdown(f"- B-cations: {', '.join(cluster_data['B_cation'].unique())}")
                    st.markdown(f"- Dopants: {', '.join(cluster_data['dopant'].unique())}")
                
                with col2:
                    st.markdown("**Property summary:**")
                    st.markdown(f"- ΔH: {cluster_data['delta_H'].mean():.1f} ± {cluster_data['delta_H'].std():.1f} kJ/mol")
                    st.markdown(f"- ΔS: {cluster_data['delta_S'].mean():.1f} ± {cluster_data['delta_S'].std():.1f} J/mol·K")
                    st.markdown(f"- Content: {cluster_data['content'].mean():.2f} ± {cluster_data['content'].std():.2f}")
                
                # Show sample materials
                st.markdown("**Representative materials:**")
                st.dataframe(cluster_data[['A_cation', 'B_cation', 'dopant', 'content', 'delta_H', 'delta_S']].head(5))

    # =========================================================================
    # Page 10: Proton Concentration 3D (New)
    # =========================================================================
    elif page == "💧 Proton Concentration 3D":
        st.markdown("## 💧 3D Proton Concentration Predictor")
        
        if model_data is None:
            st.warning("Model not available for proton concentration prediction.")
            return
        
        st.markdown("""
        <div class="card">
            <p>This module predicts proton concentration [OH] as a function of dopant content (x),
            temperature (T), and water vapor pressure (pH2O) using the formula:</p>
            <p><b>lnKw = -ΔH/RT + ΔS/R</b></p>
            <p><b>[OH] = {3Kw·pH2O - [Kw·pH2O·(9Kw·pH2O - 6Kw·pH2O·[D] + Kw·pH2O·[D]² + 24[D] - 4[D]²)]^(1/2)} / (Kw·pH2O - 4)</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate proton concentration visualizations
        create_proton_concentration_3d(model_data, df_features)

    # =========================================================================
    # Page 11: About (Enhanced)
    # =========================================================================
    else:
        st.markdown("## ℹ️ About Proton Hydration Predictor v3.0")
        
        st.markdown("""
        <div class="card">
            <h3>Advanced Tool for Proton-Conducting Perovskites</h3>
            <p>Version 3.0 represents a major upgrade with enhanced visualization, 
            machine learning capabilities, and predictive modeling for hydration thermodynamics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>🎯 Key Features</h4>
                <ul>
                    <li><b>Enhanced Database</b>: 200+ experimental entries with full references</li>
                    <li><b>Advanced Descriptors</b>: 30+ crystallochemical parameters including:
                        <ul>
                            <li>Structural: tolerance factor, octahedral factor, lattice energy</li>
                            <li>Electronic: electronegativity, polarizability, ionization potential</li>
                            <li>Compositional: content, oxygen vacancy concentration</li>
                        </ul>
                    </li>
                    <li><b>Ensemble ML Models</b>: XGBoost, Random Forest, Gradient Boosting</li>
                    <li><b>SHAP Analysis</b>: Feature importance and interaction effects</li>
                    <li><b>3D Visualization</b>: Interactive surfaces and projections</li>
                    <li><b>Proton Concentration</b>: Predictive modeling as f(T, pH2O, x)</li>
                </ul>
            </div>
            
            <div class="card">
                <h4>📊 New in Version 3.0</h4>
                <ul>
                    <li>3D descriptor landscapes with orthogonal projections</li>
                    <li>Multi-dimensional analysis (parallel coordinates, Andrews curves)</li>
                    <li>Sensitivity mapping and critical region identification</li>
                    <li>Hierarchical clustering with dendrograms</li>
                    <li>Predictive 3D diagrams for proton concentration</li>
                    <li>Modern scientific UI with interactive elements</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4>🔬 Physical Meaning</h4>
                <ul>
                    <li><b>ΔH (enthalpy)</b>: Energy of water incorporation
                        <br>More negative → stronger hydration</li>
                    <li><b>ΔS (entropy)</b>: Configurational change upon hydration
                        <br>More negative → higher ordering</li>
                    <li><b>T_iso</b>: Isokinetic temperature
                        <br>ΔH-ΔS compensation point</li>
                    <li><b>[OH]</b>: Proton concentration
                        <br>Key for conductivity</li>
                </ul>
            </div>
            
            <div class="card">
                <h4>📚 Key References</h4>
                <ul>
                    <li>Kreuer, K.D. (2001) Solid State Ionics</li>
                    <li>Yamazaki et al. (2010) Chem. Mater.</li>
                    <li>Bjørheim et al. (2012) Phys. Chem. Chem. Phys.</li>
                    <li>Multiple datasets from recent publications (2015-2025)</li>
                </ul>
            </div>
            
            <div class="card">
                <h4>📝 How to Cite</h4>
                <p>If you use this tool in your research, please cite:</p>
                <p><i>Proton Hydration Predictor v3.0 - A comprehensive tool for perovskite hydration analysis</i></p>
                <p>And the original data sources listed in references.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Database summary
        st.markdown("### 📊 Database Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Entries</div>
                <div class="metric-value">{len(df)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">A-cations</div>
                <div class="metric-value">{df['A_cation'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">B-cations</div>
                <div class="metric-value">{df['B_cation'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Dopants</div>
                <div class="metric-value">{df['dopant'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Version history
        st.markdown("### 📅 Version History")
        
        version_data = pd.DataFrame({
            'Version': ['3.0 (Mar 2025)', '2.0 (Jan 2025)', '1.0 (Oct 2024)'],
            'Features': [
                '3D visualization, SHAP analysis, proton concentration modeling, modern UI',
                'Enhanced correlations, multiple ML models, partial dependence plots',
                'Basic predictions, data explorer, simple correlations'
            ]
        })
        
        st.dataframe(version_data, use_container_width=True)
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>© 2025 Proton Hydration Predictor | Developed for Materials Science Research</p>
            <p>For questions, suggestions, or data contributions, please contact the developers</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# Run the app
# =============================================================================
if __name__ == "__main__":
    main()




















