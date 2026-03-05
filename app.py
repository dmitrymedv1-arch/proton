"""
Proton Hydration Predictor for Perovskite Materials
Scientific visualization and prediction of hydration thermodynamics
"""

# ============================================================================
# FIX 1: Set matplotlib backend BEFORE any other imports
# ============================================================================
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Perovskite Hydration Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Scientific plotting style
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
})

# ============================================================================
# FIX 2: Cache resource initialization to prevent reloading
# ============================================================================
@st.cache_resource
def get_ionic_radii_db():
    return IonicRadiiDatabase()

@st.cache_resource
def get_electroneg_db():
    return ElectronegativityDatabase()

@st.cache_resource
def get_dataset():
    return HydrationDataset()

@st.cache_resource
def get_predictor(dataset):
    pred = HydrationPredictor(dataset)
    pred.train_models()
    return pred

# ============================================================================
# DATA MODELS
# ============================================================================

class IonicRadiiDatabase:
    """Ionic radii database for perovskite materials"""
    
    def __init__(self):
        self.radii = {
            # A-site cations (XII coordination for 2+, IX/XII for 3+)
            'Ba': {'charge': 2, 'XII': 1.61},
            'Sr': {'charge': 2, 'XII': 1.44},
            'Ca': {'charge': 2, 'XII': 1.34},
            'La': {'charge': 3, 'IX': 1.216, 'XII': 1.36},
            'Nd': {'charge': 3, 'IX': 1.163, 'XII': 1.27},
            'Gd': {'charge': 3, 'IX': 1.107},
            'Sm': {'charge': 3, 'IX': 1.132},
            'Y': {'charge': 3, 'IX': 1.075},
            
            # B-site cations (VI coordination)
            'Ti': {'charge': 4, 'VI': 0.605},
            'Zr': {'charge': 4, 'VI': 0.72},
            'Sn': {'charge': 4, 'VI': 0.69},
            'Ce': {'charge': 4, 'VI': 0.87},
            'Hf': {'charge': 4, 'VI': 0.71},
            'Sc': {'charge': 3, 'VI': 0.745},
            'In': {'charge': 3, 'VI': 0.80},
            'Y': {'charge': 3, 'VI': 0.90},
            'Yb': {'charge': 3, 'VI': 0.868},
            'Gd': {'charge': 3, 'VI': 0.938},
            'Er': {'charge': 3, 'VI': 0.89},
            'Lu': {'charge': 3, 'VI': 0.861},
            'Fe': {'charge': 3, 'VI': 0.645},  # High spin
            'Co': {'charge': 3, 'VI': 0.61},   # High spin
            'Ni': {'charge': 3, 'VI': 0.56},   # Low spin
            'Ga': {'charge': 3, 'VI': 0.62},
            'Al': {'charge': 3, 'VI': 0.535},
            'Mg': {'charge': 2, 'VI': 0.72},
            'Zn': {'charge': 2, 'VI': 0.74},
            'Cu': {'charge': 2, 'VI': 0.73},
        }
        
        # FIX 3: Store oxygen radius as simple float
        self.r_O = 1.40  # O2- in VI coordination
    
    def get_r_A(self, element, coord='XII'):
        """Get A-site ionic radius"""
        if element not in self.radii:
            return None
        data = self.radii[element]
        if coord in data:
            return data[coord]
        # Fallback to available coordination
        if 'XII' in data:
            return data['XII']
        elif 'IX' in data:
            return data['IX']
        return None
    
    def get_r_B(self, element):
        """Get B-site ionic radius (VI coordination)"""
        if element not in self.radii:
            return None
        data = self.radii[element]
        return data.get('VI', None)
    
    def get_r_O(self):
        """Get oxygen ionic radius"""
        return self.r_O


class ElectronegativityDatabase:
    """Electronegativity database (Pauling scale)"""
    
    def __init__(self):
        self.electroneg = {
            'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
            'N': 3.04, 'O': 3.44, 'F': 3.98, 'Na': 0.93, 'Mg': 1.31,
            'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
            'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63,
            'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91,
            'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18,
            'Se': 2.55, 'Br': 2.96, 'Kr': 3.00, 'Rb': 0.82, 'Sr': 0.95,
            'Y': 1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 2.16, 'Tc': 1.90,
            'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
            'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.10, 'I': 2.66,
            'Xe': 2.60, 'Cs': 0.79, 'Ba': 0.89, 'La': 1.10, 'Ce': 1.12,
            'Pr': 1.13, 'Nd': 1.14, 'Pm': None, 'Sm': 1.17, 'Eu': None,
            'Gd': 1.20, 'Tb': None, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24,
            'Tm': 1.25, 'Yb': None, 'Lu': 1.27, 'Hf': 1.30, 'Ta': 1.50,
            'W': 2.36, 'Re': 1.90, 'Os': 2.20, 'Ir': 2.20, 'Pt': 2.28,
            'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02,
        }
    
    def get(self, element):
        """Get electronegativity for element"""
        if element is None:
            return None
        return self.electroneg.get(element, None)


class HydrationDataset:
    """Combined dataset from all sources"""
    
    def __init__(self):
        self.radii_db = IonicRadiiDatabase()
        self.en_db = ElectronegativityDatabase()
        self.data = self._create_dataset()
    
    def _create_dataset(self):
        """Create the combined dataset from all sources"""
        
        # ====================================================================
        # DATA FROM EXCEL FILE (SHEET 1 - для ИИ)
        # ====================================================================
        excel_data = [
            # BaTiO3-based
            {'a': 'Ba', 'b': 'Ti', 'dopant': 'Sc', 'x': 0.5, 'dh': -55, 'ds': -120, 
             'structure': 'cubic', 'ref': '10.1021/ic503006u'},
            {'a': 'Ba', 'b': 'Ti', 'dopant': 'Sc', 'x': 0.6, 'dh': -53, 'ds': -102, 
             'structure': 'cubic', 'ref': '10.1021/ic503006u'},
            {'a': 'Ba', 'b': 'Ti', 'dopant': 'Sc', 'x': 0.7, 'dh': -56, 'ds': -93, 
             'structure': 'cubic', 'ref': '10.1021/ic503006u'},
            {'a': 'Ba', 'b': 'Ti', 'dopant': 'Sc', 'x': 0.2, 'dh': -86, 'ds': -111, 
             'structure': 'cubic', 'ref': '10.1016/j.actamat.2020.03.010'},
            {'a': 'Ba', 'b': 'Ti', 'dopant': 'In', 'x': 0.5, 'dh': -57, 'ds': -132, 
             'structure': 'cubic', 'ref': '10.1021/ic503006u'},
            {'a': 'Ba', 'b': 'Ti', 'dopant': 'In', 'x': 0.7, 'dh': -68, 'ds': -125, 
             'structure': 'cubic', 'ref': '10.1021/ic503006u'},
            {'a': 'Ba', 'b': 'Ti', 'dopant': 'In', 'x': 0.5, 'dh': -129, 'ds': -129, 
             'structure': 'cubic', 'ref': '10.1016/j.ceramint.2025.10.295'},
            {'a': 'Ba', 'b': 'Ti', 'dopant': 'In', 'x': 0.6, 'dh': -149, 'ds': -149, 
             'structure': 'cubic', 'ref': '10.1016/j.ceramint.2025.10.295'},
            
            # BaZrO3-based
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Y', 'x': 0.02, 'dh': -80.9, 'ds': -94.4,
             'structure': 'cubic', 'ref': '10.1016/S0167-2738(01)00953-5'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Y', 'x': 0.05, 'dh': -79.5, 'ds': -93.5,
             'structure': 'cubic', 'ref': '10.1016/S0167-2738(01)00953-5'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Y', 'x': 0.1, 'dh': -79.5, 'ds': -88.9,
             'structure': 'tetragonal', 'ref': '10.1016/S0167-2738(01)00953-5'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Y', 'x': 0.15, 'dh': -83.4, 'ds': -92.1,
             'structure': 'tetragonal', 'ref': '10.1016/S0167-2738(01)00953-5'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Y', 'x': 0.2, 'dh': -93.3, 'ds': -103.2,
             'structure': 'tetragonal', 'ref': '10.1016/S0167-2738(01)00953-5'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Y', 'x': 0.25, 'dh': -83.4, 'ds': -92.1,
             'structure': 'cubic', 'ref': '10.1016/S0167-2738(01)00953-5'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Sc', 'x': 0.1, 'dh': -119.4, 'ds': -124.9,
             'structure': 'cubic', 'ref': '10.1016/S0167-2738(01)00953-5'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'In', 'x': 0.1, 'dh': -66.6, 'ds': -90.2,
             'structure': 'tetragonal', 'ref': '10.1016/S0167-2738(01)00953-5'},
            
            # BaSnO3-based
            {'a': 'Ba', 'b': 'Sn', 'dopant': 'Y', 'x': 0.05, 'dh': -50, 'ds': -102,
             'structure': 'cubic', 'ref': '10.1016/j.ssi.2012.02.045'},
            {'a': 'Ba', 'b': 'Sn', 'dopant': 'Y', 'x': 0.125, 'dh': -49.01, 'ds': -86.81,
             'structure': 'cubic', 'ref': '10.1016/j.ijhydene.2011.03.105'},
            {'a': 'Ba', 'b': 'Sn', 'dopant': 'In', 'x': 0.125, 'dh': -58.9, 'ds': -109.8,
             'structure': 'cubic', 'ref': '10.1016/j.ijhydene.2011.03.105'},
            {'a': 'Ba', 'b': 'Sn', 'dopant': 'Sc', 'x': 0.125, 'dh': -73.4, 'ds': -106.4,
             'structure': 'cubic', 'ref': '10.1016/j.ijhydene.2011.03.105'},
            {'a': 'Ba', 'b': 'Sn', 'dopant': 'Y', 'x': 0.2, 'dh': -73, 'ds': -115,
             'structure': 'cubic', 'ref': '10.1016/j.jpowsour.2023.232883'},
            {'a': 'Ba', 'b': 'Sn', 'dopant': 'In', 'x': 0.05, 'dh': -57.379, 'ds': -57.379,
             'structure': 'cubic', 'ref': '10.1016/j.matre.2025.100382'},
            {'a': 'Ba', 'b': 'Sn', 'dopant': 'In', 'x': 0.1, 'dh': -58.885, 'ds': -58.885,
             'structure': 'cubic', 'ref': '10.1016/j.matre.2025.100382'},
            {'a': 'Ba', 'b': 'Sn', 'dopant': 'In', 'x': 0.2, 'dh': -65.562, 'ds': -65.562,
             'structure': 'cubic', 'ref': '10.1016/j.matre.2025.100382'},
            {'a': 'Ba', 'b': 'Sn', 'dopant': 'In', 'x': 0.3, 'dh': -78.922, 'ds': -78.922,
             'structure': 'cubic', 'ref': '10.1016/j.matre.2025.100382'},
            {'a': 'Ba', 'b': 'Sn', 'dopant': 'In', 'x': 0.4, 'dh': -84.221, 'ds': -84.221,
             'structure': 'cubic', 'ref': '10.1016/j.matre.2025.100382'},
            {'a': 'Ba', 'b': 'Sn', 'dopant': 'In', 'x': 0.5, 'dh': -75.784, 'ds': -75.784,
             'structure': 'cubic', 'ref': '10.1016/j.matre.2025.100382'},
            
            # BaCeO3-based
            {'a': 'Ba', 'b': 'Ce', 'dopant': 'Gd', 'x': 0.1, 'dh': -133, 'ds': -141,
             'structure': 'orthorhombic', 'ref': '10.1039/c5ta04932f'},
            {'a': 'Ba', 'b': 'Ce', 'dopant': 'Y', 'x': 0.1, 'dh': -135, 'ds': -141,
             'structure': 'orthorhombic', 'ref': '10.1039/c5ta04932f'},
            
            # Sr-based
            {'a': 'Sr', 'b': 'Sn', 'dopant': 'Sc', 'x': 0.2, 'dh': -76, 'ds': -116,
             'structure': 'orthorhombic', 'ref': '10.1021/acsenergylett.1c01239'},
        ]
        
        # ====================================================================
        # DATA FROM TABLE 1 (Classical proton conductors)
        # ====================================================================
        table1_data = [
            # BaZrO3-based
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Sc', 'x': 0.59, 'dh': -121, 'ds': -117,
             'structure': 'cubic', 'ref': '[109]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Sc', 'x': 0.6, 'dh': -121, 'ds': -117,
             'structure': 'cubic', 'ref': '[110]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Sc', 'x': 0.4, 'dh': -108, 'ds': -115,
             'structure': 'cubic', 'ref': '[111]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Y', 'x': 0.4, 'dh': -26, 'ds': -41,
             'structure': 'cubic', 'ref': '[51]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Sc', 'x': 0.3, 'dh': -111, 'ds': -118,
             'structure': 'cubic', 'ref': '[112]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Y', 'x': 0.3, 'dh': -26, 'ds': -44,
             'structure': 'cubic', 'ref': '[51]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Y', 'x': 0.15, 'dh': -83.4, 'ds': -92.1,
             'structure': 'cubic', 'ref': '[21]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Er', 'x': 0.2, 'dh': -82, 'ds': -106,
             'structure': 'cubic', 'ref': '[111]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'In', 'x': 0.2, 'dh': -71, 'ds': -101,
             'structure': 'cubic', 'ref': '[111]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Lu', 'x': 0.2, 'dh': -99, 'ds': -112,
             'structure': 'cubic', 'ref': '[111]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Sc', 'x': 0.2, 'dh': -104, 'ds': -96,
             'structure': 'cubic', 'ref': '[110]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Sc', 'x': 0.2, 'dh': -103.6, 'ds': -95.6,
             'structure': 'cubic', 'ref': '[113]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Y', 'x': 0.12, 'dh': -78, 'ds': -94,
             'structure': 'cubic', 'ref': '[114]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Y', 'x': 0.2, 'dh': -89, 'ds': -124,
             'structure': 'cubic', 'ref': '[51]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Y', 'x': 0.2, 'dh': -91.1, 'ds': -104.1,
             'structure': 'cubic', 'ref': '[113]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Gd', 'x': 0.1, 'dh': -66.1, 'ds': -85.9,
             'structure': 'cubic', 'ref': '[21]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'In', 'x': 0.1, 'dh': -66.6, 'ds': -90.2,
             'structure': 'cubic', 'ref': '[21]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Sc', 'x': 0.1, 'dh': -119.4, 'ds': -124.9,
             'structure': 'cubic', 'ref': '[21]'},
            {'a': 'Ba', 'b': 'Zr', 'dopant': 'Y', 'x': 0.1, 'dh': -76, 'ds': -98,
             'structure': 'cubic', 'ref': '[115]'},
            
            # BaCeO3-based
            {'a': 'Ba', 'b': 'Ce', 'dopant': 'Y', 'x': 0.2, 'dh': -136.9, 'ds': -129.9,
             'structure': 'orthorhombic', 'ref': '[113]'},
            {'a': 'Ba', 'b': 'Ce', 'dopant': 'Er', 'x': 0.1, 'dh': -124, 'ds': -129,
             'structure': 'orthorhombic', 'ref': '[53]'},
            {'a': 'Ba', 'b': 'Ce', 'dopant': 'Gd', 'x': 0.1, 'dh': -133, 'ds': -141,
             'structure': 'orthorhombic', 'ref': '[53]'},
            {'a': 'Ba', 'b': 'Ce', 'dopant': 'In', 'x': 0.1, 'dh': -96, 'ds': -129,
             'structure': 'orthorhombic', 'ref': '[53]'},
            {'a': 'Ba', 'b': 'Ce', 'dopant': 'Sc', 'x': 0.1, 'dh': -139, 'ds': -141,
             'structure': 'orthorhombic', 'ref': '[53]'},
            {'a': 'Ba', 'b': 'Ce', 'dopant': 'Yb', 'x': 0.1, 'dh': -127, 'ds': -126,
             'structure': 'orthorhombic', 'ref': '[113]'},
            
            # Sr-based
            {'a': 'Sr', 'b': 'Ce', 'dopant': 'Yb', 'x': 0.05, 'dh': -157, 'ds': -128,
             'structure': 'orthorhombic', 'ref': '[39]'},
            {'a': 'Sr', 'b': 'Ce', 'dopant': 'Yb', 'x': 0.023, 'dh': -157, 'ds': -128,
             'structure': 'orthorhombic', 'ref': '[39]'},
            
            # La-based
            {'a': 'La', 'b': 'Sc', 'dopant': 'Sr', 'x': 0.09, 'dh': -97, 'ds': -112,
             'structure': 'cubic', 'ref': '[122]'},
            {'a': 'La', 'b': 'Sc', 'dopant': 'Ba', 'x': 0.05, 'dh': -85, 'ds': -106,
             'structure': 'cubic', 'ref': '[58]'},
            {'a': 'La', 'b': 'Sc', 'dopant': 'Ca', 'x': 0.4, 'dh': -132, 'ds': -126,
             'structure': 'cubic', 'ref': '[123]'},
        ]
        
        # ====================================================================
        # DATA FROM TABLE 2 (Layered and related structures)
        # ====================================================================
        table2_data = [
            {'a': 'Ba', 'b': 'Y,Sn', 'dopant': None, 'x': 0, 'dh': -80, 'ds': -109,
             'structure': 'layered', 'composition': 'Ba2YSnO5.5', 'ref': '[4]'},
            {'a': 'Ba', 'b': 'Ca,Nd', 'dopant': None, 'x': 0, 'dh': -65, 'ds': -104,
             'structure': 'layered', 'composition': 'Ba3Ca1.17Nd1.83O9', 'ref': '[4]'},
            {'a': 'Ba', 'b': 'Ca,Nb', 'dopant': None, 'x': 0, 'dh': -94, 'ds': -154,
             'structure': 'layered', 'composition': 'Ba4Ca2Nb2O11', 'ref': '[127,128]'},
            {'a': 'La', 'b': 'Ca,Mg,Ti', 'dopant': None, 'x': 0, 'dh': -125, 'ds': -160,
             'structure': 'double perovskite', 'composition': 'La1.8Ca0.2MgTiO6', 'ref': '[134]'},
            {'a': 'La', 'b': 'Mg,Zr', 'dopant': None, 'x': 0, 'dh': -110, 'ds': -155,
             'structure': 'double perovskite', 'composition': 'La2Mg1.14Zr0.86O6', 'ref': '[134]'},
            {'a': 'La', 'b': 'Ce', 'dopant': None, 'x': 0, 'dh': -77, 'ds': -128,
             'structure': 'fluorite', 'composition': 'La2Ce2O7', 'ref': '[135]'},
        ]
        
        # Combine all data
        all_data = excel_data + table1_data + table2_data
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # FIX 4: Handle None values in dopant column for composition
        df['dopant'] = df['dopant'].fillna('')
        df['composition'] = df.apply(
            lambda row: f"{row['a']}{row['b']}_{row['dopant'] if row['dopant'] else 'undoped'}{row['x']}O3", 
            axis=1
        )
        
        # Calculate descriptors
        df = self._calculate_descriptors(df)
        
        return df
    
    def _calculate_descriptors(self, df):
        """Calculate all descriptors for the dataset"""
        
        # Initialize descriptor columns
        df['r_A'] = None
        df['r_B'] = None
        df['r_dopant'] = None
        df['r_B_avg'] = None
        df['dr_B'] = None
        df['t_Goldschmidt'] = None
        df['chi_A'] = None
        df['chi_B'] = None
        df['chi_dopant'] = None
        df['chi_B_avg'] = None
        df['chi_diff'] = None
        
        r_O = self.radii_db.get_r_O()
        
        for idx, row in df.iterrows():
            # Get A-site radius
            if row['a'] in ['Ba', 'Sr', 'Ca']:
                r_A = self.radii_db.get_r_A(row['a'], 'XII')
            else:
                r_A = self.radii_db.get_r_A(row['a'], 'IX')
            df.at[idx, 'r_A'] = r_A
            
            # Get B-site radius (handle mixed B-sites)
            b_element = row['b']
            if ',' in str(b_element):
                b_element = b_element.split(',')[0]  # Use first element for simplicity
            r_B = self.radii_db.get_r_B(b_element)
            df.at[idx, 'r_B'] = r_B
            
            # Get dopant radius
            if row['dopant'] and row['dopant'] not in ['', 'None', 'none']:
                r_dop = self.radii_db.get_r_B(row['dopant'])
                chi_dop = self.en_db.get(row['dopant'])
            else:
                r_dop = None
                chi_dop = None
            df.at[idx, 'r_dopant'] = r_dop
            df.at[idx, 'chi_dopant'] = chi_dop
            
            # Calculate average B-site radius
            if r_B is not None and r_dop is not None and pd.notna(row['x']):
                r_B_avg = (1 - row['x']) * r_B + row['x'] * r_dop
                dr_B = abs(r_dop - r_B)
            else:
                r_B_avg = r_B
                dr_B = 0
            df.at[idx, 'r_B_avg'] = r_B_avg
            df.at[idx, 'dr_B'] = dr_B
            
            # Calculate Goldschmidt tolerance factor
            if r_A is not None and r_B_avg is not None and r_O is not None:
                # FIX 5: Avoid division by zero
                denominator = np.sqrt(2) * (r_B_avg + r_O)
                if denominator > 0:
                    t = (r_A + r_O) / denominator
                else:
                    t = None
                df.at[idx, 't_Goldschmidt'] = t
            
            # Get electronegativities
            df.at[idx, 'chi_A'] = self.en_db.get(row['a'])
            df.at[idx, 'chi_B'] = self.en_db.get(b_element)
            
            # Calculate average B electronegativity
            chi_B = df.at[idx, 'chi_B']
            chi_A = df.at[idx, 'chi_A']
            
            if chi_B is not None and chi_dop is not None and pd.notna(row['x']):
                chi_B_avg = (1 - row['x']) * chi_B + row['x'] * chi_dop
                # FIX 6: Handle None chi_A properly
                if chi_A is not None:
                    chi_diff = chi_B_avg - chi_A
                else:
                    chi_diff = None
            else:
                chi_B_avg = chi_B
                if chi_B is not None and chi_A is not None:
                    chi_diff = chi_B - chi_A
                else:
                    chi_diff = None
            df.at[idx, 'chi_B_avg'] = chi_B_avg
            df.at[idx, 'chi_diff'] = chi_diff
        
        return df


# ============================================================================
# PREDICTION MODELS
# ============================================================================

class HydrationPredictor:
    """Machine learning models for predicting hydration parameters"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.models_dh = {}
        self.models_ds = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.is_trained = False
    
    def prepare_features(self, df):
        """Prepare feature matrix for ML models"""
        
        # Select numerical features
        numerical_features = [
            'x', 'r_A', 'r_B', 'r_dopant', 'r_B_avg', 'dr_B',
            't_Goldschmidt', 'chi_A', 'chi_B', 'chi_dopant',
            'chi_B_avg', 'chi_diff'
        ]
        
        # Filter rows with valid numerical features
        mask = df[numerical_features].notna().all(axis=1)
        df_clean = df[mask].copy()
        
        if len(df_clean) == 0:
            return None, None, None
        
        # Prepare feature matrix
        X_num = df_clean[numerical_features].values
        
        # Encode categorical features
        categorical_features = ['a', 'b', 'dopant', 'structure']
        X_cat_list = []
        
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                # Fit on all available data
                all_values = df[col].fillna('unknown').unique()
                self.label_encoders[col].fit(all_values)
            
            # Transform current values
            values = df_clean[col].fillna('unknown').astype(str)
            encoded = self.label_encoders[col].transform(values)
            X_cat_list.append(encoded.reshape(-1, 1))
        
        if X_cat_list:
            # FIX 7: Use column_stack instead of hstack for different shapes
            X_cat = np.column_stack([x.ravel() for x in X_cat_list])
            X = np.column_stack([X_num, X_cat])
        else:
            X = X_num
        
        # Target variables
        y_dh = df_clean['dh'].values
        y_ds = df_clean['ds'].values
        
        self.feature_names = numerical_features + categorical_features
        
        return X, y_dh, y_ds
    
    def train_models(self):
        """Train multiple ML models for prediction"""
        
        X, y_dh, y_ds = self.prepare_features(self.dataset.data)
        
        if X is None:
            st.warning("Not enough data to train models")
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_dh_train, y_dh_test, y_ds_train, y_ds_test = train_test_split(
            X_scaled, y_dh, y_ds, test_size=0.2, random_state=42
        )
        
        # Train Random Forest for ∆H
        rf_dh = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_dh.fit(X_train, y_dh_train)
        self.models_dh['Random Forest'] = rf_dh
        
        # Train Gradient Boosting for ∆H
        gb_dh = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb_dh.fit(X_train, y_dh_train)
        self.models_dh['Gradient Boosting'] = gb_dh
        
        # Train Random Forest for ∆S
        rf_ds = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_ds.fit(X_train, y_ds_train)
        self.models_ds['Random Forest'] = rf_ds
        
        # Train Gradient Boosting for ∆S
        gb_ds = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb_ds.fit(X_train, y_ds_train)
        self.models_ds['Gradient Boosting'] = gb_ds
        
        # Evaluate models
        self.evaluation = {}
        for name, model in self.models_dh.items():
            y_pred = model.predict(X_test)
            self.evaluation[f'{name}_dh'] = {
                'R2': r2_score(y_dh_test, y_pred),
                'MAE': mean_absolute_error(y_dh_test, y_pred)
            }
        
        for name, model in self.models_ds.items():
            y_pred = model.predict(X_test)
            self.evaluation[f'{name}_ds'] = {
                'R2': r2_score(y_ds_test, y_pred),
                'MAE': mean_absolute_error(y_ds_test, y_pred)
            }
        
        self.is_trained = True
    
    def predict(self, a, b, dopant, x, structure='cubic'):
        """Predict hydration parameters for a new composition"""
        
        if not self.is_trained:
            return None, None, None, None
        
        # Create feature vector for new composition
        radii_db = IonicRadiiDatabase()
        en_db = ElectronegativityDatabase()
        
        # Get A-site radius
        if a in ['Ba', 'Sr', 'Ca']:
            r_A = radii_db.get_r_A(a, 'XII')
        else:
            r_A = radii_db.get_r_A(a, 'IX')
        
        # Get B-site radius
        r_B = radii_db.get_r_B(b)
        
        # Get dopant radius and electronegativity
        r_dop = radii_db.get_r_B(dopant) if dopant else None
        chi_dop = en_db.get(dopant) if dopant else None
        
        # Calculate averages
        if r_B is not None and r_dop is not None:
            r_B_avg = (1 - x) * r_B + x * r_dop
            dr_B = abs(r_dop - r_B)
        else:
            r_B_avg = r_B
            dr_B = 0
        
        # Tolerance factor
        r_O = radii_db.get_r_O()
        denominator = np.sqrt(2) * (r_B_avg + r_O) if r_B_avg is not None else 0
        if r_A is not None and r_B_avg is not None and denominator > 0:
            t = (r_A + r_O) / denominator
        else:
            t = None
        
        # Electronegativities
        chi_A = en_db.get(a)
        chi_B = en_db.get(b)
        
        if chi_B is not None and chi_dop is not None:
            chi_B_avg = (1 - x) * chi_B + x * chi_dop
            if chi_A is not None:
                chi_diff = chi_B_avg - chi_A
            else:
                chi_diff = None
        else:
            chi_B_avg = chi_B
            if chi_B is not None and chi_A is not None:
                chi_diff = chi_B - chi_A
            else:
                chi_diff = None
        
        # Create numerical feature vector
        numerical_features = [
            x, r_A, r_B, r_dop, r_B_avg, dr_B,
            t, chi_A, chi_B, chi_dop, chi_B_avg, chi_diff
        ]
        
        # FIX 8: Check for None values - if critical features are None, return None
        if any(v is None for v in [r_A, r_B, r_B_avg, t, chi_A, chi_B]):
            return None, None, None, None
        
        # Replace remaining None with 0 (less critical features)
        numerical_features = [0 if v is None else v for v in numerical_features]
        
        # Encode categorical features
        categorical_features = []
        for col, val in zip(['a', 'b', 'dopant', 'structure'], [a, b, dopant, structure]):
            if col in self.label_encoders:
                try:
                    encoded = self.label_encoders[col].transform([str(val)])[0]
                except:
                    encoded = -1  # Unknown category
            else:
                encoded = -1
            categorical_features.append(encoded)
        
        # Combine features
        X = np.array(numerical_features + categorical_features).reshape(1, -1)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict with ensemble
        dh_preds = []
        ds_preds = []
        
        for name, model in self.models_dh.items():
            dh_preds.append(model.predict(X_scaled)[0])
        
        for name, model in self.models_ds.items():
            ds_preds.append(model.predict(X_scaled)[0])
        
        if dh_preds and ds_preds:
            dh_mean = np.mean(dh_preds)
            dh_std = np.std(dh_preds)
            ds_mean = np.mean(ds_preds)
            ds_std = np.std(ds_preds)
            return dh_mean, dh_std, ds_mean, ds_std
        else:
            return None, None, None, None
    
    def find_similar_materials(self, a, b, dopant, x, n_neighbors=5):
        """Find most similar materials in the dataset"""
        
        # Prepare feature matrix for all materials
        X_all, _, _ = self.prepare_features(self.dataset.data)
        if X_all is None:
            return pd.DataFrame()
        
        # Create feature vector for query
        temp_df = pd.DataFrame([{
            'a': a, 'b': b, 'dopant': dopant, 'x': x,
            'structure': 'cubic', 'dh': 0, 'ds': 0
        }])
        # Calculate descriptors for temp_df
        temp_df = self.dataset._calculate_descriptors(temp_df)
        X_query, _, _ = self.prepare_features(temp_df)
        
        if X_query is None:
            return pd.DataFrame()
        
        # FIX 9: Use already fitted scaler, don't refit
        X_all_scaled = self.scaler.transform(X_all)
        X_query_scaled = self.scaler.transform(X_query)
        
        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(X_all)), metric='euclidean')
        nn.fit(X_all_scaled)
        distances, indices = nn.kneighbors(X_query_scaled)
        
        # Get similar materials
        similar = self.dataset.data.iloc[indices[0]].copy()
        similar['similarity_score'] = 1 / (1 + distances[0])
        
        return similar[['a', 'b', 'dopant', 'x', 'dh', 'ds', 'structure', 'ref', 'similarity_score']]


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

class ScientificVisualizer:
    """Create scientific-quality visualizations"""
    
    @staticmethod
    def plot_dh_vs_dopant_content(df, b_matrix='Zr', dopant_type='Y'):
        """Plot ∆H vs dopant content for a specific system"""
        
        # Filter data
        mask = (df['b'] == b_matrix) & (df['dopant'] == dopant_type)
        filtered = df[mask].copy()
        
        if len(filtered) < 2:
            return None
        
        # Sort by content
        filtered = filtered.sort_values('x')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot data points
        ax.errorbar(
            filtered['x'], filtered['dh'],
            yerr=5,  # Estimated error
            fmt='o', markersize=8, capsize=4,
            color='red', ecolor='black', markeredgecolor='black',
            label=f'{b_matrix}-based, {dopant_type}-doped'
        )
        
        # Add trend line
        z = np.polyfit(filtered['x'], filtered['dh'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(filtered['x'].min(), filtered['x'].max(), 50)
        ax.plot(x_trend, p(x_trend), '--', color='blue', alpha=0.7,
                label=f'Trend: ∆H = {z[0]:.1f}·x + {z[1]:.1f}')
        
        # Labels and formatting
        ax.set_xlabel('Dopant content, x', fontsize=12, fontweight='bold')
        ax.set_ylabel('∆H$_{hyd}$ (kJ mol$^{-1}$)', fontsize=12, fontweight='bold')
        ax.set_title(f'{b_matrix}{"1-x"}{dopant_type}{"x"}O₃−δ', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_dh_vs_ionic_radius(df, b_matrix='Zr'):
        """Plot ∆H vs dopant ionic radius for fixed B-site"""
        
        # Filter data
        mask = (df['b'] == b_matrix) & (df['r_dopant'].notna())
        filtered = df[mask].copy()
        
        if len(filtered) < 3:
            return None
        
        # Group by dopant type (take median for multiple x values)
        grouped = filtered.groupby('dopant').agg({
            'r_dopant': 'first',
            'dh': 'median',
            'ds': 'median',
            'x': 'mean'
        }).reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Color by dopant content
        scatter = ax.scatter(
            grouped['r_dopant'], grouped['dh'],
            c=grouped['x'], cmap='viridis', s=100,
            edgecolor='black', linewidth=1, alpha=0.8,
            vmin=0, vmax=0.5
        )
        
        # Add labels for each point
        for idx, row in grouped.iterrows():
            ax.annotate(
                row['dopant'],
                (row['r_dopant'], row['dh']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, fontweight='bold'
            )
        
        # Colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Dopant content, x', fontsize=11, fontweight='bold')
        
        # Labels
        ax.set_xlabel('Ionic radius of dopant (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel('∆H$_{hyd}$ (kJ mol$^{-1}$)', fontsize=12, fontweight='bold')
        ax.set_title(f'{b_matrix}O₃-based perovskites', fontsize=14, fontweight='bold')
        
        # Add vertical line for B-site radius
        b_radius = filtered['r_B'].iloc[0] if not filtered.empty else None
        if b_radius:
            ax.axvline(x=b_radius, color='red', linestyle='--', alpha=0.5,
                       label=f'{b_matrix} radius = {b_radius:.3f} Å')
        
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.invert_yaxis()  # More negative ∆H at top
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_dh_vs_electronegativity(df, b_matrix='Zr'):
        """Plot ∆H vs dopant electronegativity"""
        
        # Filter data
        mask = (df['b'] == b_matrix) & (df['chi_dopant'].notna())
        filtered = df[mask].copy()
        
        if len(filtered) < 3:
            return None
        
        # Group by dopant type
        grouped = filtered.groupby('dopant').agg({
            'chi_dopant': 'first',
            'dh': 'median',
            'x': 'mean'
        }).reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Plot
        scatter = ax.scatter(
            grouped['chi_dopant'], grouped['dh'],
            c=grouped['x'], cmap='plasma', s=100,
            edgecolor='black', linewidth=1, alpha=0.8
        )
        
        # Add labels
        for idx, row in grouped.iterrows():
            ax.annotate(
                row['dopant'],
                (row['chi_dopant'], row['dh']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10
            )
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Dopant content, x', fontsize=11)
        
        ax.set_xlabel('Electronegativity of dopant (Pauling)', fontsize=12, fontweight='bold')
        ax.set_ylabel('∆H$_{hyd}$ (kJ mol$^{-1}$)', fontsize=12, fontweight='bold')
        ax.set_title(f'{b_matrix}O₃-based perovskites', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_compensation_effect(df):
        """Plot ∆H vs ∆S (compensation effect)"""
        
        # Filter valid data
        filtered = df[df['dh'].notna() & df['ds'].notna()].copy()
        
        if len(filtered) < 3:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Color by A-site
        a_sites = filtered['a'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(a_sites)))
        
        for a_site, color in zip(a_sites, colors):
            mask = filtered['a'] == a_site
            ax.scatter(
                filtered.loc[mask, 'ds'], filtered.loc[mask, 'dh'],
                label=a_site, color=color, s=80,
                edgecolor='black', linewidth=0.5, alpha=0.7
            )
        
        # Add trend line
        z = np.polyfit(filtered['ds'], filtered['dh'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(filtered['ds'].min(), filtered['ds'].max(), 50)
        ax.plot(x_trend, p(x_trend), '--', color='gray', alpha=0.7,
                label=f'Trend: ∆H = {z[0]:.2f}·∆S + {z[1]:.1f}')
        
        # Labels
        ax.set_xlabel('∆S$_{hyd}$ (J mol$^{-1}$ K$^{-1}$)', fontsize=12, fontweight='bold')
        ax.set_ylabel('∆H$_{hyd}$ (kJ mol$^{-1}$)', fontsize=12, fontweight='bold')
        ax.set_title('Compensation effect in proton conductors', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.invert_yaxis()
        ax.invert_xaxis()
        
        # Add correlation coefficient
        corr = filtered['dh'].corr(filtered['ds'])
        ax.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_tolerance_factor_correlation(df):
        """Plot ∆H vs Goldschmidt tolerance factor"""
        
        # Filter data
        filtered = df[df['dh'].notna() & df['t_Goldschmidt'].notna()].copy()
        
        if len(filtered) < 3:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Color by structure
        structures = filtered['structure'].unique()
        markers = {'cubic': 'o', 'hexagonal': 's', 'orthorhombic': '^', 'tetragonal': 'D', 'layered': 'v'}
        
        for structure in structures:
            mask = filtered['structure'] == structure
            marker = markers.get(structure, 'o')
            ax.scatter(
                filtered.loc[mask, 't_Goldschmidt'], filtered.loc[mask, 'dh'],
                label=structure, marker=marker, s=80,
                edgecolor='black', linewidth=0.5, alpha=0.7
            )
        
        # Add ideal perovskite region
        ax.axvspan(0.95, 1.05, alpha=0.2, color='gray', label='Ideal perovskite')
        
        # Labels
        ax.set_xlabel('Goldschmidt tolerance factor, t', fontsize=12, fontweight='bold')
        ax.set_ylabel('∆H$_{hyd}$ (kJ mol$^{-1}$)', fontsize=12, fontweight='bold')
        ax.set_title('Structure stability vs hydration', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_3d_surface(df, b_matrix='Zr', x_param='x', y_param='r_dopant', z_param='dh'):
        """Create 3D surface plot using plotly"""
        
        # Filter data
        mask = (df['b'] == b_matrix) & df[x_param].notna() & df[y_param].notna() & df[z_param].notna()
        filtered = df[mask].copy()
        
        if len(filtered) < 5:
            return None
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter3d(
            x=filtered[x_param],
            y=filtered[y_param],
            z=filtered[z_param],
            mode='markers+text',
            marker=dict(
                size=8,
                color=filtered[z_param],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="∆H (kJ/mol)")
            ),
            text=filtered['dopant'],
            textposition="top center",
            name='Experimental data'
        ))
        
        # Labels
        axis_labels = {
            'x': 'Dopant content, x',
            'r_dopant': 'Ionic radius of dopant (Å)',
            'chi_dopant': 'Electronegativity of dopant',
            'dh': '∆H<sub>hyd</sub> (kJ mol<sup>-1</sup>)',
            'ds': '∆S<sub>hyd</sub> (J mol<sup>-1</sup> K<sup>-1</sup>)'
        }
        
        fig.update_layout(
            title=f'{b_matrix}O₃-based perovskites - 3D correlation',
            scene=dict(
                xaxis_title=axis_labels.get(x_param, x_param),
                yaxis_title=axis_labels.get(y_param, y_param),
                zaxis_title=axis_labels.get(z_param, z_param),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # FIX 10: Initialize cached resources
    with st.spinner("Loading data and training models..."):
        dataset = get_dataset()
        predictor = get_predictor(dataset)
        visualizer = ScientificVisualizer()
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/ffffff/000000?text=Perovskite+Hydration+Lab", 
                 use_container_width=True)
        st.title("💧 Perovskite Hydration")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔍 Correlations", "🤖 Predictor", "📁 Data Explorer", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.caption("© 2026 | Scientific v1.0")
    
    # Main content
    if page == "📊 Dashboard":
        show_dashboard(dataset)
    elif page == "🔍 Correlations":
        show_correlations(dataset, visualizer)
    elif page == "🤖 Predictor":
        show_predictor(predictor, dataset)
    elif page == "📁 Data Explorer":
        show_data_explorer(dataset)
    elif page == "ℹ️ About":
        show_about()


def show_dashboard(dataset):
    """Dashboard page with overview"""
    
    st.title("📊 Perovskite Hydration Database")
    st.markdown("---")
    
    df = dataset.data
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_materials = len(df)
        st.metric("Total Materials", n_materials)
    
    with col2:
        n_with_dh = df['dh'].notna().sum()
        st.metric("With ∆H data", n_with_dh)
    
    with col3:
        avg_dh = df['dh'].mean()
        st.metric("Average ∆H", f"{avg_dh:.1f} kJ/mol")
    
    with col4:
        min_dh = df['dh'].min()
        max_dh = df['dh'].max()
        st.metric("∆H Range", f"[{min_dh:.0f}, {max_dh:.0f}]")
    
    st.markdown("---")
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("∆H Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        df_dh = df[df['dh'].notna()]
        ax.hist(df_dh['dh'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel("∆H$_{hyd}$ (kJ mol$^{-1}$)")
        ax.set_ylabel("Frequency")
        ax.axvline(df_dh['dh'].mean(), color='red', linestyle='--', label=f"Mean: {df_dh['dh'].mean():.1f}")
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Materials by A-site")
        a_counts = df['a'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(a_counts.index, a_counts.values, color='darkorange', edgecolor='black')
        ax.set_xlabel("A-site cation")
        ax.set_ylabel("Number of materials")
        st.pyplot(fig)
        plt.close()
    
    # Recent data
    st.subheader("Sample Data")
    display_cols = ['a', 'b', 'dopant', 'x', 'dh', 'ds', 'structure', 'ref']
    st.dataframe(df[display_cols].head(10), use_container_width=True)
    
    # Quick stats
    with st.expander("📈 Summary Statistics"):
        st.dataframe(df[['dh', 'ds', 'x', 'r_A', 'r_B', 't_Goldschmidt']].describe())


def show_correlations(dataset, visualizer):
    """Correlations page with various plots"""
    
    st.title("🔍 Scientific Correlations")
    st.markdown("---")
    
    df = dataset.data
    
    # Sidebar controls for this page
    with st.sidebar:
        st.subheader("Plot Controls")
        plot_type = st.selectbox(
            "Select correlation",
            [
                "∆H vs Dopant Content",
                "∆H vs Ionic Radius",
                "∆H vs Electronegativity",
                "Compensation Effect (∆H vs ∆S)",
                "Tolerance Factor Correlation",
                "3D Surface Plot"
            ]
        )
        
        if plot_type in ["∆H vs Dopant Content", "∆H vs Ionic Radius", "∆H vs Electronegativity", "3D Surface Plot"]:
            b_matrix = st.selectbox("B-site matrix", ["Ti", "Zr", "Sn", "Ce", "All"])
            if plot_type == "∆H vs Dopant Content":
                dopant = st.selectbox("Dopant", ["Sc", "Y", "In", "Yb", "Gd", "All"])
    
    # Generate plots
    if plot_type == "∆H vs Dopant Content":
        if b_matrix != "All" and dopant != "All":
            fig = visualizer.plot_dh_vs_dopant_content(df, b_matrix, dopant)
            if fig:
                st.pyplot(fig)
                plt.close()
            else:
                st.warning(f"Not enough data for {b_matrix}-based, {dopant}-doped system")
        else:
            st.info("Please select specific B-site and dopant")
    
    elif plot_type == "∆H vs Ionic Radius":
        if b_matrix != "All":
            fig = visualizer.plot_dh_vs_ionic_radius(df, b_matrix)
            if fig:
                st.pyplot(fig)
                plt.close()
            else:
                st.warning(f"Not enough data for {b_matrix}-based system")
        else:
            st.info("Please select specific B-site")
    
    elif plot_type == "∆H vs Electronegativity":
        if b_matrix != "All":
            fig = visualizer.plot_dh_vs_electronegativity(df, b_matrix)
            if fig:
                st.pyplot(fig)
                plt.close()
            else:
                st.warning(f"Not enough data for {b_matrix}-based system")
        else:
            st.info("Please select specific B-site")
    
    elif plot_type == "Compensation Effect (∆H vs ∆S)":
        fig = visualizer.plot_compensation_effect(df)
        if fig:
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Not enough data for compensation plot")
    
    elif plot_type == "Tolerance Factor Correlation":
        fig = visualizer.plot_tolerance_factor_correlation(df)
        if fig:
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Not enough data for tolerance factor plot")
    
    elif plot_type == "3D Surface Plot":
        if b_matrix != "All":
            x_param = st.selectbox("X-axis", ["x", "r_dopant", "chi_dopant"])
            y_param = st.selectbox("Y-axis", ["r_dopant", "chi_dopant", "x"])
            z_param = st.selectbox("Z-axis", ["dh", "ds"])
            
            if x_param != y_param:
                fig = visualizer.plot_3d_surface(df, b_matrix, x_param, y_param, z_param)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Not enough data for 3D plot")
            else:
                st.warning("X and Y axes must be different")
        else:
            st.info("Please select specific B-site")


def show_predictor(predictor, dataset):
    """Prediction page"""
    
    st.title("🤖 Hydration Predictor")
    st.markdown("---")
    
    # Check if models are trained
    if not predictor.is_trained:
        st.warning("Models are being trained. Please wait...")
        return
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Enter material composition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            a_cation = st.selectbox(
                "A-site cation",
                ["Ba", "Sr", "La", "Ca", "Nd", "Gd"]
            )
            b_cation = st.selectbox(
                "B-site cation",
                ["Zr", "Ce", "Sn", "Ti", "Hf"]
            )
            structure = st.selectbox(
                "Crystal structure",
                ["cubic", "hexagonal", "orthorhombic", "tetragonal", "layered"]
            )
        
        with col2:
            dopant = st.selectbox(
                "Dopant",
                ["Y", "Sc", "In", "Yb", "Gd", "Er", "Lu", ""]
            )
            x = st.slider(
                "Dopant content (x)",
                min_value=0.0, max_value=0.8, value=0.2, step=0.05
            )
        
        submitted = st.form_submit_button("Predict Hydration Parameters", type="primary")
    
    if submitted:
        with st.spinner("Calculating..."):
            # Handle empty dopant
            if dopant == "":
                dopant = None
            
            # Make prediction
            dh_mean, dh_std, ds_mean, ds_std = predictor.predict(
                a_cation, b_cation, dopant, x, structure
            )
            
            if dh_mean is not None:
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "∆H$_{hyd}$",
                        f"{dh_mean:.1f} ± {dh_std:.1f} kJ/mol",
                        delta=None
                    )
                    
                    # Interpret ∆H
                    if dh_mean < -100:
                        interpretation = "Strong hydration (exothermic)"
                    elif dh_mean < -70:
                        interpretation = "Moderate hydration"
                    elif dh_mean < -40:
                        interpretation = "Weak hydration"
                    else:
                        interpretation = "Very weak hydration"
                    
                    st.caption(f"Interpretation: {interpretation}")
                
                with col2:
                    st.metric(
                        "∆S$_{hyd}$",
                        f"{ds_mean:.1f} ± {ds_std:.1f} J/mol·K",
                        delta=None
                    )
                    
                    # Interpret ∆S
                    if ds_mean < -120:
                        interp_s = "High ordering upon hydration"
                    elif ds_mean < -90:
                        interp_s = "Moderate ordering"
                    else:
                        interp_s = "Low ordering"
                    
                    st.caption(f"Interpretation: {interp_s}")
                
                # Find similar materials
                if dopant:
                    st.markdown("---")
                    st.subheader("Similar Materials in Database")
                    
                    similar = predictor.find_similar_materials(
                        a_cation, b_cation, dopant, x, n_neighbors=5
                    )
                    
                    if not similar.empty:
                        st.dataframe(similar, use_container_width=True)
                    else:
                        st.info("No similar materials found")
                
                # Add to database option
                st.markdown("---")
                st.caption("Note: This is a research tool. Predictions should be verified experimentally.")
                
            else:
                st.error("Could not make prediction. Please check input parameters.")


def show_data_explorer(dataset):
    """Data explorer page"""
    
    st.title("📁 Data Explorer")
    st.markdown("---")
    
    df = dataset.data
    
    # Filters
    with st.expander("🔍 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            a_filter = st.multiselect("A-site", options=df['a'].unique())
        with col2:
            b_filter = st.multiselect("B-site", options=df['b'].unique())
        with col3:
            dopant_filter = st.multiselect("Dopant", options=[d for d in df['dopant'].unique() if d])
        
        col4, col5 = st.columns(2)
        with col4:
            min_dh = float(df['dh'].min()) if not df['dh'].isna().all() else -200
            max_dh = float(df['dh'].max()) if not df['dh'].isna().all() else 0
            dh_range = st.slider(
                "∆H range (kJ/mol)",
                min_dh, max_dh,
                (min_dh, max_dh)
            )
        with col5:
            x_range = st.slider(
                "Dopant content range",
                0.0, 1.0, (0.0, 1.0)
            )
    
    # Apply filters
    filtered = df.copy()
    if a_filter:
        filtered = filtered[filtered['a'].isin(a_filter)]
    if b_filter:
        filtered = filtered[filtered['b'].isin(b_filter)]
    if dopant_filter:
        filtered = filtered[filtered['dopant'].isin(dopant_filter)]
    
    filtered = filtered[
        (filtered['dh'] >= dh_range[0]) & (filtered['dh'] <= dh_range[1]) &
        (filtered['x'] >= x_range[0]) & (filtered['x'] <= x_range[1])
    ]
    
    # Display
    st.subheader(f"Materials: {len(filtered)}")
    
    display_cols = ['a', 'b', 'dopant', 'x', 'dh', 'ds', 'structure', 
                    'r_B_avg', 't_Goldschmidt', 'chi_diff', 'ref']
    
    st.dataframe(filtered[display_cols], use_container_width=True)
    
    # Download button
    if not filtered.empty:
        csv = filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="perovskite_hydration_data.csv",
            mime="text/csv"
        )


def show_about():
    """About page"""
    
    st.title("ℹ️ About")
    st.markdown("---")
    
    about_text = '''
    ## Perovskite Hydration Predictor
    
    This application provides a comprehensive database and machine learning tools 
    for predicting hydration thermodynamics in proton-conducting perovskite materials.
    
    ### Features
    
    - **📊 Dashboard**: Overview of the database with key statistics
    - **🔍 Correlations**: Scientific visualizations of structure-property relationships
    - **🤖 Predictor**: ML-based prediction of ∆H and ∆S for new compositions
    - **📁 Data Explorer**: Filter and export the database
    
    ### Data Sources
    
    - Compiled from over 50 scientific publications
    - Includes BaZrO₃, BaCeO₃, BaSnO₃, BaTiO₃, and La-based perovskites
    - Also contains layered and double perovskite structures
    
    ### Descriptors Used
    
    - **Ionic radii**: A-site (XII/IX), B-site (VI), dopant (VI)
    - **Electronegativity**: Pauling scale for all elements
    - **Structural**: Goldschmidt tolerance factor
    - **Compositional**: Dopant content
    
    ### Models
    
    - Random Forest Regression
    - Gradient Boosting
    - Ensemble averaging with uncertainty estimation
    
    ### Citation
    
    If you use this tool in your research, please cite'''
    

