"""
utils.py - Data Loading and Time Conversion Utilities
======================================================
Handles CDF file loading and epoch time conversion for MMS mission data.
"""

import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
import cdflib
from cdflib import cdfepoch
import tempfile
import os


class CDFLoader:
    """
    CDF File Loader for NASA MMS Mission Data.
    
    Handles file loading, variable extraction, and epoch conversion.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the CDF loader with a file path.
        
        Args:
            file_path: Path to the CDF file
        """
        self.file_path = file_path
        self.cdf = cdflib.CDF(file_path)
        self._info = self.cdf.cdf_info()
        self._time_var = None
        self._time_data = None
    
    @classmethod
    def from_uploaded_file(cls, uploaded_file) -> 'CDFLoader':
        """
        Create a CDFLoader from a Streamlit uploaded file.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            CDFLoader instance
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.cdf') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        return cls(tmp_path)
    
    def cleanup(self):
        """Remove temporary file if created from upload."""
        try:
            if self.file_path and os.path.exists(self.file_path):
                if '/tmp' in self.file_path or 'temp' in self.file_path.lower():
                    os.unlink(self.file_path)
        except Exception:
            pass
    
    @property
    def z_variables(self) -> List[str]:
        """Get list of zVariables in the CDF file."""
        return list(getattr(self._info, 'zVariables', []) or [])
    
    @property
    def r_variables(self) -> List[str]:
        """Get list of rVariables in the CDF file."""
        return list(getattr(self._info, 'rVariables', []) or [])
    
    @property
    def all_variables(self) -> List[str]:
        """Get all variables in the CDF file."""
        return self.z_variables + self.r_variables
    
    def get_global_attributes(self) -> Dict[str, Any]:
        """Get global attributes from the CDF file."""
        try:
            return self.cdf.globalattsget()
        except Exception:
            return {}
    
    def get_variable_attributes(self, var_name: str) -> Dict[str, Any]:
        """Get attributes for a specific variable."""
        try:
            return self.cdf.varattsget(var_name)
        except Exception:
            return {}
    
    def get_variable_data(self, var_name: str) -> Optional[np.ndarray]:
        """
        Get data for a specific variable.
        
        Args:
            var_name: Name of the variable
            
        Returns:
            NumPy array of variable data or None if not found
        """
        try:
            return self.cdf.varget(var_name)
        except Exception:
            return None
    
    def detect_time_variable(self) -> Optional[str]:
        """
        Automatically detect the time/epoch variable.
        
        Returns:
            Name of the time variable or None if not found
        """
        if self._time_var is not None:
            return self._time_var
        
        # Priority list for time variable names
        time_candidates = [
            'Epoch', 'Epoch_TT2000', 'epoch', 'EPOCH',
            'time', 'Time', 'TIME', 'unix_time', 'Unix_Time',
            'tt2000', 'TT2000'
        ]
        
        all_vars = self.all_variables
        
        for candidate in time_candidates:
            if candidate in all_vars:
                self._time_var = candidate
                return candidate
        
        # Fallback: look for variables containing 'epoch' or 'time'
        for var in all_vars:
            if 'epoch' in var.lower() or 'time' in var.lower():
                self._time_var = var
                return var
        
        return None
    
    def get_time_data(self, convert_to_datetime: bool = True) -> Optional[np.ndarray]:
        """
        Get time data, optionally converted to datetime objects.
        
        Args:
            convert_to_datetime: If True, convert to datetime64
            
        Returns:
            NumPy array of time data
        """
        time_var = self.detect_time_variable()
        if time_var is None:
            return None
        
        if self._time_data is not None and convert_to_datetime:
            return self._time_data
        
        raw_epoch = self.cdf.varget(time_var)
        
        if convert_to_datetime:
            self._time_data = convert_epoch_to_datetime(raw_epoch)
            return self._time_data
        
        return raw_epoch
    
    def classify_variable(self, var_name: str) -> Dict[str, Any]:
        """
        Classify a variable by its type and shape.
        
        Args:
            var_name: Name of the variable
            
        Returns:
            Dictionary with 'type' ('scalar', 'vector', 'matrix', 'unknown'),
            'shape', 'n_components', and 'is_plottable'
        """
        data = self.get_variable_data(var_name)
        if data is None:
            return {'type': 'unknown', 'shape': None, 'n_components': 0, 'is_plottable': False}
        
        shape = data.shape
        
        if len(shape) == 1:
            return {
                'type': 'scalar',
                'shape': shape,
                'n_components': 1,
                'is_plottable': True
            }
        elif len(shape) == 2:
            n_components = shape[1]
            if n_components <= 4:
                return {
                    'type': 'vector',
                    'shape': shape,
                    'n_components': n_components,
                    'is_plottable': True
                }
            else:
                return {
                    'type': 'matrix',
                    'shape': shape,
                    'n_components': n_components,
                    'is_plottable': False
                }
        else:
            return {
                'type': 'multidimensional',
                'shape': shape,
                'n_components': shape[1] if len(shape) > 1 else 1,
                'is_plottable': False
            }
    
    def get_plottable_variables(self) -> List[str]:
        """Get list of variables suitable for plotting (1D or small 2D)."""
        time_var = self.detect_time_variable()
        plottable = []
        
        for var in self.z_variables:
            if var == time_var:
                continue
            
            info = self.classify_variable(var)
            if info['is_plottable']:
                plottable.append(var)
        
        return plottable
    
    def get_physics_variables(self) -> Dict[str, List[str]]:
        """
        Categorize variables by physical type (B-field, velocity, etc.).
        
        Returns:
            Dictionary with categories as keys and variable lists as values
        """
        categories = {
            'magnetic_field': [],
            'electric_field': [],
            'velocity': [],
            'density': [],
            'temperature': [],
            'pressure': [],
            'other': []
        }
        
        patterns = {
            'magnetic_field': ['_b_', 'bfield', 'b_gse', 'b_gsm', 'fgm', 'afg', 'dfg'],
            'electric_field': ['_e_', 'efield', 'e_gse', 'e_gsm', 'edp'],
            'velocity': ['velocity', 'vel_', '_v_', 'vi_', 've_', 'bulkv'],
            'density': ['density', 'numberdensity', '_n_', 'ni_', 'ne_'],
            'temperature': ['temp', 't_para', 't_perp', 'ti_', 'te_'],
            'pressure': ['press', 'pres_']
        }
        
        for var in self.get_plottable_variables():
            var_lower = var.lower()
            categorized = False
            
            for category, pattern_list in patterns.items():
                if any(p in var_lower for p in pattern_list):
                    categories[category].append(var)
                    categorized = True
                    break
            
            if not categorized:
                categories['other'].append(var)
        
        return categories


def convert_epoch_to_datetime(epoch_data: np.ndarray) -> np.ndarray:
    """
    Convert CDF epoch data to numpy datetime64.
    
    Handles TT2000, Epoch, and Epoch16 formats automatically.
    
    Args:
        epoch_data: Raw epoch data from CDF file
        
    Returns:
        NumPy array of datetime64 values
    """
    try:
        # cdfepoch.to_datetime handles all epoch types
        # Convert to list of datetime, then to numpy array
        dt_list = cdfepoch.to_datetime(epoch_data)
        return np.array(dt_list, dtype='datetime64[ns]')
    except Exception as e:
        raise ValueError(f"Failed to convert epoch data: {e}")


def calculate_sampling_frequency(time_data: np.ndarray) -> float:
    """
    Calculate sampling frequency from time array.
    
    Args:
        time_data: Array of datetime64 or datetime objects
        
    Returns:
        Sampling frequency in Hz
    """
    if len(time_data) < 2:
        raise ValueError("Need at least 2 time points to calculate sampling frequency")
    
    # Convert to numpy datetime64 if needed
    if isinstance(time_data[0], datetime):
        time_data = np.array(time_data, dtype='datetime64[ns]')
    
    # Calculate mean time delta in seconds
    dt = np.diff(time_data).astype('timedelta64[ns]').astype(float) / 1e9
    
    # Use median to be robust against outliers
    mean_dt = np.median(dt)
    
    if mean_dt <= 0:
        raise ValueError("Invalid time delta (zero or negative)")
    
    return 1.0 / mean_dt


def extract_component(data: np.ndarray, component: str) -> np.ndarray:
    """
    Extract a component from vector data.
    
    Args:
        data: 2D array of shape (N, 3) or (N, 4)
        component: One of 'X', 'Y', 'Z', 'Magnitude', or index 0-3
        
    Returns:
        1D array of the selected component
    """
    if len(data.shape) == 1:
        return data
    
    component_map = {'X': 0, 'Y': 1, 'Z': 2, 'W': 3, 'T': 3}
    
    if component == 'Magnitude':
        # Calculate magnitude
        if data.shape[1] >= 3:
            return np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2)
        else:
            return np.sqrt(np.sum(data**2, axis=1))
    
    if component in component_map:
        idx = component_map[component]
    else:
        try:
            idx = int(component)
        except ValueError:
            raise ValueError(f"Unknown component: {component}")
    
    if idx >= data.shape[1]:
        raise ValueError(f"Component index {idx} out of range for data with {data.shape[1]} components")
    
    return data[:, idx]
