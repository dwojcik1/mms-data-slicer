"""
downloader.py - NASA CDAWeb Data Download Module
==================================================
Download MMS mission data directly from NASA CDAWeb using cdasws.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional
import warnings


def load_fgm_cdasws(
    trange: List[str],
    probe: str = '1',
    data_rate: str = 'srvy',
    level: str = 'l2',
    coord: str = 'gse'
) -> pd.DataFrame:
    """
    Download MMS FGM (Fluxgate Magnetometer) data using CDAWeb API.
    
    Args:
        trange: Time range as ['YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS']
        probe: MMS spacecraft number ('1', '2', '3', or '4')
        data_rate: Data rate ('brst', 'fast', 'slow', 'srvy')
        level: Data level ('l2', 'l1b', 'ql')
        coord: Coordinate system ('gse', 'gsm', 'dmpa', 'bcs')
    
    Returns:
        DataFrame with DatetimeIndex and columns ['Bx', 'By', 'Bz', 'Bt']
    
    Raises:
        ImportError: If cdasws not installed
        ValueError: If no data found for the specified parameters
    """
    try:
        from cdasws import CdasWs
        from cdasws.datarepresentation import DataRepresentation
    except ImportError as e:
        raise ImportError(
            "cdasws not installed. Install with: pip install cdasws\n"
            f"Original error: {e}"
        )
    
    # Initialize CDAWeb client
    cdas = CdasWs()
    
    # Construct dataset ID
    # Format: MMS1_FGM_SRVY_L2
    dataset = f"MMS{probe}_FGM_{data_rate.upper()}_{level.upper()}"
    
    # Variable name for magnetic field
    # Format: mms1_fgm_b_gse_srvy_l2
    var_name = f"mms{probe}_fgm_b_{coord}_{data_rate}_{level}"
    
    # Parse time range
    start_time = datetime.strptime(trange[0], '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(trange[1], '%Y-%m-%d %H:%M:%S')
    
    try:
        # Get data from CDAWeb in xarray format
        status, data = cdas.get_data(
            dataset,
            [var_name],
            start_time,
            end_time,
            dataRepresentation=DataRepresentation.XARRAY
        )
    except Exception as e:
        raise ValueError(
            f"Failed to download data from CDAWeb: {e}\n"
            f"Dataset: {dataset}, Variable: {var_name}"
        )
    
    if data is None:
        raise ValueError(
            f"No FGM data found for MMS{probe} "
            f"({data_rate}/{level}) in range {trange[0]} to {trange[1]}. "
            "Check time range and data availability."
        )
    
    # Extract data from xarray Dataset
    if var_name not in data:
        # Try to find a matching variable
        matching_vars = [k for k in data.keys() if 'fgm' in k.lower() and 'b_' in k.lower()]
        if matching_vars:
            var_name = matching_vars[0]
        else:
            raise ValueError(f"Variable {var_name} not found. Available: {list(data.keys())}")
    
    # Get the data array
    da = data[var_name]
    
    # Extract time coordinate
    time_dim = None
    for dim in da.dims:
        if 'epoch' in dim.lower() or 'time' in dim.lower():
            time_dim = dim
            break
    
    if time_dim is None:
        # Use the first dimension as time
        time_dim = da.dims[0]
    
    times = da.coords[time_dim].values
    values = da.values
    
    # Convert times to datetime
    datetime_index = pd.to_datetime(times)
    
    # Determine column names based on shape
    if len(values.shape) == 1:
        columns = ['Bt']
        values = values.reshape(-1, 1)
    elif values.shape[1] == 3:
        columns = ['Bx', 'By', 'Bz']
    elif values.shape[1] == 4:
        columns = ['Bx', 'By', 'Bz', 'Bt']
    else:
        columns = [f'B{i}' for i in range(values.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(values, index=datetime_index, columns=columns)
    df.index.name = 'time'
    
    # Add magnitude if not present
    if 'Bt' not in df.columns and 'Bx' in df.columns:
        df['Bt'] = np.sqrt(df['Bx']**2 + df['By']**2 + df['Bz']**2)
    
    # Clean NaN/fill values
    df = df.replace(-1e31, np.nan)  # Common CDF fill value
    
    return df


def check_cdasws_available() -> bool:
    """Check if cdasws is installed and importable."""
    try:
        from cdasws import CdasWs
        from cdasws.datarepresentation import DataRepresentation
        return True
    except ImportError:
        return False


def format_trange(start_date, start_time, end_date, end_time) -> List[str]:
    """
    Format date/time inputs into trange format.
    
    Args:
        start_date: datetime.date object
        start_time: datetime.time object
        end_date: datetime.date object
        end_time: datetime.time object
    
    Returns:
        List of two strings in 'YYYY-MM-DD HH:MM:SS' format
    """
    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)
    
    return [
        start_dt.strftime('%Y-%m-%d %H:%M:%S'),
        end_dt.strftime('%Y-%m-%d %H:%M:%S')
    ]


def list_available_datasets(probe: str = '1') -> List[str]:
    """List available MMS datasets for a given probe."""
    try:
        from cdasws import CdasWs
        cdas = CdasWs()
        datasets = cdas.get_datasets(observatoryGroup='MMS')
        mms_datasets = [d['Id'] for d in datasets if f'MMS{probe}' in d['Id']]
        return sorted(mms_datasets)
    except Exception:
        return []
