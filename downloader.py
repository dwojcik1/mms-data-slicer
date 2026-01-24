"""
downloader.py - PySPEDAS Data Download Module
===============================================
Download MMS mission data directly from NASA CDAWeb.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple
import warnings


def load_fgm_pyspedas(
    trange: List[str],
    probe: str = '1',
    data_rate: str = 'srvy',
    level: str = 'l2',
    coord: str = 'gse'
) -> pd.DataFrame:
    """
    Download MMS FGM (Fluxgate Magnetometer) data using PySPEDAS.
    
    Args:
        trange: Time range as ['YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS']
        probe: MMS spacecraft number ('1', '2', '3', or '4')
        data_rate: Data rate ('brst', 'fast', 'slow', 'srvy')
        level: Data level ('l2', 'l1b', 'ql')
        coord: Coordinate system ('gse', 'gsm', 'dmpa', 'bcs')
    
    Returns:
        DataFrame with DatetimeIndex and columns ['Bx', 'By', 'Bz', 'Bt']
    
    Raises:
        ImportError: If pyspedas/pytplot not installed
        ValueError: If no data found for the specified parameters
    """
    # Import here to avoid hard dependency
    try:
        import pyspedas
        from pyspedas.mms import fgm
        import pytplot
    except ImportError as e:
        raise ImportError(
            "PySPEDAS not installed. Install with: pip install pyspedas\n"
            f"Original error: {e}"
        )
    
    # Suppress pyspedas warnings during load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Call pyspedas FGM loader
        loaded_vars = fgm(
            trange=trange,
            probe=probe,
            data_rate=data_rate,
            level=level,
            time_clip=True,
            latest_version=True
        )
    
    if not loaded_vars:
        raise ValueError(
            f"No FGM data found for MMS{probe} "
            f"({data_rate}/{level}) in range {trange[0]} to {trange[1]}. "
            "Check time range and data availability on CDAWeb."
        )
    
    # Construct expected variable name pattern
    var_name = f"mms{probe}_fgm_b_{coord}_{data_rate}_{level}"
    
    # Try to find the variable in loaded list
    target_var = None
    for v in loaded_vars:
        if coord in v.lower() and 'fgm' in v.lower():
            target_var = v
            break
    
    # Fallback: try exact name
    if target_var is None:
        if var_name in loaded_vars:
            target_var = var_name
        elif loaded_vars:
            # Use first loaded variable
            target_var = loaded_vars[0]
    
    if target_var is None:
        raise ValueError(
            f"Could not identify FGM variable. Loaded: {loaded_vars}"
        )
    
    # Extract data using pytplot
    try:
        data = pytplot.get_data(target_var)
    except Exception as e:
        raise ValueError(f"Failed to extract data from tplot variable: {e}")
    
    if data is None:
        raise ValueError(f"No data in tplot variable: {target_var}")
    
    times = data.times
    values = data.y
    
    if times is None or values is None or len(times) == 0:
        raise ValueError("Empty data arrays returned from PySPEDAS")
    
    # Convert times (Unix epoch) to datetime
    datetime_index = pd.to_datetime(times, unit='s', utc=True)
    
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
    if 'Bt' not in df.columns and len(df.columns) >= 3:
        df['Bt'] = np.sqrt(df['Bx']**2 + df['By']**2 + df['Bz']**2)
    
    # Clean up tplot variables to free memory
    try:
        for v in loaded_vars:
            pytplot.del_data(v)
    except Exception:
        pass
    
    return df


def load_fpi_pyspedas(
    trange: List[str],
    probe: str = '1',
    data_rate: str = 'fast',
    level: str = 'l2',
    datatype: str = 'dis-moms'
) -> pd.DataFrame:
    """
    Download MMS FPI (Fast Plasma Investigation) data using PySPEDAS.
    
    Args:
        trange: Time range as ['YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS']
        probe: MMS spacecraft number
        data_rate: Data rate ('fast', 'brst')
        level: Data level
        datatype: 'dis-moms' for ions, 'des-moms' for electrons
    
    Returns:
        DataFrame with velocity and density data
    """
    try:
        import pyspedas
        from pyspedas.mms import fpi
        import pytplot
    except ImportError as e:
        raise ImportError(f"PySPEDAS not installed: {e}")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded_vars = fpi(
            trange=trange,
            probe=probe,
            data_rate=data_rate,
            level=level,
            datatype=datatype,
            time_clip=True
        )
    
    if not loaded_vars:
        raise ValueError(f"No FPI data found for the specified parameters")
    
    # Extract bulk velocity and density
    species = 'i' if 'dis' in datatype else 'e'
    vel_var = f"mms{probe}_d{species}s_bulkv_gse_{data_rate}"
    n_var = f"mms{probe}_d{species}s_numberdensity_{data_rate}"
    
    result_dfs = []
    
    for var_pattern, prefix in [(vel_var, 'V'), (n_var, 'N')]:
        for v in loaded_vars:
            if var_pattern in v or prefix.lower() in v.lower():
                try:
                    data = pytplot.get_data(v)
                    if data is not None and data.times is not None:
                        times = pd.to_datetime(data.times, unit='s', utc=True)
                        if len(data.y.shape) == 1:
                            df = pd.DataFrame({prefix: data.y}, index=times)
                        else:
                            cols = [f'{prefix}{c}' for c in ['x', 'y', 'z'][:data.y.shape[1]]]
                            df = pd.DataFrame(data.y, index=times, columns=cols)
                        result_dfs.append(df)
                except Exception:
                    continue
    
    # Cleanup
    try:
        for v in loaded_vars:
            pytplot.del_data(v)
    except Exception:
        pass
    
    if not result_dfs:
        raise ValueError("Could not extract any FPI data")
    
    # Merge all dataframes
    result = result_dfs[0]
    for df in result_dfs[1:]:
        result = result.join(df, how='outer')
    
    result.index.name = 'time'
    return result


def check_pyspedas_available() -> bool:
    """Check if PySPEDAS is installed and importable."""
    try:
        import pyspedas
        import pytplot
        return True
    except ImportError:
        return False


def format_trange(start_date, start_time, end_date, end_time) -> List[str]:
    """
    Format date/time inputs into PySPEDAS trange format.
    
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
