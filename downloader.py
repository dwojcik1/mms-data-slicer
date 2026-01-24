"""
downloader.py - NASA CDAWeb Data Download Module
==================================================
Download MMS mission data directly from NASA CDAWeb.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional
import warnings
import tempfile
import os


def load_fgm_cdasws(
    trange: List[str],
    probe: str = '1',
    data_rate: str = 'srvy',
    level: str = 'l2',
    coord: str = 'gse'
) -> pd.DataFrame:
    """
    Download MMS FGM (Fluxgate Magnetometer) data using CDAWeb API.
    
    Downloads the CDF file and parses it directly with cdflib to avoid
    cdasws internal module detection issues.
    
    Args:
        trange: Time range as ['YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS']
        probe: MMS spacecraft number ('1', '2', '3', or '4')
        data_rate: Data rate ('brst', 'fast', 'slow', 'srvy')
        level: Data level ('l2', 'l1b', 'ql')
        coord: Coordinate system ('gse', 'gsm', 'dmpa', 'bcs')
    
    Returns:
        DataFrame with DatetimeIndex and columns ['Bx', 'By', 'Bz', 'Bt']
    """
    try:
        from cdasws import CdasWs
        import cdflib
    except ImportError as e:
        raise ImportError(f"Required modules not installed: {e}")
    
    # Initialize CDAWeb client
    cdas = CdasWs()
    
    # Construct dataset ID: MMS1_FGM_SRVY_L2
    dataset = f"MMS{probe}_FGM_{data_rate.upper()}_{level.upper()}"
    
    # Variable name: mms1_fgm_b_gse_srvy_l2
    var_name = f"mms{probe}_fgm_b_{coord}_{data_rate}_{level}"
    epoch_var = f"Epoch"
    
    # Parse time range
    start_time = datetime.strptime(trange[0], '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(trange[1], '%Y-%m-%d %H:%M:%S')
    
    # Get the file URLs instead of data directly
    try:
        status_code, result = cdas.get_data_file(
            dataset,
            [var_name],
            start_time,
            end_time
        )
    except Exception as e:
        raise ValueError(f"Failed to get file list from CDAWeb: {e}")
    
    if not result or 'FileDescription' not in result:
        raise ValueError(f"No data files found for {dataset} in range {trange}")
    
    file_descriptions = result.get('FileDescription', [])
    if not file_descriptions:
        raise ValueError(f"No data files available for {dataset}")

    
    # Download and process each CDF file
    all_times = []
    all_values = []
    
    for file_desc in file_descriptions:
        file_url = file_desc.get('Name')
        if not file_url:
            continue
        
        # Download CDF file to temp location
        import urllib.request
        
        with tempfile.NamedTemporaryFile(suffix='.cdf', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            urllib.request.urlretrieve(file_url, tmp_path)
            
            # Read with cdflib
            cdf = cdflib.CDF(tmp_path)
            
            # Get epoch data
            epoch_data = None
            for possible_epoch in ['Epoch', 'epoch', f'mms{probe}_fgm_epoch_{data_rate}_{level}']:
                try:
                    epoch_data = cdf.varget(possible_epoch)
                    break
                except:
                    continue
            
            if epoch_data is None:
                # Try to find any epoch variable
                info = cdf.cdf_info()
                for zvar in getattr(info, 'zVariables', []):
                    if 'epoch' in zvar.lower():
                        epoch_data = cdf.varget(zvar)
                        break
            
            if epoch_data is None:
                continue
            
            # Get field data
            try:
                field_data = cdf.varget(var_name)
            except:
                # Try alternate naming
                alt_var = f"mms{probe}_fgm_b_{coord}_{data_rate}_{level}"
                try:
                    field_data = cdf.varget(alt_var)
                except:
                    continue
            
            # Convert epoch to datetime
            times = cdflib.cdfepoch.to_datetime(epoch_data)
            
            # Filter to requested time range
            times_np = np.array(times, dtype='datetime64[ns]')
            start_np = np.datetime64(start_time)
            end_np = np.datetime64(end_time)
            
            mask = (times_np >= start_np) & (times_np <= end_np)
            times_filtered = times_np[mask]
            values_filtered = field_data[mask]
            
            if len(times_filtered) > 0:
                all_times.append(times_filtered)
                all_values.append(values_filtered)
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    if not all_times:
        raise ValueError(
            f"No FGM data found for MMS{probe} ({data_rate}/{level}) "
            f"in range {trange[0]} to {trange[1]}"
        )
    
    # Concatenate all data
    times = np.concatenate(all_times)
    values = np.concatenate(all_values)
    
    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    values = values[sort_idx]
    
    # Create DataFrame
    datetime_index = pd.to_datetime(times)
    
    if len(values.shape) == 1:
        columns = ['Bt']
        values = values.reshape(-1, 1)
    elif values.shape[1] == 3:
        columns = ['Bx', 'By', 'Bz']
    elif values.shape[1] == 4:
        columns = ['Bx', 'By', 'Bz', 'Bt']
    else:
        columns = [f'B{i}' for i in range(values.shape[1])]
    
    df = pd.DataFrame(values, index=datetime_index, columns=columns)
    df.index.name = 'time'
    
    # Add magnitude if not present
    if 'Bt' not in df.columns and 'Bx' in df.columns:
        df['Bt'] = np.sqrt(df['Bx']**2 + df['By']**2 + df['Bz']**2)
    
    # Clean fill values
    df = df.replace(-1e31, np.nan)
    
    return df


def check_cdasws_available() -> bool:
    """Check if cdasws is installed and importable."""
    try:
        from cdasws import CdasWs
        import cdflib
        return True
    except ImportError:
        return False


def format_trange(start_date, start_time, end_date, end_time) -> List[str]:
    """Format date/time inputs into trange format."""
    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)
    
    return [
        start_dt.strftime('%Y-%m-%d %H:%M:%S'),
        end_dt.strftime('%Y-%m-%d %H:%M:%S')
    ]
