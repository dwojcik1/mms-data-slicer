"""
downloader.py - NASA CDAWeb Data Download Module
==================================================
Download MMS mission data directly from NASA CDAWeb.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import warnings
import tempfile
import os
import streamlit as st
import zipfile
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib
import json
import pickle
from pathlib import Path

# ============================================================================
# Server-Side Caching Configuration
# ============================================================================

# Cache directory for downloaded data
CACHE_DIR = Path(os.environ.get('MMS_CACHE_DIR', '.cache/mms_data'))
CACHE_MAX_AGE_DAYS = int(os.environ.get('MMS_CACHE_MAX_AGE_DAYS', '30'))  # Keep files for 30 days
CACHE_MAX_SIZE_MB = int(os.environ.get('MMS_CACHE_MAX_SIZE_MB', '5000'))  # Max 5GB cache

def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR

def _generate_cache_key(params: Dict[str, Any]) -> str:
    """Generate a unique cache key from download parameters."""
    # Create a deterministic string representation
    key_str = json.dumps(params, sort_keys=True, default=str)
    # Hash it for a compact, safe filename
    return hashlib.md5(key_str.encode()).hexdigest()

def _get_cache_path(cache_key: str) -> Path:
    """Get the full path for a cached file."""
    return CACHE_DIR / f"{cache_key}.pkl"

def _get_cache_metadata_path(cache_key: str) -> Path:
    """Get the path for cache metadata."""
    return CACHE_DIR / f"{cache_key}.json"

def _is_cache_valid(cache_key: str) -> bool:
    """Check if cached data exists and is not expired."""
    cache_path = _get_cache_path(cache_key)
    metadata_path = _get_cache_metadata_path(cache_key)
    
    if not cache_path.exists() or not metadata_path.exists():
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if cache is expired
        cached_time = datetime.fromisoformat(metadata['cached_at'])
        max_age = timedelta(days=CACHE_MAX_AGE_DAYS)
        
        if datetime.now() - cached_time > max_age:
            # Cache expired, clean it up
            _remove_from_cache(cache_key)
            return False
        
        return True
    except (json.JSONDecodeError, KeyError, ValueError):
        # Corrupted metadata, clean up
        _remove_from_cache(cache_key)
        return False

def _save_to_cache(cache_key: str, data: Any, params: Dict[str, Any]):
    """Save data to cache with metadata."""
    _ensure_cache_dir()
    
    cache_path = _get_cache_path(cache_key)
    metadata_path = _get_cache_metadata_path(cache_key)
    
    # Clean up old cache if needed
    _cleanup_cache_if_needed()
    
    # Save the data
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    # Save metadata
    metadata = {
        'cached_at': datetime.now().isoformat(),
        'params': params,
        'size_bytes': cache_path.stat().st_size
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

def _load_from_cache(cache_key: str) -> Any:
    """Load data from cache."""
    cache_path = _get_cache_path(cache_key)
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def _remove_from_cache(cache_key: str):
    """Remove cached data and metadata."""
    cache_path = _get_cache_path(cache_key)
    metadata_path = _get_cache_metadata_path(cache_key)
    
    try:
        if cache_path.exists():
            cache_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
    except OSError:
        pass

def _cleanup_cache_if_needed():
    """Clean up old cache files if total size exceeds limit."""
    if not CACHE_DIR.exists():
        return
    
    # Get all cache files with their ages
    cache_files = []
    total_size = 0
    
    for json_file in CACHE_DIR.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            cache_key = json_file.stem
            cache_path = _get_cache_path(cache_key)
            
            if cache_path.exists():
                cached_time = datetime.fromisoformat(metadata['cached_at'])
                size = cache_path.stat().st_size
                total_size += size
                cache_files.append((cache_key, cached_time, size))
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            continue
    
    # Check if we need to clean up
    total_size_mb = total_size / (1024 * 1024)
    
    if total_size_mb > CACHE_MAX_SIZE_MB:
        # Sort by age (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        # Remove oldest files until we're under the limit
        target_size = CACHE_MAX_SIZE_MB * 0.8 * 1024 * 1024  # Target 80% of max
        
        for cache_key, _, size in cache_files:
            if total_size <= target_size:
                break
            _remove_from_cache(cache_key)
            total_size -= size


def _download_and_process_cdf(
    file_url: str,
    start_time: datetime,
    end_time: datetime,
    probe: str,
    data_rate: str,
    level: str,
    coord: str,
    var_name: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Download and process a single CDF file.
    
    Returns:
        Tuple of (times_filtered, values_filtered) or (None, None) on error
    """
    import cdflib
    
    # Create session with retry strategy and connection pooling
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=5, pool_maxsize=10))
    
    with tempfile.NamedTemporaryFile(suffix='.cdf', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Download with streaming for memory efficiency
        response = session.get(file_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(tmp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
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
            return None, None
        
        # Get field data
        try:
            field_data = cdf.varget(var_name)
        except:
            # Try alternate naming
            alt_var = f"mms{probe}_fgm_b_{coord}_{data_rate}_{level}"
            try:
                field_data = cdf.varget(alt_var)
            except:
                return None, None
        
        # Convert epoch to datetime - use encode for safe TT2000 conversion
        times = cdflib.cdfepoch.encode(epoch_data)
        
        # Filter to requested time range
        times_np = np.array(times, dtype='datetime64[ns]')
        start_np = np.datetime64(start_time)
        end_np = np.datetime64(end_time)
        
        mask = (times_np >= start_np) & (times_np <= end_np)
        times_filtered = times_np[mask]
        values_filtered = field_data[mask]
        
        if len(times_filtered) > 0:
            return times_filtered, values_filtered
        return None, None
        
    except Exception as e:
        warnings.warn(f"Failed to process file {file_url}: {e}")
        return None, None
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
        session.close()


@st.cache_data(show_spinner="Loading FGM data...")
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
    cdasws internal module detection issues. Uses server-side caching and 
    parallel downloads for improved performance.
    
    Args:
        trange: Time range as ['YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS']
        probe: MMS spacecraft number ('1', '2', '3', or '4')
        data_rate: Data rate ('brst', 'fast', 'slow', 'srvy')
        level: Data level ('l2', 'l1b', 'ql')
        coord: Coordinate system ('gse', 'gsm', 'dmpa', 'bcs')
    
    Returns:
        DataFrame with DatetimeIndex and columns ['Bx', 'By', 'Bz', 'Bt']
    """
    # Check cache first
    cache_params = {
        'instrument': 'fgm',
        'trange': trange,
        'probe': probe,
        'data_rate': data_rate,
        'level': level,
        'coord': coord
    }
    cache_key = _generate_cache_key(cache_params)
    
    if _is_cache_valid(cache_key):
        try:
            df = _load_from_cache(cache_key)
            print(f"[Cache] Loaded FGM data from cache ({len(df)} rows)")
            return df
        except Exception:
            # If cache loading fails, continue to download
            pass
    
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

    
    # Download and process CDF files in parallel
    all_times = []
    all_values = []
    
    # Use ThreadPoolExecutor for parallel downloads (max 4 workers to avoid overwhelming CDAWeb)
    with ThreadPoolExecutor(max_workers=min(4, len(file_descriptions))) as executor:
        # Submit all download tasks
        future_to_file = {
            executor.submit(
                _download_and_process_cdf,
                file_desc.get('Name'),
                start_time,
                end_time,
                probe,
                data_rate,
                level,
                coord,
                var_name
            ): file_desc
            for file_desc in file_descriptions
            if file_desc.get('Name')
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            times_filtered, values_filtered = future.result()
            if times_filtered is not None and values_filtered is not None:
                all_times.append(times_filtered)
                all_values.append(values_filtered)
    
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
    
    # Save to server-side cache for future requests
    try:
        _save_to_cache(cache_key, df, cache_params)
        print(f"[Cache] Saved FGM data to cache ({len(df)} rows)")
    except Exception as e:
        warnings.warn(f"Failed to save to cache: {e}")
    
    return df


def load_fgm_cdasws_progressive(
    trange: List[str],
    probe: str = '1',
    data_rate: str = 'srvy',
    level: str = 'l2',
    coord: str = 'gse',
    progress_callback=None,
    update_interval: int = 1
):
    """
    Download MMS FGM data with progressive loading.
    
    Yields partial DataFrames as files are downloaded, allowing the UI
    to display data incrementally instead of waiting for all downloads.
    
    Args:
        trange: Time range as ['YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS']
        probe: MMS spacecraft number ('1', '2', '3', or '4')
        data_rate: Data rate ('brst', 'fast', 'slow', 'srvy')
        level: Data level ('l2', 'l1b', 'ql')
        coord: Coordinate system ('gse', 'gsm', 'dmpa', 'bcs')
        progress_callback: Optional callback function(files_completed, total_files)
        update_interval: Yield update every N files (default 1)
    
    Yields:
        Tuple of (DataFrame, files_completed, total_files) showing current progress
        Final yield is (complete_df, total_files, total_files)
    """
    try:
        from cdasws import CdasWs
        import cdflib
    except ImportError as e:
        raise ImportError(f"Required modules not installed: {e}")
    
    # Initialize CDAWeb client
    cdas = CdasWs()
    
    # Construct dataset ID
    dataset = f"MMS{probe}_FGM_{data_rate.upper()}_{level.upper()}"
    var_name = f"mms{probe}_fgm_b_{coord}_{data_rate}_{level}"
    
    # Parse time range
    start_time = datetime.strptime(trange[0], '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(trange[1], '%Y-%m-%d %H:%M:%S')
    
    # Get file list
    try:
        status_code, result = cdas.get_data_file(dataset, [var_name], start_time, end_time)
    except Exception as e:
        raise ValueError(f"Failed to get file list from CDAWeb: {e}")
    
    if not result or 'FileDescription' not in result:
        raise ValueError(f"No data files found for {dataset}")
    
    file_descriptions = result.get('FileDescription', [])
    if not file_descriptions:
        raise ValueError(f"No data files available for {dataset}")
    
    total_files = len([f for f in file_descriptions if f.get('Name')])
    all_times = []
    all_values = []
    files_completed = 0
    
    # Download and process files in parallel with progressive updates
    with ThreadPoolExecutor(max_workers=min(4, total_files)) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                _download_and_process_cdf,
                file_desc.get('Name'),
                start_time,
                end_time,
                probe,
                data_rate,
                level,
                coord,
                var_name
            ): file_desc
            for file_desc in file_descriptions
            if file_desc.get('Name')
        }
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            times_filtered, values_filtered = future.result()
            files_completed += 1
            
            if times_filtered is not None and values_filtered is not None:
                all_times.append(times_filtered)
                all_values.append(values_filtered)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(files_completed, total_files)
            
            # Yield intermediate results every N files or on last file
            if files_completed % update_interval == 0 or files_completed == total_files:
                if all_times:
                    # Create partial DataFrame
                    times = np.concatenate(all_times)
                    values = np.concatenate(all_values)
                    sort_idx = np.argsort(times)
                    times = times[sort_idx]
                    values = values[sort_idx]
                    
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
                    
                    if 'Bt' not in df.columns and 'Bx' in df.columns:
                        df['Bt'] = np.sqrt(df['Bx']**2 + df['By']**2 + df['Bz']**2)
                    
                    df = df.replace(-1e31, np.nan)
                    
                    yield df, files_completed, total_files
    
    if not all_times:
        raise ValueError(
            f"No FGM data found for MMS{probe} ({data_rate}/{level}) "
            f"in range {trange[0]} to {trange[1]}"
        )


def download_cdf_pyspedas(download_info: Dict[str, Any]) -> Tuple[str, str]:
    """
    Download MMS CDF files using PySPEDAS and return a local file path + filename.

    If multiple CDFs are downloaded for the requested trange, they are zipped.
    """
    try:
        import pyspedas
    except ImportError as e:
        raise ImportError("pyspedas is required for CDF export. Please install it.") from e

    trange = download_info.get("trange", None)
    if not trange or len(trange) != 2:
        raise ValueError("Missing time range for CDF export.")

    instrument = (download_info.get("instrument", "") or "").lower()
    probe = download_info.get("probe", "1")
    data_rate = download_info.get("data_rate", "srvy") or "srvy"
    level = download_info.get("level", "l2") or "l2"
    datatype = download_info.get("datatype", "")
    coord = download_info.get("coord", "")

    func_map = {
        "fgm": pyspedas.projects.mms.fgm,
        "scm": pyspedas.projects.mms.scm,
        "fsm": pyspedas.projects.mms.fsm,
        "edp": pyspedas.projects.mms.edp,
        "edi": pyspedas.projects.mms.edi,
        "feeps": pyspedas.projects.mms.feeps,
        "eis": pyspedas.projects.mms.eis,
        "fpi": pyspedas.projects.mms.fpi,
        "hpca": pyspedas.projects.mms.hpca,
        "mec": pyspedas.projects.mms.mec,
        "state": pyspedas.projects.mms.state,
    }

    if instrument not in func_map:
        raise ValueError(f"PySPEDAS export not supported for instrument: {instrument}")

    temp_dir = tempfile.mkdtemp(prefix="pyspedas_mms_")
    os.environ["SPEDAS_DATA_DIR"] = temp_dir
    os.environ["MMS_DATA_DIR"] = temp_dir
    # Ensure PySPEDAS writes to our temp dir
    try:
        pyspedas.config.CONFIG["local_data_dir"] = temp_dir
    except Exception:
        pass

    def _format_trange(tr):
        # Convert "YYYY-MM-DD HH:MM:SS" to "YYYY-MM-DD/HH:MM:SS"
        out = []
        for t in tr:
            if " " in t and "/" not in t:
                out.append(t.replace(" ", "/"))
            else:
                out.append(t)
        return out

    trange_fmt = _format_trange(trange)

    load_func = func_map[instrument]

    # Attempt download-only mode; fall back if signature doesn't accept it
    kwargs = dict(
        trange=trange_fmt,
        probe=probe,
        data_rate=data_rate,
        level=level,
        time_clip=True,
        spdf=True
    )
    if datatype:
        kwargs["datatype"] = datatype
    if coord and instrument not in {"fgm", "fsm"}:
        kwargs["coord"] = coord.lower()

    file_list = []
    try:
        file_list = load_func(**kwargs, downloadonly=True, notplot=True) or []
    except TypeError:
        # Fallback if downloadonly isn't supported
        try:
            file_list = load_func(**kwargs, notplot=True) or []
        except TypeError:
            file_list = load_func(**kwargs) or []

    cdf_files = []
    # If the loader returns file paths directly, prefer those
    for f in file_list:
        if isinstance(f, str) and f.lower().endswith(".cdf") and os.path.exists(f):
            cdf_files.append(f)

    # Fallback: search in temp_dir and PySPEDAS local data dir
    search_dirs = [temp_dir]
    try:
        base_dir = pyspedas.config.CONFIG.get("local_data_dir", "")
        if base_dir and base_dir not in search_dirs:
            search_dirs.append(base_dir)
    except Exception:
        pass

    for base in search_dirs:
        for root, _, files in os.walk(base):
            for fname in files:
                if fname.lower().endswith(".cdf"):
                    cdf_files.append(os.path.join(root, fname))
    for root, _, files in os.walk(temp_dir):
        for fname in files:
            if fname.lower().endswith(".cdf"):
                cdf_files.append(os.path.join(root, fname))

    if not cdf_files:
        raise FileNotFoundError("No CDF files downloaded by PySPEDAS.")

    cdf_files.sort()

    if len(cdf_files) == 1:
        return cdf_files[0], os.path.basename(cdf_files[0])

    zip_path = os.path.join(temp_dir, "mms_cdf_export.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in cdf_files:
            zf.write(f, arcname=os.path.basename(f))

    return zip_path, os.path.basename(zip_path)


def _download_fpi_species(
    species: str,
    label: str,
    probe: str,
    data_rate: str,
    level: str,
    coord: str,
    start_time: datetime,
    end_time: datetime
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Download and process FPI data for a single species (DIS or DES).
    
    Returns:
        Tuple of (label, DataFrame) or (label, None) on error
    """
    try:
        from cdasws import CdasWs
        import cdflib
    except ImportError as e:
        warnings.warn(f"Required modules not installed: {e}")
        return label, None
    
    cdas = CdasWs()
    
    # Dataset: MMS1_FPI_FAST_L2_DIS-MOMS or MMS1_FPI_FAST_L2_DES-MOMS
    dataset = f"MMS{probe}_FPI_{data_rate.upper()}_{level.upper()}_{species.upper()}-MOMS"
    
    # Variable: mms1_dis_bulkv_gse_fast or mms1_des_bulkv_gse_fast
    var_name = f"mms{probe}_{species}_bulkv_{coord}_{data_rate}"
    
    try:
        status_code, result = cdas.get_data_file(
            dataset,
            [var_name],
            start_time,
            end_time
        )
    except Exception as e:
        warnings.warn(f"Failed to get {label} data: {e}")
        return label, None
    
    if not result or 'FileDescription' not in result:
        warnings.warn(f"No {label} files found for {dataset}")
        return label, None
    
    file_descriptions = result.get('FileDescription', [])
    if not file_descriptions:
        return label, None
    
    all_times = []
    all_values = []
    
    # Create session with retry strategy and connection pooling
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=5, pool_maxsize=10))
    
    for file_desc in file_descriptions:
        file_url = file_desc.get('Name')
        if not file_url:
            continue
        
        with tempfile.NamedTemporaryFile(suffix='.cdf', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Download with streaming for memory efficiency
            response = session.get(file_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(tmp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            cdf = cdflib.CDF(tmp_path)
            
            # Find epoch variable
            epoch_data = None
            epoch_patterns = [
                f'Epoch',
                f'mms{probe}_{species}_epoch_{data_rate}',
                'epoch'
            ]
            
            for pattern in epoch_patterns:
                try:
                    epoch_data = cdf.varget(pattern)
                    break
                except:
                    continue
            
            if epoch_data is None:
                info = cdf.cdf_info()
                for zvar in getattr(info, 'zVariables', []):
                    if 'epoch' in zvar.lower():
                        epoch_data = cdf.varget(zvar)
                        break
            
            if epoch_data is None:
                continue
            
            # Get velocity data
            try:
                vel_data = cdf.varget(var_name)
            except:
                continue
            
            # Convert epoch to datetime - use encode for safe TT2000 conversion
            times = cdflib.cdfepoch.encode(epoch_data)
            times_np = np.array(times, dtype='datetime64[ns]')
            start_np = np.datetime64(start_time)
            end_np = np.datetime64(end_time)
            
            mask = (times_np >= start_np) & (times_np <= end_np)
            times_filtered = times_np[mask]
            values_filtered = vel_data[mask]
            
            if len(times_filtered) > 0:
                all_times.append(times_filtered)
                all_values.append(values_filtered)
            
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    session.close()
    
    if not all_times:
        warnings.warn(f"No {label} data found in time range")
        return label, None
    
    # Concatenate and sort
    times = np.concatenate(all_times)
    values = np.concatenate(all_values)
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    values = values[sort_idx]
    
    # Create DataFrame
    datetime_index = pd.to_datetime(times)
    
    if len(values.shape) == 1:
        columns = ['Vt']
        values = values.reshape(-1, 1)
    elif values.shape[1] == 3:
        columns = ['Vx', 'Vy', 'Vz']
    else:
        columns = [f'V{i}' for i in range(values.shape[1])]
    
    df = pd.DataFrame(values, index=datetime_index, columns=columns)
    df.index.name = 'time'
    
    # Add magnitude if not present
    if 'Vt' not in df.columns and 'Vx' in df.columns:
        df['Vt'] = np.sqrt(df['Vx']**2 + df['Vy']**2 + df['Vz']**2)
    
    # Clean fill values
    df = df.replace(-1e31, np.nan)
    
    return label, df


@st.cache_data(show_spinner="Loading FPI data...")
def load_fpi_cdasws(
    trange: List[str],
    probe: str = '1',
    data_rate: str = 'fast',
    level: str = 'l2',
    coord: str = 'gse'
) -> dict:

    """
    Download MMS FPI (Fast Plasma Investigation) data using CDAWeb API.
    
    Downloads both DIS (ion) and DES (electron) bulk velocity moments in parallel
    for improved performance. Uses server-side caching.
    
    Args:
        trange: Time range as ['YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS']
        probe: MMS spacecraft number ('1', '2', '3', or '4')
        data_rate: Data rate ('fast' or 'brst')
        level: Data level ('l2')
        coord: Coordinate system ('gse', 'gsm', 'dbcs')
    
    Returns:
        Dictionary with keys 'DIS' (ions) and 'DES' (electrons), each containing
        a DataFrame with DatetimeIndex and columns ['Vx', 'Vy', 'Vz', 'Vt']
    """
    # Check cache first
    cache_params = {
        'instrument': 'fpi',
        'trange': trange,
        'probe': probe,
        'data_rate': data_rate,
        'level': level,
        'coord': coord
    }
    cache_key = _generate_cache_key(cache_params)
    
    if _is_cache_valid(cache_key):
        try:
            results = _load_from_cache(cache_key)
            total_rows = sum(len(df) for df in results.values())
            print(f"[Cache] Loaded FPI data from cache ({total_rows} total rows)")
            return results
        except Exception:
            # If cache loading fails, continue to download
            pass
    
    start_time = datetime.strptime(trange[0], '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(trange[1], '%Y-%m-%d %H:%M:%S')
    
    results = {}
    
    # Download both DIS (ions) and DES (electrons) in parallel
    species_list = [('dis', 'DIS'), ('des', 'DES')]
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit tasks for both species
        future_to_species = {
            executor.submit(
                _download_fpi_species,
                species,
                label,
                probe,
                data_rate,
                level,
                coord,
                start_time,
                end_time
            ): (species, label)
            for species, label in species_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_species):
            label, df = future.result()
            if df is not None:
                results[label] = df
    
    if not results:
        raise ValueError(
            f"No FPI data found for MMS{probe} ({data_rate}/{level}) "
            f"in range {trange[0]} to {trange[1]}"
        )
    
    # Save to server-side cache for future requests
    try:
        _save_to_cache(cache_key, results, cache_params)
        total_rows = sum(len(df) for df in results.values())
        print(f"[Cache] Saved FPI data to cache ({total_rows} total rows)")
    except Exception as e:
        warnings.warn(f"Failed to save to cache: {e}")
    
    return results


def load_fpi_cdasws_progressive(
    trange: List[str],
    probe: str = '1',
    data_rate: str = 'fast',
    level: str = 'l2',
    coord: str = 'gse',
    progress_callback=None
):
    """
    Download MMS FPI data with progressive loading.
    
    Yields partial results as DIS and DES data are downloaded.
    
    Args:
        trange: Time range as ['YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS']
        probe: MMS spacecraft number ('1', '2', '3', or '4')
        data_rate: Data rate ('fast' or 'brst')
        level: Data level ('l2')
        coord: Coordinate system ('gse', 'gsm', 'dbcs')
        progress_callback: Optional callback function(species, completed, total)
    
    Yields:
        Tuple of (results_dict, completed_species, total_species)
    """
    start_time = datetime.strptime(trange[0], '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(trange[1], '%Y-%m-%d %H:%M:%S')
    
    results = {}
    species_list = [('dis', 'DIS'), ('des', 'DES')]
    total_species = len(species_list)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_species = {
            executor.submit(
                _download_fpi_species,
                species,
                label,
                probe,
                data_rate,
                level,
                coord,
                start_time,
                end_time
            ): (species, label)
            for species, label in species_list
        }
        
        for future in as_completed(future_to_species):
            label, df = future.result()
            completed += 1
            
            if df is not None:
                results[label] = df
            
            if progress_callback:
                progress_callback(label, completed, total_species)
            
            yield results.copy(), completed, total_species
    
    if not results:
        raise ValueError(
            f"No FPI data found for MMS{probe} ({data_rate}/{level}) "
            f"in range {trange[0]} to {trange[1]}"
        )


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


# ============================================================================
# Universal MMS Instrument Loader
# ============================================================================

# Dataset ID patterns for each instrument
# Format strings use: p=probe, r=rate, l=level, t=datatype, c=coord
# Based on PySPEDAS documentation: https://pyspedas.readthedocs.io/en/latest/mms.html
INSTRUMENT_DATASET_MAP = {
    'fgm': {
        # Variable: mms1_fgm_b_gse_srvy_l2
        'dataset': 'MMS{p}_FGM_{r}_{l}',
        'var_patterns': ['mms{p}_fgm_b_{c}_{r}_{l}'],
        'columns': ['Bx', 'By', 'Bz', 'Bt'],
        'units': 'nT',
        'type': 'vector'
    },
    'scm': {
        # Variable: mms1_scm_acb_gse_scsrvy_srvy_l2
        # SCM datatype (scsrvy, scb, etc.) goes between gse and rate
        'dataset': 'MMS{p}_SCM_{r}_{l}_{t}',
        'var_patterns': ['mms{p}_scm_acb_gse_{t}_{r}_{l}'],
        'columns': ['Bx', 'By', 'Bz'],
        'units': 'nT',
        'type': 'vector'
    },
    'fsm': {
        # Variable: mms1_fsm_b_gse_brst_l3  (FSM is only burst, L3, 8khz)
        'dataset': 'MMS{p}_FSM_{r}_{l}_{t}',
        'var_patterns': ['mms{p}_fsm_b_gse_{r}_{l}'],
        'columns': ['Bx', 'By', 'Bz', 'Bt'],
        'units': 'nT',
        'type': 'vector'
    },
    'edp': {
        # Variable: mms1_edp_dce_gse_fast_l2
        'dataset': 'MMS{p}_EDP_{r}_{l}_{t}',
        'var_patterns': ['mms{p}_edp_{t}_gse_{r}_{l}'],
        'columns': ['Ex', 'Ey', 'Ez'],
        'units': 'mV/m',
        'type': 'vector'
    },
    'edi': {
        # Variable: mms1_edi_e_gse_srvy_l2 (for efield datatype)
        'dataset': 'MMS{p}_EDI_{r}_{l}_{t}',
        'var_patterns': ['mms{p}_edi_e_gse_{r}_{l}', 'mms{p}_edi_vdrift_gse_{r}_{l}'],
        'columns': ['Ex', 'Ey', 'Ez'],
        'units': 'mV/m',
        'type': 'vector'
    },
    'hpca': {
        # Variable: mms1_hpca_hplus_number_density (moments datatype)
        'dataset': 'MMS{p}_HPCA_{r}_{l}_{t}',
        'var_patterns': ['mms{p}_hpca_hplus_number_density', 
                         'mms{p}_hpca_hplus_scalar_temperature',
                         'mms{p}_hpca_hplus_ion_bulk_velocity'],
        'columns': ['value'],
        'units': 'cm^-3',
        'type': 'scalar'
    },
    'feeps': {
        # Variable: mms1_epd_feeps_srvy_l2_electron_intensity_omni
        'dataset': 'MMS{p}_FEEPS_{r}_{l}_{t}',
        'var_patterns': ['mms{p}_epd_feeps_{r}_{l}_{t}_intensity_omni'],
        'columns': ['value'],
        'units': '1/(cm^2 s sr keV)',
        'type': 'flux'
    },
    'eis': {
        # Variable: mms1_epd_eis_srvy_l2_extof_proton_flux_omni
        'dataset': 'MMS{p}_EPDE_{r}_{l}_{t}',
        'var_patterns': ['mms{p}_epd_eis_{r}_{l}_{t}_proton_flux_omni'],
        'columns': ['value'],
        'units': '1/(cm^2 s sr keV)',
        'type': 'flux'
    },
    'aspoc': {
        # Variable: mms1_aspoc_ionc_l2
        'dataset': 'MMS{p}_ASPOC_{r}_{l}',
        'var_patterns': ['mms{p}_aspoc_ionc_{l}', 'mms{p}_asp1_ionc_{l}', 'mms{p}_asp2_ionc_{l}'],
        'columns': ['current'],
        'units': 'μA',
        'type': 'scalar'
    },
    'mec': {
        # Variable: mms1_mec_r_gsm (MEC has r and v for position and velocity)
        'dataset': 'MMS{p}_MEC_{r}_{l}_{t}',
        'var_patterns': ['mms{p}_mec_r_{c}'],
        'columns': ['X', 'Y', 'Z'],
        'units': 'km',
        'type': 'vector'
    },
    'state': {
        # STATE uses ASCII files, not CDF - special handling needed
        # Variable: mms1_defeph_pos (from ASCII)
        'dataset': 'MMS{p}_DEFATT',
        'var_patterns': ['mms{p}_defeph_pos', 'mms{p}_defeph_vel'],
        'columns': ['X', 'Y', 'Z'],
        'units': 'km',
        'type': 'vector',
        'note': 'STATE uses ASCII files - limited CDAWeb support'
    },
    'tqf': {
        # Tetrahedron Quality Factor - no probe number
        'dataset': 'MMS_TETRAHEDRON_QF',
        'var_patterns': ['mms_tetrahedron_qf'],
        'columns': ['QF'],
        'units': '',
        'type': 'scalar'
    },
}


def _process_single_cdf_universal(
    file_url: str,
    var_names: List[str],
    var_patterns: List[str],
    start_time: datetime,
    end_time: datetime
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Download and process a single CDF file for universal loader.
    
    Returns:
        Tuple of (times_filtered, values_filtered) or (None, None) on error
    """
    import cdflib
    
    # Create session with retry strategy and connection pooling
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=5, pool_maxsize=10))
    
    with tempfile.NamedTemporaryFile(suffix='.cdf', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Download with streaming for memory efficiency
        response = session.get(file_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(tmp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        cdf = cdflib.CDF(tmp_path)
        
        # Find epoch variable
        epoch_data = None
        info = cdf.cdf_info()
        for zvar in getattr(info, 'zVariables', []):
            if 'epoch' in zvar.lower():
                try:
                    epoch_data = cdf.varget(zvar)
                    break
                except:
                    continue
        
        if epoch_data is None:
            return None, None
        
        # Try each var pattern
        field_data = None
        for var_name in var_names:
            try:
                field_data = cdf.varget(var_name)
                break
            except:
                continue
        
        # If still nothing, try to find a matching variable
        if field_data is None:
            for zvar in getattr(info, 'zVariables', []):
                for pattern in var_patterns:
                    # Simple pattern matching
                    base = pattern.split('{')[0]
                    if base and base in zvar.lower():
                        try:
                            test_data = cdf.varget(zvar)
                            if test_data is not None and len(test_data) > 0:
                                field_data = test_data
                                break
                        except:
                            continue
                if field_data is not None:
                    break
        
        if field_data is None:
            return None, None
        
        # Convert epoch to datetime - use encode for safe TT2000 conversion
        times = cdflib.cdfepoch.encode(epoch_data)
        times_np = np.array(times, dtype='datetime64[ns]')
        
        # Filter to time range
        start_np = np.datetime64(start_time)
        end_np = np.datetime64(end_time)
        mask = (times_np >= start_np) & (times_np <= end_np)
        
        times_filtered = times_np[mask]
        
        # Handle multi-dimensional data (spectra -> mean over energy)
        if len(field_data.shape) > 2:
            # Average over energy dimension for spectra
            field_data = np.nanmean(field_data, axis=tuple(range(1, len(field_data.shape)-1)))
        
        values_filtered = field_data[mask]
        
        if len(times_filtered) > 0:
            return times_filtered, values_filtered
        return None, None
        
    except Exception as e:
        warnings.warn(f"Failed to process file {file_url}: {e}")
        return None, None
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
        session.close()


def _download_cdf_and_extract(
    dataset: str,
    var_patterns: List[str],
    start_time: datetime,
    end_time: datetime,
    columns: List[str],
    probe: str = '1',
    data_rate: str = 'srvy',
    level: str = 'l2',
    coord: str = 'gse',
    datatype: str = ''
) -> pd.DataFrame:
    """
    Generic CDF download and extraction helper.
    
    Downloads CDF files from CDAWeb and extracts the specified variable.
    Uses parallel downloads for improved performance when multiple files are available.
    """
    try:
        from cdasws import CdasWs
        import cdflib
    except ImportError as e:
        raise ImportError(f"Required modules not installed: {e}")
    
    cdas = CdasWs()
    
    # Build variable names from patterns
    var_names = []
    for pattern in var_patterns:
        var_name = pattern.format(
            p=probe, r=data_rate.lower(), l=level.lower(), 
            c=coord.lower(), t=datatype.lower()
        )
        var_names.append(var_name)
    
    # Try to get files
    try:
        status_code, result = cdas.get_data_file(
            dataset,
            var_names[:1],  # Request first pattern
            start_time,
            end_time
        )
    except Exception as e:
        raise ValueError(f"Failed to get file list from CDAWeb for {dataset}: {e}")
    
    if not result or 'FileDescription' not in result:
        raise ValueError(f"No data files found for {dataset}")
    
    file_descriptions = result.get('FileDescription', [])
    if not file_descriptions:
        raise ValueError(f"No data files available for {dataset}")
    
    all_times = []
    all_values = []
    
    # Download and process CDF files in parallel
    with ThreadPoolExecutor(max_workers=min(4, len(file_descriptions))) as executor:
        # Submit all download tasks
        future_to_file = {
            executor.submit(
                _process_single_cdf_universal,
                file_desc.get('Name'),
                var_names,
                var_patterns,
                start_time,
                end_time
            ): file_desc
            for file_desc in file_descriptions
            if file_desc.get('Name')
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            times_filtered, values_filtered = future.result()
            if times_filtered is not None and values_filtered is not None:
                all_times.append(times_filtered)
                all_values.append(values_filtered)
    
    if not all_times:
        raise ValueError(f"No data extracted from {dataset}")
    
    # Concatenate and sort
    times = np.concatenate(all_times)
    values = np.concatenate(all_values)
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    values = values[sort_idx]
    
    # Create DataFrame
    datetime_index = pd.to_datetime(times)
    
    # Handle shape
    if len(values.shape) == 1:
        values = values.reshape(-1, 1)
    
    # Use provided columns or generate
    if values.shape[1] >= len(columns):
        cols = columns[:values.shape[1]]
    else:
        cols = columns + [f'col{i}' for i in range(len(columns), values.shape[1])]
    
    df = pd.DataFrame(values[:, :len(cols)], index=datetime_index, columns=cols)
    df.index.name = 'time'
    
    # Clean fill values
    df = df.replace(-1e31, np.nan)
    df = df.replace(1e31, np.nan)
    
    # Add magnitude for vector data
    if len(cols) >= 3 and cols[0] in ['Bx', 'Ex', 'X', 'Vx']:
        mag_col = 'Bt' if 'Bx' in cols else 'Et' if 'Ex' in cols else 'R' if 'X' in cols else 'Vt'
        if mag_col not in df.columns:
            df[mag_col] = np.sqrt(df.iloc[:, 0]**2 + df.iloc[:, 1]**2 + df.iloc[:, 2]**2)
    
    return df


@st.cache_data(show_spinner="Loading instrument data...")
def load_mms_universal(
    instrument: str,
    trange: List[str],
    probe: str = '1',
    data_rate: str = 'srvy',
    level: str = 'l2',
    coord: str = 'gse',
    datatype: str = ''
) -> dict:
    """
    Universal MMS instrument data loader.
    
    Downloads data from NASA CDAWeb for any MMS instrument and returns
    standardized Pandas DataFrames. Uses server-side caching for improved
    performance on repeated requests.
    
    Args:
        instrument: Instrument key ('fgm', 'fpi', 'scm', 'edp', etc.)
        trange: Time range as ['YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS']
        probe: MMS spacecraft number ('1', '2', '3', or '4')
        data_rate: Data rate (instrument-specific)
        level: Data level ('l2', 'l1b', etc.)
        coord: Coordinate system ('gse', 'gsm')
        datatype: Instrument-specific datatype
        
    Returns:
        Dictionary with instrument data as DataFrames
    """
    instrument = instrument.lower()
    
    # Use existing optimized loaders for FGM and FPI (they have their own caching)
    if instrument == 'fgm':
        df = load_fgm_cdasws(trange, probe, data_rate, level, coord)
        return {'FGM': df}
    
    if instrument == 'fpi':
        return load_fpi_cdasws(trange, probe, data_rate, level, coord)
    
    # For other instruments, use server-side caching
    cache_params = {
        'instrument': instrument,
        'trange': trange,
        'probe': probe,
        'data_rate': data_rate,
        'level': level,
        'coord': coord,
        'datatype': datatype
    }
    cache_key = _generate_cache_key(cache_params)
    
    if _is_cache_valid(cache_key):
        try:
            result = _load_from_cache(cache_key)
            print(f"[Cache] Loaded {instrument.upper()} data from cache ({len(result[instrument.upper()])} rows)")
            return result
        except Exception:
            # If cache loading fails, continue to download
            pass
    
    # Get instrument config
    if instrument not in INSTRUMENT_DATASET_MAP:
        raise ValueError(f"Unknown instrument: {instrument}")
    
    config = INSTRUMENT_DATASET_MAP[instrument]
    
    # Build dataset ID
    dataset = config['dataset'].format(
        p=probe, r=data_rate.upper(), l=level.upper(), t=datatype.upper()
    )
    
    # Parse time range
    start_time = datetime.strptime(trange[0], '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(trange[1], '%Y-%m-%d %H:%M:%S')
    
    try:
        df = _download_cdf_and_extract(
            dataset=dataset,
            var_patterns=config['var_patterns'],
            start_time=start_time,
            end_time=end_time,
            columns=config['columns'],
            probe=probe,
            data_rate=data_rate,
            level=level,
            coord=coord,
            datatype=datatype
        )
        
        # Store metadata
        df.attrs['instrument'] = instrument.upper()
        df.attrs['units'] = config['units']
        df.attrs['type'] = config['type']
        
        result = {instrument.upper(): df}
        
        # Save to server-side cache for future requests
        try:
            _save_to_cache(cache_key, result, cache_params)
            print(f"[Cache] Saved {instrument.upper()} data to cache ({len(df)} rows)")
        except Exception as e:
            warnings.warn(f"Failed to save to cache: {e}")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to load {instrument.upper()} data: {e}")
