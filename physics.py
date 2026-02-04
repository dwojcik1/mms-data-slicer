"""
physics.py - Turbulence Analysis Engine
========================================
Pure mathematical functions for spectral analysis, statistics, and turbulence.
"""

import numpy as np
from scipy import signal
from scipy import stats
from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclass
import streamlit as st


@dataclass
class PSDResult:
    """Container for Power Spectral Density results."""
    frequencies: np.ndarray
    power: np.ndarray
    sampling_frequency: float
    nperseg: int
    method: str = 'welch'


@dataclass
class PDFResult:
    """Container for Probability Density Function results."""
    bin_centers: np.ndarray
    density: np.ndarray
    bin_edges: np.ndarray
    n_samples: int
    n_bins: int


@dataclass
class StatisticsResult:
    """Container for statistical moments and descriptors."""
    mean: float
    median: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    min_val: float
    max_val: float
    n_samples: int
    n_nan: int


def clean_data(data: np.ndarray, fill_value: float = np.nan) -> Tuple[np.ndarray, int]:
    """
    Clean data by removing NaN and infinite values.
    
    Args:
        data: Input data array
        fill_value: Value to identify as invalid (default: np.nan)
        
    Returns:
        Tuple of (cleaned_data, number_of_removed_points)
    """
    # Create mask for valid data
    valid_mask = np.isfinite(data)
    
    if fill_value != np.nan:
        valid_mask &= (data != fill_value)
    
    n_removed = np.sum(~valid_mask)
    cleaned = data[valid_mask]
    
    return cleaned, n_removed


def _clean_data_and_time(
    data: np.ndarray,
    time_data: np.ndarray,
    fill_value: float = np.nan
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean data and time arrays together to keep alignment.
    """
    if data is None or time_data is None:
        raise ValueError("Data and time arrays are required for PSD.")
    # Ensure 1D arrays
    data = np.asarray(data).reshape(-1)
    time_data = np.asarray(time_data).reshape(-1)

    # Align lengths if needed
    n = min(len(data), len(time_data))
    data = data[:n]
    time_data = time_data[:n]

    valid_mask = np.isfinite(data)
    if fill_value != np.nan:
        valid_mask &= (data != fill_value)

    # Filter invalid time values
    if np.issubdtype(time_data.dtype, np.datetime64):
        # datetime64 can be NaT
        valid_mask &= ~np.isnat(time_data)
    else:
        valid_mask &= np.isfinite(time_data.astype(float))

    return data[valid_mask], time_data[valid_mask]


def _estimate_typical_dt(time_data: np.ndarray) -> float:
    """
    Estimate typical sampling interval, robust to gaps.
    """
    if len(time_data) < 2:
        raise ValueError("Need at least 2 time points")

    if np.issubdtype(time_data.dtype, np.datetime64):
        dt = np.diff(time_data).astype('timedelta64[ns]').astype(float) / 1e9
    else:
        dt = np.diff(time_data.astype(float))

    positive_dt = dt[dt > 0]
    if len(positive_dt) == 0:
        raise ValueError("Invalid time delta")

    # Use lower-quantile median to avoid large gaps
    q = np.quantile(positive_dt, 0.2)
    dt_core = positive_dt[positive_dt <= q] if np.any(positive_dt <= q) else positive_dt
    return float(np.median(dt_core))


def calculate_sampling_frequency(time_data: np.ndarray) -> float:
    """
    Calculate sampling frequency from time array.
    
    Args:
        time_data: Array of datetime64 values
        
    Returns:
        Sampling frequency in Hz
    """
    typical_dt = _estimate_typical_dt(time_data)
    if typical_dt <= 0:
        raise ValueError("Invalid time delta")
    return 1.0 / typical_dt


def _largest_contiguous_segment(
    time_data: np.ndarray,
    gap_factor: float = 5.0
) -> slice:
    """
    Return slice for the largest contiguous segment without large gaps.
    """
    if len(time_data) < 2:
        return slice(0, len(time_data))

    if np.issubdtype(time_data.dtype, np.datetime64):
        dt = np.diff(time_data).astype('timedelta64[ns]').astype(float) / 1e9
    else:
        dt = np.diff(time_data.astype(float))

    positive_dt = dt[dt > 0]
    if len(positive_dt) == 0:
        return slice(0, len(time_data))

    typical_dt = _estimate_typical_dt(time_data)
    gap_thresh = gap_factor * typical_dt

    # Identify segment boundaries where gaps are large
    gap_idxs = np.where(dt > gap_thresh)[0]
    if len(gap_idxs) == 0:
        return slice(0, len(time_data))

    # Build segments
    starts = np.concatenate(([0], gap_idxs + 1))
    ends = np.concatenate((gap_idxs + 1, [len(time_data)]))

    # Choose longest segment
    lengths = ends - starts
    max_idx = int(np.argmax(lengths))
    return slice(int(starts[max_idx]), int(ends[max_idx]))


@st.cache_data(show_spinner=False)
def compute_psd_welch(
    _data: np.ndarray,
    _time_data: np.ndarray,
    fs_override: Optional[float] = None,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    detrend: str = 'linear',
    scaling: str = 'density'
) -> PSDResult:
    """
    Compute PSD for MMS data, handling gaps by averaging PSDs of valid segments.
    """
    # 1. Clean NaNs (Basic cleanup) and keep time aligned
    data = np.asarray(_data).reshape(-1)
    time_data = np.asarray(_time_data).reshape(-1)
    n = min(len(data), len(time_data))
    data = data[:n]
    time_data = time_data[:n]

    if np.issubdtype(time_data.dtype, np.datetime64):
        time_valid = ~np.isnat(time_data)
    else:
        time_valid = np.isfinite(time_data.astype(float))

    mask = np.isfinite(data) & time_valid
    clean_data = data[mask]
    clean_time = time_data[mask]

    if len(clean_data) < 16:
        raise ValueError("Insufficient data points.")

    # 2. Robust Sampling Frequency (Median Diff)
    # in Space Physics, always use median to ignore gaps/jitter
    if np.issubdtype(clean_time.dtype, np.datetime64):
        dt_array = np.diff(clean_time).astype('timedelta64[ns]').astype(float) / 1e9
    else:
        dt_array = np.diff(clean_time.astype(float))
    dt_array = dt_array[dt_array > 0]
    if len(dt_array) == 0:
        raise ValueError("Invalid time cadence.")
    dt_median = np.median(dt_array)
    fs = fs_override if fs_override and fs_override > 0 else (1.0 / dt_median)
    
    # 3. Gap Handling: Identify continuous segments
    # Define a gap threshold (e.g., 1.5x the sampling rate)
    gap_indices = np.where(dt_array > 1.5 * dt_median)[0]
    
    # Create slices for start/end of valid blocks
    segment_slices = []
    start_idx = 0
    for gap_idx in gap_indices:
        end_idx = gap_idx + 1  # +1 because diff is shorter by 1
        segment_slices.append(slice(start_idx, end_idx))
        start_idx = end_idx
    segment_slices.append(slice(start_idx, len(clean_data)))  # Last segment
    
    # 4. Set nperseg based on the longest segment (or user input)
    # We want a consistent frequency binning for all segments
    if nperseg is None:
        lengths = [s.stop - s.start for s in segment_slices]
        max_len = max(lengths)
        nperseg = min(max_len // 4, 4096)
        nperseg = max(nperseg, 64)  # Floor constraint
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    psd_list = []
    
    # 5. Compute PSD for each valid segment
    for sl in segment_slices:
        segment_data = clean_data[sl]
        
        # Skip segments shorter than nperseg
        if len(segment_data) < nperseg:
            continue
        
        freqs, p_seg = signal.welch(
            segment_data,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling=scaling
        )
        psd_list.append(p_seg)
    
    if not psd_list:
        raise ValueError(f"No contiguous data segments were long enough for nperseg={nperseg}")
    
    # 6. Average the PSDs (weighted by number of windows could be done, but simple mean is standard)
    # Note: signal.welch returns same freq bins if fs and nperseg are constant
    avg_power = np.mean(psd_list, axis=0)
    
    return PSDResult(
        frequencies=freqs,  # Frequencies are consistent across segments
        power=avg_power,
        sampling_frequency=fs,
        nperseg=nperseg,
        method='welch_multi_segment'
    )


def compute_psd_welch_mms(*args, **kwargs) -> PSDResult:
    """
    Alias for MMS gap-aware Welch PSD.
    """
    return compute_psd_welch(*args, **kwargs)


def compute_mms_trace_psd(data_vector: np.ndarray, time: np.ndarray, kind: str = 'scalar', **kwargs) -> PSDResult:
    """
    Wrapper to handle Vector vs Scalar logic for MMS instruments.
    data_vector shape should be (N, 3) for vectors or (N,) for scalars.
    """
    # Handle EDI specifically if many NaNs exist
    if np.isnan(data_vector).mean() > 0.2:
        print("Warning: High gap ratio (EDI?), consider Lomb-Scargle.")

    # Case: Scalar (Density, V_sc, Magnitude)
    if kind == 'scalar' or data_vector.ndim == 1:
        return compute_psd_welch_mms(data_vector, time, **kwargs)

    # Case: Vector (FGM, EDP, FPI Velocity) -> Trace PSD
    if kind == 'vector' and data_vector.ndim == 2 and data_vector.shape[1] == 3:
        psd_x = compute_psd_welch_mms(data_vector[:, 0], time, **kwargs)
        psd_y = compute_psd_welch_mms(data_vector[:, 1], time, **kwargs)
        psd_z = compute_psd_welch_mms(data_vector[:, 2], time, **kwargs)

        total_power = psd_x.power + psd_y.power + psd_z.power
        return PSDResult(
            frequencies=psd_x.frequencies,
            power=total_power,
            sampling_frequency=psd_x.sampling_frequency,
            nperseg=psd_x.nperseg,
            method='welch_trace_sum'
        )

    raise ValueError("Data must be 1D array or (N,3) vector.")


@st.cache_data(show_spinner=False)
def compute_pdf(
    _data: np.ndarray,
    n_bins: int = 50,
    range_sigma: float = 5.0,
    density: bool = True
) -> PDFResult:
    """
    Compute Probability Density Function using histogram.
    
    Args:
        data: 1D data array
        n_bins: Number of histogram bins
        range_sigma: Range in standard deviations from mean (for auto range)
        density: If True, normalize to probability density
        
    Returns:
        PDFResult with bin centers and density values
    """
    # Clean data
    clean, n_removed = clean_data(_data)
    
    if len(clean) < 10:
        raise ValueError(f"Insufficient data points: {len(clean)}")
    
    # Calculate range based on data statistics
    mean = np.mean(clean)
    std = np.std(clean)
    
    # Set range to ±range_sigma standard deviations
    data_min = np.min(clean)
    data_max = np.max(clean)
    
    # Use actual data range if it's narrower than sigma range
    bin_min = max(data_min, mean - range_sigma * std)
    bin_max = min(data_max, mean + range_sigma * std)
    
    # Compute histogram
    counts, bin_edges = np.histogram(
        clean,
        bins=n_bins,
        range=(bin_min, bin_max),
        density=density
    )
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return PDFResult(
        bin_centers=bin_centers,
        density=counts,
        bin_edges=bin_edges,
        n_samples=len(clean),
        n_bins=n_bins
    )


@st.cache_data(show_spinner=False)
def compute_statistics(_data: np.ndarray) -> StatisticsResult:
    """
    Compute statistical moments and descriptors.
    
    Args:
        data: 1D data array
        
    Returns:
        StatisticsResult with mean, median, std, variance, skewness, kurtosis
    """
    # Count NaN before cleaning
    n_nan = np.sum(~np.isfinite(_data))
    
    # Clean data
    clean, _ = clean_data(_data)
    
    if len(clean) < 4:
        raise ValueError(f"Insufficient data points: {len(clean)}")
    
    return StatisticsResult(
        mean=float(np.mean(clean)),
        median=float(np.median(clean)),
        std=float(np.std(clean, ddof=1)),
        variance=float(np.var(clean, ddof=1)),
        skewness=float(stats.skew(clean)),
        kurtosis=float(stats.kurtosis(clean)),
        min_val=float(np.min(clean)),
        max_val=float(np.max(clean)),
        n_samples=len(clean),
        n_nan=int(n_nan)
    )


def compute_structure_function(
    data: np.ndarray,
    time_data: np.ndarray,
    orders: list = [1, 2, 3, 4],
    max_lag: Optional[int] = None
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute structure functions S_q(tau) = <|B(t+tau) - B(t)|^q>.
    
    Important for turbulence intermittency analysis.
    
    Args:
        data: 1D data array
        time_data: Time array
        orders: List of structure function orders to compute
        max_lag: Maximum lag in samples (default: len(data)//10)
        
    Returns:
        Dictionary mapping order q to (tau_array, S_q_array)
    """
    clean, _ = clean_data(data)
    
    if max_lag is None:
        max_lag = min(len(clean) // 10, 1000)
    
    fs = calculate_sampling_frequency(time_data)
    
    results = {}
    
    for q in orders:
        taus = []
        s_q = []
        
        for lag in range(1, max_lag + 1):
            increments = clean[lag:] - clean[:-lag]
            s_q_value = np.mean(np.abs(increments) ** q)
            taus.append(lag / fs)  # Convert to time
            s_q.append(s_q_value)
        
        results[q] = (np.array(taus), np.array(s_q))
    
    return results


def compute_flatness(data: np.ndarray, scales: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scale-dependent flatness (kurtosis of increments).
    
    F(tau) = <delta_tau(B)^4> / <delta_tau(B)^2>^2
    
    For Gaussian: F = 3
    For intermittent turbulence: F > 3
    
    Args:
        data: 1D data array
        scales: Array of lag scales (samples)
        
    Returns:
        Tuple of (scales, flatness_values)
    """
    clean, _ = clean_data(data)
    
    if scales is None:
        max_scale = min(len(clean) // 10, 500)
        scales = np.unique(np.logspace(0, np.log10(max_scale), 30).astype(int))
    
    flatness = []
    
    for scale in scales:
        if scale >= len(clean):
            break
        
        increments = clean[scale:] - clean[:-scale]
        
        m2 = np.mean(increments ** 2)
        m4 = np.mean(increments ** 4)
        
        if m2 > 0:
            f = m4 / (m2 ** 2)
        else:
            f = np.nan
        
        flatness.append(f)
    
    return scales[:len(flatness)], np.array(flatness)


@st.cache_data(show_spinner=False)
def fit_power_law(
    _x: np.ndarray,
    _y: np.ndarray,
    x_range: Optional[Tuple[float, float]] = None
) -> Tuple[float, float, float]:
    """
    Fit a power law y = A * x^alpha to data.
    
    Args:
        x: Independent variable (e.g., frequency)
        y: Dependent variable (e.g., PSD)
        x_range: Optional (min, max) range for fitting
        
    Returns:
        Tuple of (alpha, A, r_squared) where y = A * x^alpha
    """
    # Filter to positive values
    mask = (_x > 0) & (_y > 0) & np.isfinite(_x) & np.isfinite(_y)
    
    if x_range is not None:
        mask &= (_x >= x_range[0]) & (_x <= x_range[1])
    
    x_fit = _x[mask]
    y_fit = _y[mask]
    
    if len(x_fit) < 3:
        raise ValueError("Insufficient data points for power law fit")
    
    # Linear regression in log-log space
    log_x = np.log10(x_fit)
    log_y = np.log10(y_fit)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    alpha = slope
    A = 10 ** intercept
    r_squared = r_value ** 2
    
    return alpha, A, r_squared


@st.cache_data(show_spinner=False)
def find_target_alpha_range(
    _frequencies: np.ndarray,
    _power: np.ndarray,
    target_alpha: float = -5/3,
    window_size: int = 15,
    min_points: int = 10
) -> Tuple[float, float, float]:
    """
    Find frequency range where local spectral slope matches target alpha.
    
    Uses a sliding window to compute local spectral indices and finds
    the window where the slope is closest to the target value.
    
    Args:
        frequencies: Frequency array in Hz
        power: Power spectral density array
        target_alpha: Target spectral index (e.g., -1.67 for Kolmogorov)
        window_size: Number of points in sliding window for local slope
        min_points: Minimum points required in a valid window
        
    Returns:
        Tuple of (f_min, f_max, actual_alpha) for the best matching range
        
    Raises:
        ValueError: If insufficient valid data points
    """
    # Filter to positive values only
    mask = (_frequencies > 0) & (_power > 0) & np.isfinite(_frequencies) & np.isfinite(_power)
    f_valid = _frequencies[mask]
    p_valid = _power[mask]
    
    if len(f_valid) < min_points:
        raise ValueError(f"Insufficient valid data points: {len(f_valid)}")
    
    # Work in log-log space
    log_f = np.log10(f_valid)
    log_p = np.log10(p_valid)
    
    # Ensure window size is reasonable
    window_size = min(window_size, len(log_f) // 2)
    window_size = max(window_size, min_points)
    
    best_diff = np.inf
    best_range = (f_valid[0], f_valid[-1])
    best_alpha = None
    
    # Slide window across the spectrum
    for i in range(len(log_f) - window_size + 1):
        log_f_window = log_f[i:i + window_size]
        log_p_window = log_p[i:i + window_size]
        
        # Linear regression in log-log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_f_window, log_p_window)
        
        # Check how close this slope is to target
        diff = abs(slope - target_alpha)
        
        if diff < best_diff:
            best_diff = diff
            best_range = (f_valid[i], f_valid[i + window_size - 1])
            best_alpha = slope
    
    return (best_range[0], best_range[1], best_alpha)
