"""
Physics models integration for MMS Data Slicer.
Handles execution of external field models (Tsyganenko) via PySPEDAS/PyTplot.
"""

import pandas as pd
import streamlit as st
import numpy as np

def run_mms_tsyganenko(trange, probe='1'):
    """
    Executes the full pipeline: Load MMS Data -> Load Solar Wind -> Run T96 -> Calculate Residuals.
    
    Args:
        trange (list): Start and end time ['YYYY-MM-DD/HH:MM', 'YYYY-MM-DD/HH:MM']
        probe (str): MMS probe identifier ('1', '2', '3', or '4')
        
    Returns:
        dict: Dictionary containing 'measured', 'model', 'residual' dataframes
              and 'metadata' dictionary.
    """
    # Lazy import to prevent "Python stopped working" crashes on module load
    # associated with compiled Fortran extensions in geopack
    import pyspedas
    from pytplot import get_data, store_data, join_vec, options, tplot_names, del_data
    from pyspedas.geopack import tt96
    from pyspedas.geopack import get_tsy_params

    st.info(f"Step 1: Loading MMS{probe} Position (GSM)...")
    pyspedas.mms.mec(trange=trange, probe=probe, data_rate='srvy', time_clip=True, varformat='*gse*|*gsm*')
    
    pos_var = f'mms{probe}_mec_r_gsm'
    if pos_var not in tplot_names(quiet=True):
        st.error(f"Position variable {pos_var} failed to load.")
        return None

    st.info(f"Step 2: Loading MMS{probe} Magnetometer Data (GSM)...")
    pyspedas.mms.fgm(trange=trange, probe=probe, data_rate='srvy', time_clip=True)
    
    # We use the B-vector in GSM to match the model output.
    meas_mag_var = f'mms{probe}_fgm_b_gsm_srvy_l2_bvec'
    if meas_mag_var not in tplot_names(quiet=True):
         # Try fallback if exact name doesn't match standard pattern (sometimes happens with latest pyspedas)
         meas_mag_var = f'mms{probe}_fgm_b_gsm_srvy_l2'
         if meas_mag_var not in tplot_names(quiet=True):
             st.error("Could not find FGM GSM variable.")
             return None

    st.info("Step 3: Loading Solar Wind (OMNI) & Interpolating T96 Inputs...")
    # Load OMNI data
    pyspedas.omni.data(trange=trange)

    # T96 Model Inputs
    model_input_var = 't96_inputs'
    get_tsy_params(
        'OMNI_HRO_1min_DST',                # Dst index
        'OMNI_HRO_1min_IMF',                # IMF Vectors
        'OMNI_HRO_1min_proton_density',     # Density (N)
        'OMNI_HRO_1min_SW_Plasma_Flow_Speed', # Velocity (V)
        newname=model_input_var, 
        model='t96', 
        speed=True  # Indicates flow input is scalar speed
    )

    st.info("Step 4: Running Tsyganenko 96 (T96) Model...")
    model_output_var = f'mms{probe}_b_t96'
    
    # tt96 calculates the total field (External T96 + Internal IGRF)
    # tt96 uses the time tags from pos_var
    tt96(
        pos_var,                # Input: Spacecraft Position
        parmod=model_input_var, # Input: Solar Wind Parameters
        newname=model_output_var # Output: Modeled B-field
    )

    st.info("Step 5: Calculating Residuals (Measured - Model)...")
    
    # Interpolate Measured FGM data to the Model timestamps (Position timestamps)
    # Model/Pos data is usually 30s or 60s (MEC), FGM is 8/16Hz or similar. 
    # We downsample FGM to Model resolution for clean subtraction.
    meas_interp_var = f'{meas_mag_var}_interp'
    pyspedas.tinterpol(meas_mag_var, pos_var, newname=meas_interp_var)
    
    # Extract data for pandas conversion
    times_model, data_model = get_data(model_output_var)
    times_meas, data_meas = get_data(meas_interp_var)
    
    # Calculate Residuals (Delta B)
    # Just in case tinterpol didn't align perfectly (should match pos_var), 
    # we use numpy directly since shapes should match now.
    if data_meas.shape != data_model.shape:
        # Fallback: align by intersection if needed, or Trim
        min_len = min(len(data_meas), len(data_model))
        data_meas = data_meas[:min_len]
        data_model = data_model[:min_len]
        times_model = times_model[:min_len]
        
    residuals = data_meas - data_model
    
    # --- Convert to DataFrames for Frontend ---
    
    # Model DataFrame
    df_model = pd.DataFrame(data_model, index=times_model, columns=['Bx_T96', 'By_T96', 'Bz_T96'])
    
    # Measured DataFrame (Interpolated)
    df_measured = pd.DataFrame(data_meas, index=times_model, columns=['Bx_Meas', 'By_Meas', 'Bz_Meas'])
    
    # Residual DataFrame
    df_residual = pd.DataFrame(residuals, index=times_model, columns=['dBx', 'dBy', 'dBz'])
    
    # --- Cleanup tplot variables to free memory ---
    # Optional: Clear everything created in this session? 
    # For now, keep them in case user wants something else, 
    # but maybe delete the intermediate ones.
    # del_data([model_input_var, meas_interp_var])

    return {
        'model': df_model,
        'measured': df_measured,
        'residual': df_residual,
        'metadata': {
            'probe': probe,
            'model': 'T96 + IGRF',
            'coords': 'GSM'
        }
    }
