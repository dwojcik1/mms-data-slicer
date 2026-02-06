# MMS Turbulence Laboratory 🛰️

**Kinetic scale time series processing for space plasma physics.**

The **MMS Turbulence Laboratory** is a specialized tool designed for the analysis of high-resolution magnetic field and plasma data from the **Magnetospheric Multiscale (MMS)** mission. It provides an interactive interface for slicing, visualizing, and calculating statistical properties of turbulence in the Earth's magnetosphere.

![MMS](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Magnetospheric_Multiscale_Mission.jpg/800px-Magnetospheric_Multiscale_Mission.jpg)

## 🚀 Features

### 1. Data Retrieval & Management
-   **Automated Downloader**: Seamless integration with CDAWeb to search and download MMS Fluxgate Magnetometer (FGM) data (Burst and Fast modes).
-   **Local Caching**: Efficient file management to minimize repeated downloads.

### 2. Time Series Analysis
-   **Interactive Plotting**: High-performance visualization of magnetic field components ($B_x, B_y, B_z, |B|$) using `plotly`.
-   **Subsampling**: Dynamic data thinning for responsive UI interaction without losing local extrema.

### 3. Power Spectral Density (PSD)
-   **Welch's Method**: Robust estimation of spectral density.
-   **Spectral Index Fitting**: Physics-informed fitting algorithms (Savitzky-Golay smoothing + Sliding Decade window) to determine inertial range slopes (e.g., -5/3).
-   **Publication-Ready Exports**: Clean, square-aspect ratio plots with standardized metadata.

### 4. PDF & Statistical Moments
-   **Probability Density Functions**: Analyze the distribution of magnetic fluctuations.
-   **Advanced Visualization**: 
    -   Dual view: Histograms and Kernel Density Estimation (KDE).
    -   Multiple Kernels: Gaussian, Tophat, Epanechnikov, Cosine, etc.
-   **Higher-Order Stats**: Real-time calculation of Mean, Variance, **Skewness**, and **Kurtosis**.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/mms-turbulence-lab.git
    cd mms-turbulence-lab
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *Requires `streamlit`, `numpy`, `pandas`, `scipy`, `plotly`, `cdasws`, and `scikit-learn`.*

## 🖥️ Usage

Run the Streamlit application:

```bash
streamlit run main.py
```

Open your browser to `http://localhost:8501`.

1.  **Load Data**: Use the sidebar to select MMS Probe, Instrument (FGM), Date, and Mode.
2.  **Slice**: Use the time slider to focus on specific burst intervals.
3.  **Analyze**: Switch between tabs (Time Series, PSD, PDF) to perform specific analyses.

## 🔮 Future Roadmap

The tool is actively developing towards a comprehensive suite for turbulence and reconnection analysis:

*   **Advanced Spectral Analysis:**
    *   Full **PSD** (Power Spectral Density) and **ESD** (Energy Spectral Density) integration.
    *   Wavelet transforms for time-frequency localization.
*   **Structure Detection:**
    *   Automated prediction of **Reconnection** events.
    *   Identification of **Electron Diffusion Regions (EDR)** using geometric and kinetic signatures.
*   **Intermittency Analysis:**
    *   **Scale-dependent analysis** (analysis in scale $\tau$).
    *   Investigation of **Flatness**, Skewness, and Kurtosis as functions of scale.
*   **Orbit Visualization**: 3D trajectory plots of the MMS formation.

---
*Built with Python & Streamlit.*
