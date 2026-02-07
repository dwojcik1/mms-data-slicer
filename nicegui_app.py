from nicegui import ui
import os

# Proof of Concept: MMS Data Slicer in NiceGUI

def main():
    with ui.header().classes('bg-blue-700 text-white items-center gap-2'):
        ui.label('MMS Data Slicer').classes('text-2xl font-bold')
        ui.label('NiceGUI Edition').classes('text-sm italic opacity-80')

    with ui.left_drawer(value=True).classes('bg-gray-100 p-4'):
        ui.label('Data Source').classes('text-lg font-bold mb-2')
        
        ui.select(['1', '2', '3', '4'], value='1', label='Probe').classes('w-full mb-2')
        ui.select(['fgm', 'fpi', 'scm', 'edp'], value='fgm', label='Instrument').classes('w-full mb-2')
        ui.select(['srvy', 'brst'], value='srvy', label='Data Rate').classes('w-full mb-2')
        ui.select(['gsm', 'gse', 'dmpa'], value='gsm', label='Coordinates').classes('w-full mb-2')
        
        ui.separator()
        ui.button('Load Data', on_click=lambda: ui.notify('Data loading not implemented in POC')).classes('w-full bg-blue-600 text-white')

    with ui.column().classes('w-full p-4'):
        ui.markdown('''
        ### Welcome to the NiceGUI Version
        
        This is a Proof of Concept (POC) demonstrating how the functionality of the Streamlit app could be migrated to **NiceGUI**.
        
        **Advantages of NiceGUI:**
        - Standard Python event loop (no script re-runs).
        - Native UI elements (Quasar/Vue).
        - Faster interaction for complex apps.
        
        *Currently, the backend logic from `downloader.py` and `physics.py` needs to be integrated.*
        ''')
        
        with ui.row().classes('gap-4'):
            ui.card().classes('w-64 h-32 bg-blue-50 flex items-center justify-center').content('Time Series Placeholder')
            ui.card().classes('w-64 h-32 bg-green-50 flex items-center justify-center').content('PSD Placeholder')

    ui.run(title='MMS Data Slicer')

if __name__ in {"__main__", "__mp_main__"}:
    main()
