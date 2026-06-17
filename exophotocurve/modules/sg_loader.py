"""Small import helper for FreeSimpleGUI/PySimpleGUI compatibility."""

try: #try local import if executed as script
    #GUI import
    from FreeSimpleGUI_local import FreeSimpleGUI as sg



#     from span_modules import layouts
except ModuleNotFoundError: #local import if executed as package
    #GUI import
    from ..FreeSimpleGUI_local import FreeSimpleGUI as sg
