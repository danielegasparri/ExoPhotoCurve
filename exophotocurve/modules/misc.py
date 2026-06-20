#Miscellaneous collection of functions used by the GUI

try: #try local import if executed as script
    #GUI import
    from FreeSimpleGUI_local import FreeSimpleGUI as sg
except ModuleNotFoundError: #local import if executed as package
#     #GUI import
    from exophotocurve.FreeSimpleGUI_local import FreeSimpleGUI as sg
# 

def get_layout():
    """Function to select the layout based on the OS.
    NOTE: runtime scaling (Tk + fonts + Matplotlib DPI) is handled by ZoomManager.
    Here we only choose the layout and provide an initial scale hint.
    """
    # Set a safe base DPI once; ZoomManager will multiply this by the current scale.
    matplotlib.rcParams['figure.dpi'] = 100

    current_os = os.name  # 'posix' for Linux/Mac, 'nt' for Windows

    if current_os == "nt":
        # Keep DPI awareness on Windows for crisp rendering
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

        # OS-reported scale used only as initial HINT
        try:
            dpi_scale = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
        except Exception:
            dpi_scale = 1.0

        # If OS scale < 1.5, start from 1.5 as a comfortable default
        scale_win = 1.5 if dpi_scale < 1.5 else float(dpi_scale)

        # sg.set_options(font=("Helvetica", 11))
        default_size = 13
        return layouts.layout_windows, scale_win, None, default_size

    elif current_os == "posix":
        # Android
        if "ANDROID_BOOTLOGO" in os.environ:
            scale_win = 2.25
            sg.set_options(font=("Helvetica", 10))
            default_size = 10
            return layouts.layout_android, scale_win, None, default_size

        # macOS
        elif os.uname().sysname == "Darwin":
            # Tk widgets will scale; native titlebar/menubar will not.
            scale_win = 1.0
            sg.set_options(font=("Helvetica", 14))
            default_size = 14
            return layouts.layout_macos, scale_win, None, default_size

        # Linux
        else:
            scale_win = 1.5
            sg.set_options(font=("Helvetica", 10))
            default_size = 13
            return layouts.layout_linux, scale_win, None, default_size

    else:
        # Fallback to Linux layout + safe defaults
        scale_win = 1.5
        sg.set_options(font=("Helvetica", 10))
        default_size = 10
        return layouts.layout_linux, scale_win, None, default_size


def enable_hover_effect(window,
                        hover_color=("white", "#0078d7"),
                        exclude_keys=None):
    """
    Enable hover effect for buttons without breaking tooltips or
    dynamic colour changes. The button returns to the colour it
    had before the mouse entered, even if it was modified later.
    """
    if exclude_keys is None:
        exclude_keys = []

    for key, element in window.AllKeysDict.items():
        if isinstance(element, sg.Button) and key not in exclude_keys:
            # Define local variable to store the original colour per element
            def on_enter(event, el=element):
                el._normal_color_before_hover = el.ButtonColor
                el.update(button_color=hover_color)

            def on_leave(event, el=element):
                normal_color = getattr(el, "_normal_color_before_hover", el.ButtonColor)
                el.update(button_color=normal_color)

            element.Widget.bind("<Enter>", on_enter, add="+")
            element.Widget.bind("<Leave>", on_leave, add="+")
