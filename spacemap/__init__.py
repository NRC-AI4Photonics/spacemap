"""
Adapted from Lumerical lumopt __init__.py
"""
import os, sys, platform

# Import Lumerical Python API
# look for lumapi.py in system path
python_api_path = ''
for dir_name in sys.path:
    if os.path.isfile(os.path.join(dir_name, 'lumapi.py')):
        python_api_path = dir_name; break
# if search comes out empty, look in the default install path
if not python_api_path:
    current_platform = platform.system()
    default_api_path = ''
    if current_platform == 'Windows':
        default_api_path = '/Program Files/Lumerical/2019b/api/python'
    elif current_platform == 'Darwin':
        default_api_path = '/Applications/Lumerical/FDTD/FDTD.app/Contents/MacOS/'
    elif current_platform == 'Linux':
        default_api_path = '/opt/lumerical/2019b/api/python'
    default_api_path = os.path.normpath(default_api_path)
    if os.path.isfile(os.path.join(default_api_path, 'lumapi.py')):
        sys.path.append(default_api_path)
        python_api_path = default_api_path
    
    


from .mapping import SpaceMapping
from .utilities.parameters import Settings
