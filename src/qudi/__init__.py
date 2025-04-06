"""
Qudi - scientific experiment control framework

This package contains the core modules of the qudi framework.
"""

import os

__version__ = '1.6.1.dev0'

# Try to read version from VERSION file if it exists
version_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'VERSION')
if os.path.isfile(version_file):
    try:
        with open(version_file, 'r') as file:
            __version__ = file.read().strip()
    except:
        # If we can't read the file, keep the hardcoded version
        pass