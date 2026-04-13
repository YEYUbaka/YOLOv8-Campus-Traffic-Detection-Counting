
"""Thin launcher for the refactored GUI package."""

import os
import warnings

import PyQt5

dirname = os.path.dirname(PyQt5.__file__)
qt_dir = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_dir

warnings.filterwarnings("ignore", category=DeprecationWarning)

from gui.window import main


if __name__ == '__main__':
    main()
