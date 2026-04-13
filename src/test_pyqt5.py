# test_pyqt5.py
import os, PyQt5; print(os.path.dirname(PyQt5.__file__))

# import os, PyQt5
dirname = os.path.dirname(PyQt5.__file__)
qt_dir = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_dir