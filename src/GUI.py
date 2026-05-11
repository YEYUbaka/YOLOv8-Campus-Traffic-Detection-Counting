import os
import sys
import warnings

import PyQt5

dirname = os.path.dirname(PyQt5.__file__)
qt_dir = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_dir

warnings.filterwarnings("ignore", category=DeprecationWarning)

# PyInstaller 打包后路径兼容
if getattr(sys, 'frozen', False):
    _base_dir = os.path.dirname(sys.executable)
    _bundle_dir = sys._MEIPASS
else:
    _base_dir = None
    _bundle_dir = None


def get_resource_path(relative_path):
    """获取只读资源路径（QSS 等，打包后在 _MEIPASS 中）"""
    if _bundle_dir:
        return os.path.join(_bundle_dir, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def get_external_path(relative_path):
    """获取外部可写路径（模型、配置等，始终在 exe 旁边）"""
    if _base_dir:
        return os.path.join(_base_dir, relative_path)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, relative_path)


from gui.window import main

if __name__ == '__main__':
    main()
