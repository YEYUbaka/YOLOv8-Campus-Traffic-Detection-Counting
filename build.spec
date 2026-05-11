# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for YOLOv8 Campus Traffic Detection GUI."""

import os

block_cipher = None
src_dir = os.path.join(os.path.dirname(os.path.abspath(SPEC)), 'src')

a = Analysis(
    [os.path.join(src_dir, 'GUI.py')],
    pathex=[src_dir],
    binaries=[],
    datas=[
        # 样式文件 → 打包到 _MEIPASS/styles/
        (os.path.join(src_dir, 'styles'), 'styles'),
        # 模型文件 → 打包到 _MEIPASS/models/
        (os.path.join(os.path.dirname(os.path.abspath(SPEC)), 'models'), 'models'),
        # 配置文件 → 打包到 _MEIPASS/configs/
        (os.path.join(os.path.dirname(os.path.abspath(SPEC)), 'configs'), 'configs'),
    ],
    hiddenimports=[
        'ultralytics',
        'torch',
        'torchvision',
        'cv2',
        'PIL',
        'numpy',
        'yaml',
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='校园交通检测系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,       # GUI 应用，不显示控制台
    icon=None,
    uac_admin=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='校园交通检测系统',
)
