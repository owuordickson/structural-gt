# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules

a = Analysis(
    ['src/StructuralGT.py'],
    pathex=[os.path.abspath("src")],  # Absolute path for reliability
    binaries=[],
    datas=[('src/StructuralGT/apps/sgt_qml', 'StructuralGT/apps/sgt_qml')],  # Fix relative path
    #hiddenimports=['PySide6.QtQml', 'PySide6.QtQuick'],  # Add PySide6 dependencies
    hiddenimports=collect_submodules('PySide6'),  # Auto-detect PySide6 dependencies
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

"""splash = Splash(
    'splash.jpg',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(10, 50),
    text_size=12,
    text_color='black'
)"""

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    #splash,                   # <-- both, splash target
    #splash.binaries,          # <-- and splash binaries
    [],
    name='StructuralGT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
