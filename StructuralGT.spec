# -*- mode: python ; coding: utf-8 -*-

import os
a = Analysis(
    ['src/StructuralGT.py'],
    pathex=[os.path.abspath("src")],  # Absolute path for reliability
    binaries=[],
    datas=[('src/StructuralGT/apps/sgt_qml', 'StructuralGT/apps/sgt_qml')],  # Fix relative path
    hiddenimports=['PySide6.QtQml', 'PySide6.QtQuick'],  # Add PySide6 dependencies
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='StructuralGT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='StructuralGT',
)
