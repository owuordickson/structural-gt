# -*- mode: python ; coding: utf-8 -*-
import os
from glob import glob
from PyInstaller.utils.hooks import collect_submodules

ovito_dlls = glob('.venv_sgt/Lib/site-packages/ovito/plugins/*.dll')
a = Analysis(
    ['src/SGT.py'],
    pathex=[os.path.abspath("src")],  # Absolute path for reliability
    binaries=[(dll, 'ovito/plugins') for dll in ovito_dlls],
    datas=[('src/StructuralGT/apps/sgt_qml', 'StructuralGT/apps/sgt_qml')],  # Fix relative path
    hiddenimports=collect_submodules('ovito') + ['PySide6.QtQml', 'PySide6.QtQuick', 'subprocess', 'pip'],  # Add dependencies
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
    a.binaries,
    a.datas,
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
    icon=['icon.icns'],
)
app = BUNDLE(
    exe,
    name='StructuralGT.app',
    icon='icon.icns',
    bundle_identifier='edu.umich.kotov',
)
