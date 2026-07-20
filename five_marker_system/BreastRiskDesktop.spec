# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

datas = [('app.py', '.'), ('medical_system', 'medical_system'), ('data', 'data'), ('assets', 'assets')]
binaries = []
hiddenimports = [
    'streamlit.web.cli',
    'streamlit.runtime.scriptrunner.magic_funcs',
    'sklearn.calibration',
    'sklearn.ensemble._forest',
    'sklearn.ensemble._weight_boosting',
    'sklearn.ensemble._bagging',
    'sklearn.tree._classes',
    'sklearn.tree._tree',
    'sklearn.tree._criterion',
    'sklearn.tree._splitter',
    'sklearn.tree._utils',
    'sklearn.metrics._classification',
    'sklearn.metrics._ranking',
    'sklearn.model_selection._split',
    'sklearn.impute._base',
    'sklearn.pipeline',
    'sklearn.preprocessing._data',
    'sklearn.utils._cython_blas',
    'sklearn.utils._typedefs',
    'sklearn.utils._weight_vector',
]
datas += collect_data_files('streamlit')
datas += copy_metadata('streamlit')
tmp_ret = collect_all('reportlab')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('webview')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',
        'IPython', 'jupyter', 'jupyterlab', 'notebook', 'nbconvert', 'nbformat',
        'sphinx', 'pytest', 'skimage', 'astropy', 'plotly', 'bokeh',
        'distributed', 'dask', 'xarray', 'statsmodels', 'numba', 'h5py',
        'tables', 'botocore', 'boto3', 'tensorflow', 'torch', 'catboost',
        'sklearn.tests', 'sklearn.ensemble.tests', 'sklearn.tree.tests',
        'sklearn.metrics.tests', 'sklearn.model_selection.tests',
        'sklearn.utils.tests',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BreastHealthFiveMarkerDesktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/app_icon.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BreastHealthFiveMarkerDesktop',
)
