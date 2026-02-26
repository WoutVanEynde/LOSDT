# -*- mode: python ; coding: utf-8 -*-

# All conda libraries that xtb depends on
xtb_path = '/home/wout/.anaconda/envs/LOSDT_streamlit/bin/xtb'
conda_lib = '/home/wout/.anaconda/envs/LOSDT_streamlit/lib'
xtb_libraries = [
    f'{conda_lib}/libxtb.so.6',
    f'{conda_lib}/libmctc-lib.so.0',
    f'{conda_lib}/libgfortran.so.5',
    f'{conda_lib}/libgomp.so.1',
    f'{conda_lib}/libblas.so.3',
    f'{conda_lib}/liblapack.so.3',
    f'{conda_lib}/libquadmath.so.0',
    f'{conda_lib}/libgcc_s.so.1',
]

# Build binaries list
binaries = [(xtb_path, '.')]
for lib in xtb_libraries:
    if os.path.exists(lib):
        binaries.append((lib, '.'))
    else:
        print(f"Warning: {lib} not found")

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=binaries,
    datas=[
        ('/home/wout/.anaconda/envs/LOSDT_streamlit/lib/python3.11/site-packages/streamlit/runtime', 'streamlit/runtime'),    
        ('LOSDT.py', '.'),
        ('static', 'static'),
        ('templates', 'templates'),
        ('.streamlit', '.streamlit'),
    ],
    hiddenimports=[
        'streamlit_ketcher',
        'streamlit_molstar',
        'rdkit',
        'admet_ai',
        'open3d',
        'pandas',
        'py4j.java_collections',
        'pdbtools',
        'pdbtools.pdb_selresname', 
        'pdbtools.pdb_delresname',
        'pdbtools.pdb_merge',
        'pdbtools.pdb_tocif',
    ],
    hookspath=['./hooks'],
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
    name='LOSDT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='static/LOSDT_icon.ico',
)
