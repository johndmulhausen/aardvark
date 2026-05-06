# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('/Users/john/streamlit-inference/streamlit_app.py', '.'), ('/Users/john/streamlit-inference/agent.py', '.'), ('/Users/john/streamlit-inference/tools.py', '.'), ('/Users/john/streamlit-inference/wb_client.py', '.')]
binaries = []
hiddenimports = ['__future__.annotations', 'agent.run_agent_turn', 'json', 'os', 'pathlib.Path', 'streamlit', 'typing.Any', 'wb_client.list_models', 'wb_client.make_client']
datas += copy_metadata('streamlit')
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('openai')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['/var/folders/9_/0cjkgxsx73181gc97v4v7b280000gp/T/tmp_r909y_6.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name='WB Coding Agent',
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WB Coding Agent',
)
app = BUNDLE(
    coll,
    name='WB Coding Agent.app',
    icon=None,
    bundle_identifier=None,
)
