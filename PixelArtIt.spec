# -*- mode: python -*-

block_cipher = None


a = Analysis(['PixelArtIt.py'],
             pathex=['C:\\Python36-32\\Lib\\site-packages\\scipy\\extra-dll', 'C:\\Users\\Evilong\\Desktop\\TrabVC', 'C:/Program Files (x86)/Windows Kits/10/Redist/ucrt/DLLs/x86'],
             binaries=[],
             datas=[],
             hiddenimports=[] ,
             hookspath=['.'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='PixelArtIt',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True , icon='PixelArtItIcon.ico')
