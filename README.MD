Simple File Exe compilation comand
pyinstaller --onefile --icon "PixelArtItIcon.ico" --paths "C:\Program Files (x86)\Python36-32\Lib\site-packages\scipy\extra-dll" "C:/Program Files (x86)/Windows Kits/10/Redist/ucrt/DLLs/x86"--additional-hooks-dir=. PixelArtIt.py
