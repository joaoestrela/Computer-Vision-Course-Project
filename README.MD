### What is PixelArtIt?
PixelArtIt is a python script that uses a input image to make a new pixel art image !

### Paraments
    i             image       - Required  : input image path
    o             output      - Optional  : output image path
    s             saturation  - Optional  : increase in saturation
    c             clusters    - Optional  : number of color clusters/max number of colors in output
    d             downscaling - Optional  : downscaling scale for pixelated effect
    st            steps       - Optional  : enable saving images for each step

### Simple File Exe compilation comand
pyinstaller --onefile --icon "PixelArtItIcon.ico" --paths "C:\Program Files (x86)\Python36-32\Lib\site-packages\scipy\extra-dll" "C:/Program Files (x86)/Windows Kits/10/Redist/ucrt/DLLs/x86"--additional-hooks-dir=. PixelArtIt.py
