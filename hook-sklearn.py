from PyInstaller.utils.hooks import collect_data_files, eval_statement, collect_submodules
hiddenimports = collect_submodules('sklearn')
