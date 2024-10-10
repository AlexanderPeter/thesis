# Expiremental code

## Useful commands

### Format Workspace

```
npm -g install prettier
```

```
prettier --write .
```

### Open Jupyter

```
jupyter notebook
```

### Execute LaTeX manually

```
cd documentation
pdflatex Documentation
biber Documentation
pdflatex Documentation
pdflatex Documentation
```

### Execute python script

```
py .\notebooks\python\common_functions.py
```

### Execute python linting

```
pylint notebooks/python
```

### Execute python tests

```
python -m unittest notebooks/python/tests/common_functions_tests.py
coverage run -m unittest notebooks/python/tests/common_functions_tests.py
coverage report -m
```

## Frequent errors

### Could not install packages due to an OSError: [WinError 5] Access is denied

```
python -m pip install <package>
```

### ModuleNotFoundError: No module named 'pip'

```
python -m ensurepip
```

### DataLoader worker (pid(s) ...) exited unexpectedly

set num_workers=0

### Fatal error in launcher: Unable to create process using '"..." "..." install ...': The system cannot find the file specified.

```bash
.venv/Scripts/python.exe -m pip install --upgrade --force-reinstall pip
```

```powershell
\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall pip
```

### I can't find file `"|texlua ./report.markdown.lua"'.

Use `-enable-write18` or `shell-escape` when generating pdf

### Cannot find control file 'report.bcf'!

Close editor, reopen and try again
