@echo off
setlocal enabledelayedexpansion

cd /d C:\Users\admin\Tai_lieu\detaikhoahoc\COVID19\missing_impute\code-missing-imputation
set PYTHONPATH=C:\Users\admin\Tai_lieu\detaikhoahoc\COVID19\missing_impute\code-missing-imputation\src
set PY=C:\msys64\mingw64\bin\python.exe

echo ========================================
echo  COVID19 Missing Imputation Pipeline
echo  Full ZINB + Fusion
echo ========================================
echo.

echo [0/5] Checking Python libraries...
%PY% -c "import pandas,numpy,scipy,sklearn,pydantic,yaml,tqdm,pyarrow; print('  pandas',pandas.__version__); print('  numpy',numpy.__version__); print('  scipy',scipy.__version__); print('  sklearn',sklearn.__version__); print('  pydantic',pydantic.__version__); print('  yaml OK'); print('  tqdm OK'); print('  pyarrow',pyarrow.__version__)"
if errorlevel 1 (
    echo ERROR: Missing required Python libraries.
    echo Install with: pacman -S mingw-w64-x86_64-python-pyarrow mingw-w64-x86_64-python-pandas mingw-w64-x86_64-python-numpy mingw-w64-x86_64-python-scipy mingw-w64-x86_64-python-scikit-learn mingw-w64-x86_64-python-pydantic mingw-w64-x86_64-python-yaml mingw-w64-x86_64-python-tqdm
    pause
    exit /b 1
)
echo   All libraries OK.
echo.

echo Creating output directories...
if not exist "artifacts" mkdir "artifacts"
if not exist "artifacts\baselines" mkdir "artifacts\baselines"
if not exist "artifacts\zero_state" mkdir "artifacts\zero_state"
if not exist "artifacts\fused" mkdir "artifacts\fused"
echo.

echo [1/5] Generate dataset_00_tsagris (JSD-KNN baseline)...
%PY% -c "import os,sys;os.chdir(r'C:\Users\admin\Tai_lieu\detaikhoahoc\COVID19\missing_impute\code-missing-imputation');sys.path.insert(0,'src');from missing_imputation.cli import main;sys.argv=['missing-imputation','generate-dataset0','--config','configs/data.yaml','--method','jsd_knn','--k','7','--alpha','1.0','--output-dir','artifacts'];sys.exit(main())"
if errorlevel 1 (
    echo ERROR at step [1/5]: dataset_00 generation failed.
    pause
    exit /b 1
)
echo.

echo [2/5] Create M_observed / M_target masks...
%PY% -c "import os,sys;os.chdir(r'C:\Users\admin\Tai_lieu\detaikhoahoc\COVID19\missing_impute\code-missing-imputation');sys.path.insert(0,'src');import numpy as np;from missing_imputation.data.loader import load_config,load_covariants;from missing_imputation.data.panel import build_panel;config=load_config('configs/data.yaml');df=load_covariants(config.path,config);panel=build_panel(df,config,config.variant_components,freq_days=14);np.savez_compressed('artifacts/M_observed.npz',M_observed=panel.M_observed);np.savez_compressed('artifacts/M_target.npz',M_target=(panel.M_observed==0).astype(np.uint8));print('  Created 3D masks')"
if errorlevel 1 (
    echo ERROR at step [2/5]: mask creation failed.
    pause
    exit /b 1
)
echo.

echo [3/5] Run full ZINB zero-state pipeline...
%PY% -c "import os,sys;os.chdir(r'C:\Users\admin\Tai_lieu\detaikhoahoc\COVID19\missing_impute\code-missing-imputation');sys.path.insert(0,'src');from missing_imputation.cli import main;sys.argv=['missing-imputation','run-zinb','--data','data/covariants.csv','--observed-mask','artifacts/M_observed.npz','--config','configs/zero_state/zinb.yaml','--output-dir','artifacts/zero_state','--seed','42','--gate-mode','calibrated_threshold'];sys.exit(main())"
if errorlevel 1 (
    echo ERROR at step [3/5]: ZINB pipeline failed.
    pause
    exit /b 1
)
echo.

echo [4/5] Fusion (Tsagris + ZINB)...
%PY% -c "import os,sys;os.chdir(r'C:\Users\admin\Tai_lieu\detaikhoahoc\COVID19\missing_impute\code-missing-imputation');sys.path.insert(0,'src');from missing_imputation.cli import main;sys.argv=['missing-imputation','fuse-initializations','--raw','data/covariants.csv','--tsagris','artifacts/dataset_00_tsagris.csv','--zinb','artifacts/zero_state/dataset_01_zinb.csv','--posterior','artifacts/zero_state/zero_state_posterior.npz','--output-dir','artifacts/fused'];sys.exit(main())"
if errorlevel 1 (
    echo ERROR at step [4/5]: fusion failed.
    pause
    exit /b 1
)
echo.

echo [5/5] Verify fused dataset closure...
%PY% -c "import pandas as pd, numpy as np;df=pd.read_csv('artifacts/fused/dataset_0_fused.csv');v=['recombinant','20A','20B','20C','20E','Beta','Alpha','Gamma','Delta','Kappa','Epsilon','Eta','Iota','Lambda','Mu','Omicron','S:677'];c=df[v].values;t=df['total_sequence'].values;o=df['other'].values;print('  Rows:',len(df));print('  Closure valid:',np.allclose(c.sum(1)+o,t));print('  Non-negative:',np.all(c>=0))"
if errorlevel 1 (
    echo ERROR at step [5/5]: verification failed.
    pause
    exit /b 1
)
echo.
echo ========================================
echo  Pipeline completed successfully!
echo  Outputs in: artifacts\
echo    - dataset_00_tsagris.csv
echo    - M_observed.npz
echo    - zero_state\dataset_01_zinb.csv
echo    - zero_state\zero_state_posterior.npz
echo    - zero_state\zero_state_masks.npz
echo    - fused\dataset_0_fused.csv
echo    - fused\M_fixed.npz
echo ========================================
pause
exit /b 0
