@echo off
rem ================================================================
rem  PDF:Paper Decoding Flow v2.1 - Windows entry (self-bootstrapping)
rem  Use 1: drop one or more PDF files onto this .bat icon
rem  Use 2: run it, then drag a PDF path into the prompt + Enter
rem  First run: auto-creates venv, installs deps, guides API key.
rem  Result : output\<paper title>\  (Explorer opens when done)
rem  NOTE   : messages kept in ASCII to avoid console codepage issues
rem ================================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "MINERU_MODEL_SOURCE=modelscope"
set "PIP_NO_INPUT=1"

rem ---------- collect input files (dropped args first, prompt fallback) ----------
set "ARGS=%*"
if defined ARGS goto :setup

set "PDFS="
set /p "PDFS=Drag PDF into this window (or paste full path), then Enter: "
if not defined PDFS goto :empty
set "PDFS=%PDFS:"=%"
if not exist "%PDFS%" goto :notfound
set "ARGS="%PDFS%""

rem ---------- environment fingerprint: size+mtime of pyproject.toml ----------
:setup
set "VPY=.venv\Scripts\python.exe"
set "MARKER=.venv\.setup_ok"

set "PH="
for %%F in (pyproject.toml) do set "PH=%%~zF_%%~tF"

set "SAVED="
if exist "%MARKER%" for /f "usebackq delims=" %%S in ("%MARKER%") do if not defined SAVED set "SAVED=%%S"

if exist "%VPY%" if "%SAVED%"=="%PH%" goto :config
goto :bootstrap

:bootstrap
rem ---------- find Python 3.10+ (py launcher, then python, then winget) ----------
set "BOOTPY="
where py >nul 2>&1 && set "BOOTPY=py -3"
if not defined BOOTPY ( where python >nul 2>&1 && set "BOOTPY=python" )
if defined BOOTPY goto :make_venv

where winget >nul 2>&1 || goto :no_python
echo [setup] Python not found. Trying one-time install via winget...
winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "BOOTPY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined BOOTPY goto :make_venv
( where py >nul 2>&1 && set "BOOTPY=py -3" )
if defined BOOTPY goto :make_venv
echo [setup] Python was installed, but this window cannot see it yet.
echo         Close this window, then run run.bat again.
pause
exit /b 1

:no_python
echo [setup] Python 3.10+ not found and winget is unavailable.
echo         Install Python from https://www.python.org/downloads/
echo         (tick "Add python.exe to PATH"), then run run.bat again.
pause
exit /b 1

:make_venv
if exist "%VPY%" goto :install
echo [setup] creating virtualenv (.venv) ...
%BOOTPY% -m venv .venv
if not exist "%VPY%" (
    echo [setup] venv creation failed. Check the message above, then run again.
    pause
    exit /b 1
)

:install
echo [setup] installing dependencies - first run may take several minutes.
echo         Re-running this file resumes an interrupted install.
"%VPY%" -m pip install -e .
if errorlevel 1 (
    echo [setup] pip install failed. Fix the error above, then run run.bat again.
    pause
    exit /b 1
)
> "%MARKER%" echo %PH%
echo [setup] environment ready.

rem ---------- first-run API key guidance ----------
:config
if exist config\config.yaml goto :ready
echo.
echo [setup] creating config\config.yaml from template. Notepad will open.
copy /y config\config.example.yaml config\config.yaml >nul
start "" notepad config\config.yaml
echo   - fill api_key / base_url for your OpenAI-compatible provider
echo     (default: SiliconFlow; VLM=Qwen3-VL, LLM=DeepSeek), save and close Notepad.
echo   - STRONGLY RECOMMENDED: fill api.zhipu_api_key for free mode
echo     (Zhipu GLM-4.6V/4.7-Flash, zero cost; get key at open.bigmodel.cn)
echo     WARNING: the free tier is shared and often throttled at peak
echo     times. Calls wait and retry automatically, so a paper may take
echo     30-60 min in free mode. If it stalls for very long, just re-run
echo     later (finished units are cached) or switch to default mode.
echo   - optional local mode: install Ollama + import a translation model
echo     (GGUF recommended: gemma3-4b-it Q4_K_M, ~2.8GB VRAM; see README)
set /p "_=Press Enter here after saving config\config.yaml: "

rem ---------- run pipeline for each file ----------
:ready
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
rem local pipeline talks to Ollama on localhost - never route it via a proxy
set "NO_PROXY=localhost,127.0.0.1"
set "no_proxy=localhost,127.0.0.1"
set "OKCNT=0"
set "FAILCNT=0"

rem ---------- mode selection (five peer modes) ----------
set "MODEFLAG="
echo.
echo Select mode:
echo   1^) default  ^(recommended^)  - SiliconFlow cloud (paid)
echo   2^) free                    - Zhipu GLM cloud (free tier)
echo   3^) local                   - Ollama on your GPU (S3 via local VLM if configured, no cloud)
echo   4^) minimum                 - parse only (no API at all)
echo   5^) turbo      ^(EXPERIMENTAL^) - SiliconFlow cloud, concurrent S3+S4 overlap (faster)
set /p "MODE=Input 1-5 (Enter=1): "
if /i "%MODE%"=="2" set "MODEFLAG=--mode free"
if /i "%MODE%"=="2" echo [mode] free runs on a shared free tier - expect throttling waits at peak times; progress is cached and resumable.
if /i "%MODE%"=="3" set "MODEFLAG=--mode local"
if /i "%MODE%"=="4" set "MODEFLAG=--mode minimum"
if /i "%MODE%"=="5" set "MODEFLAG=--mode turbo"
if /i "%MODE%"=="5" echo [mode] turbo is experimental: same cloud/models as default, S3 and S4 run concurrently with a shared rate limiter. If it misbehaves, switch back to 1^) default - caches are shared, no double billing.
if /i "%MODE%"=="5" goto :turbo_summary
goto :modes_done2

:turbo_summary
rem --- turbo only: ask about optional overview (S6). Explicit flag either way, ---
rem --- so this prompt is the single source of truth (config default not consulted). ---
set "SUMFLAG=--no-summary"
set /p "SUM=Also generate quick-read overview file (gai-lan.md)? 1=yes / 0=no (Enter=0): "
if "%SUM%"=="1" set "SUMFLAG=--summary"
if /i "%SUM%"=="y" set "SUMFLAG=--summary"
if /i "%SUM%"=="yes" set "SUMFLAG=--summary"
set "MODEFLAG=--mode turbo %SUMFLAG%"
echo [mode] turbo overview: %SUMFLAG:2=%

:modes_done2

rem ---------- mode guards (top-level, no for /f backtick: cmd splits quoted ----------
rem ---------- commands inside backticks - use a temp file handshake instead)  ----------
if exist _zp.tmp del _zp.tmp >nul 2>&1
set "ZP=NO"
if "%MODEFLAG%"=="--mode free" "%VPY%" -c "import yaml;c=yaml.safe_load(open('config/config.yaml',encoding='utf-8')) or {};a=(c.get('api') or {});k=(a.get('zhipu_api_key') or '');open('_zp.tmp','w').write('HAS' if str(k).strip() else 'NO')" >nul 2>&1
if "%MODEFLAG%"=="--mode free" for /f %%K in (_zp.tmp) do set "ZP=%%K"
if "%MODEFLAG%"=="--mode free" if not "%ZP%"=="HAS" (
    echo [mode] free needs api.zhipu_api_key in config\config.yaml - fill it, save, retry.
    pause
    exit /b 1
)
if exist _zp.tmp del _zp.tmp >nul 2>&1

if "%MODEFLAG%"=="--mode local" goto :local_guard
goto :modes_done

:local_guard
rem --- tier 1: service already running (and really is Ollama)? ---
set "OL=NO"
if exist _ol.tmp del _ol.tmp >nul 2>&1
"%VPY%" -c "import urllib.request as u; o=u.build_opener(u.ProxyHandler({})); r=o.open('http://127.0.0.1:11434/api/tags', timeout=2); open('_ol.tmp','w').write('OK' if b'"models"' in r.read() else 'BAD')" >nul 2>&1
if exist _ol.tmp for /f %%T in (_ol.tmp) do set "OL=%%T"
if exist _ol.tmp del _ol.tmp >nul 2>&1
if "%OL%"=="OK" goto :local_model_check

rem --- tier 2: find ollama.exe and auto-launch the service ---
set "OLLAMA_EXE="
where ollama.exe >nul 2>&1 && set "OLLAMA_EXE=ollama.exe"
if not defined OLLAMA_EXE if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined OLLAMA_EXE if exist "E:\ollama_dist\ollama.exe" set "OLLAMA_EXE=E:\ollama_dist\ollama.exe"
if not defined OLLAMA_EXE (
    echo [mode] local needs Ollama. It is not installed or not on PATH.
    echo        See README for install + GGUF import steps, then retry.
    pause
    exit /b 1
)
rem models dir: keep user's env override, else known portable dir, else Ollama default
if not defined OLLAMA_MODELS if exist "E:\ollama_models" set "OLLAMA_MODELS=E:\ollama_models"
echo [mode] local: Ollama service not running - starting "%OLLAMA_EXE%" ...
start "Ollama" /min "%OLLAMA_EXE%" serve
set "OL=NO"
set /a OL_TRIES=0
:local_poll
ping -n 2 127.0.0.1 >nul
if exist _ol.tmp del _ol.tmp >nul 2>&1
"%VPY%" -c "import urllib.request as u; o=u.build_opener(u.ProxyHandler({})); r=o.open('http://127.0.0.1:11434/api/tags', timeout=2); open('_ol.tmp','w').write('OK' if b'"models"' in r.read() else 'BAD')" >nul 2>&1
if exist _ol.tmp for /f %%T in (_ol.tmp) do set "OL=%%T"
if exist _ol.tmp del _ol.tmp >nul 2>&1
if "%OL%"=="OK" goto :local_model_check
set /a OL_TRIES+=1
if %OL_TRIES% lss 15 goto :local_poll
echo [mode] local: started Ollama but the service did not come up in ~15s.
echo        Possible port conflict - check the Ollama window, then retry.
pause
exit /b 1

:local_model_check
rem --- tier 3: at least one model must be imported (lazy-load, no pre-mount needed) ---
set "OLM=EMPTY"
if exist _ol.tmp del _ol.tmp >nul 2>&1
"%VPY%" -c "import json,urllib.request as u; o=u.build_opener(u.ProxyHandler({})); d=json.load(o.open('http://127.0.0.1:11434/api/tags', timeout=3)); open('_ol.tmp','w').write('MODEL' if d.get('models') else 'EMPTY')" >nul 2>&1
if exist _ol.tmp for /f %%T in (_ol.tmp) do set "OLM=%%T"
if exist _ol.tmp del _ol.tmp >nul 2>&1
if "%OLM%"=="MODEL" goto :modes_done
echo [mode] local: Ollama is running, but no model is imported yet.
echo        Import a translation GGUF first (recommended gemma3-4b-it Q4_K_M,
echo        ~2.8GB VRAM) - step-by-step guide (incl. agent prompt) in README.
pause
exit /b 1

:modes_done

for %%F in (%ARGS%) do call :run_one %%F

echo.
if "%FAILCNT%"=="0" (
    echo ======== DONE. %OKCNT% ok. Results in output\ under paper title, Explorer opened. ========
) else (
    echo ======== Finished: %OKCNT% ok, %FAILCNT% failed. See error messages above. ========
)
pause
exit /b %FAILCNT%

:run_one
echo.
echo ======== Processing: %~1 ========
echo    First parse of a paper downloads MinerU models once and takes ~10 min.
echo    Keep this window open. CLI also supports --mode default/turbo/free/local/minimum.
"%VPY%" -m paper_translate %MODEFLAG% --open %1
if not errorlevel 1 goto :run_ok
rem no paren-block here: file names may contain ")" and break block parsing
echo [run] FAILED: %~1
set /a FAILCNT+=1
goto :eof
:run_ok
set /a OKCNT+=1
goto :eof

:empty
echo No input path.
pause
exit /b 1

:notfound
echo File not found: %PDFS%
pause
exit /b 1
