@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =========================================================
REM  Active Tunnel - Interactive Runner
REM
REM  Selection logic:
REM    field (1 or 2)  +  letter (e.g. B2)  +  channel (e.g. NIR)
REM    => builds key:  field1_B2_NIR
REM    Type "all" to run every position in the map.
REM
REM  Steps:
REM    1 - Rename raw export  -> ACDC_IN_<PHASE>
REM    2 - Build segmentation_input + position_map.csv
REM    3 - Segmentation + tracking  (Cell-ACDC)
REM    4 - TrackPy gap-closing linking  (bridges 1-2 frame gaps -> persistent track IDs)
REM    5 - Filter short tracks  (optional)
REM    6 - Export Imaris-like Excel
REM =========================================================

REM ── EDIT THIS PATH IF CONDA IS IN A DIFFERENT LOCATION ──────────────────────
set "CONDA_BAT=C:\Users\wasee\anaconda3\condabin\conda.bat"
REM ─────────────────────────────────────────────────────────────────────────────

REM ── Project root = folder where this .bat lives ─────────────────────────────
set "ROOT=%~dp0"
cd /d "%ROOT%"

echo.
echo ========================================
echo   Active Tunnel - Interactive Runner
echo   Root : %ROOT%
echo   Conda: %CONDA_BAT%
echo ========================================
echo.

if not exist "%CONDA_BAT%" (
  echo [ERROR] conda.bat not found at:
  echo   %CONDA_BAT%
  echo Fix the CONDA_BAT path at the top of this file.
  pause
  exit /b 1
)

REM ============================================================
REM  USER INPUTS
REM ============================================================

set "PHASE_NUM=4"
set /p PHASE_NUM=Phase number (e.g. 4 for Phase4)
if "!PHASE_NUM!"=="" set "PHASE_NUM=4"
set "PHASE=Phase!PHASE_NUM!"

echo.
echo  Steps available:
echo    1 - Rename raw export  (needs PhaseX_DATA folder)
echo    2 - Build segmentation_input
echo    3 - Segmentation + tracking  (Cell-ACDC)
echo    4 - TrackPy gap-closing linking  (bridges missed frames, persistent track IDs)
echo    5 - Filter short tracks
echo    6 - Export Imaris-like Excel
echo.
set "START_STEP=1"
set /p START_STEP=Start from step (1-6, default 1)
if "!START_STEP!"=="" set "START_STEP=1"

echo.
echo  Run mode: type "all" or specify field+letter+channel
echo  Example: field=1  letter=B2  channel=NIR  =>  field1_B2_NIR
echo.

set "SEL_INPUT="
set /p SEL_INPUT=Run mode - type "all" to run every position, or press Enter to specify one

REM -- Use goto to avoid set /p inside if/else blocks -------------------------
if /I "!SEL_INPUT!"=="all" goto MODE_ALL

REM ── mode = one: ask for field/letter/channel ─────────────────────────────
set "MODE=one"
set "FIELD_NUM=1"
set /p FIELD_NUM=Field number (1 or 2)
set "LOCATION="
set /p LOCATION=Location letter (e.g. B2, C4)
set "CHANNEL="
set /p CHANNEL=Channel name (NIR, GREEN, ORANGE)
if "!FIELD_NUM!"=="" set "FIELD_NUM=1"
set "FIELD_FOLDER=field!FIELD_NUM!"
set "SELECT=!FIELD_FOLDER!_!LOCATION!_!CHANNEL!"
echo.
echo   Selection key: !SELECT!
goto MODE_DONE

:MODE_ALL
set "MODE=all"
set "SELECT="
set "FIELD_FOLDER="
set "LOCATION="
set "CHANNEL="

:MODE_DONE

echo.
set "ACDC_ENV=acdc"
set /p ACDC_ENV=Conda env for segmentation/tracking
if "!ACDC_ENV!"=="" set "ACDC_ENV=acdc"

set "EXPORT_ENV=imaris_xls"
set /p EXPORT_ENV=Conda env for export (step 6)
if "!EXPORT_ENV!"=="" set "EXPORT_ENV=imaris_xls"

set "TRACKPY_ENV=acdc_trackpy"
set /p TRACKPY_ENV=Conda env for TrackPy linking (step 4)
if "!TRACKPY_ENV!"=="" set "TRACKPY_ENV=acdc_trackpy"

set "APPLY_PATCH=n"
set /p APPLY_PATCH=Apply TrackPy CLI patch? (y/n)
if "!APPLY_PATCH!"=="" set "APPLY_PATCH=n"

REM ── TrackPy linking parameters ─────────────────────────────────────────────
echo.
echo  TrackPy linking parameters (step 4):
echo    search-range : max pixels a cell can move per frame  (rule of thumb: 1.5x max displacement)
echo    memory       : max frames a cell can disappear and still be re-linked  (gap closing)
echo.
set "TP_RANGE=15"
set /p TP_RANGE=TrackPy search range in pixels (default 15)
if "!TP_RANGE!"=="" set "TP_RANGE=15"

set "TP_MEMORY=2"
set /p TP_MEMORY=TrackPy memory - max gap frames to bridge (default 2)
if "!TP_MEMORY!"=="" set "TP_MEMORY=2"

set "TP_MIN_FRAMES=3"
set /p TP_MIN_FRAMES=TrackPy min track frames to keep (default 3)
if "!TP_MIN_FRAMES!"=="" set "TP_MIN_FRAMES=3"

REM ── Filter step ────────────────────────────────────────────────────────────
echo.
set "MIN_FRAMES=15"
set /p MIN_FRAMES=Min track length for ACDC filter step (0 = skip step 5)
if "!MIN_FRAMES!"=="" set "MIN_FRAMES=15"

set "PX=0.108"
set /p PX=Pixel size (um)
if "!PX!"=="" set "PX=0.108"

set "ZSTEP=1.0"
set /p ZSTEP=Z-step (um)
if "!ZSTEP!"=="" set "ZSTEP=1.0"

set "DT=3600"
set /p DT=Frame interval (s)
if "!DT!"=="" set "DT=3600"

echo.
echo ======================================
echo   PIPELINE SUMMARY
echo ======================================
echo   Phase            : !PHASE!
echo   Start from step  : !START_STEP!
echo   Mode             : !MODE!
if /I "!MODE!"=="one" (
  echo   Selection key    : !SELECT!
)
echo   ACDC env         : !ACDC_ENV!
echo   Export env       : !EXPORT_ENV!
echo   TrackPy env      : !TRACKPY_ENV!
echo   TrackPy patch    : !APPLY_PATCH!
echo   --- TrackPy linking (step 4) ---
echo   Search range     : !TP_RANGE! px
echo   Memory (gap)     : !TP_MEMORY! frames
echo   Min track frames : !TP_MIN_FRAMES! frames
echo   --- ACDC filter (step 5) ---
echo   Min track frames : !MIN_FRAMES! (0=skip)
echo   --- Acquisition ---
echo   Pixel size (um)  : !PX!
echo   Z-step (um)      : !ZSTEP!
echo   Frame dt (s)     : !DT!
echo ======================================
echo.

set "CONFIRM=y"
set /p CONFIRM=Start pipeline? (y/n)
if "!CONFIRM!"=="" set "CONFIRM=y"
if /I not "!CONFIRM!"=="y" (
  echo Aborted.
  exit /b 0
)

REM ── Activate ACDC env for steps 1-3 ─────────────────────────────────────────
call "!CONDA_BAT!" activate !ACDC_ENV!
if errorlevel 1 (
  echo [ERROR] Could not activate conda env: !ACDC_ENV!
  pause
  exit /b 1
)
echo Activated: !ACDC_ENV!
echo.

REM ============================================================
REM  STEP 1 - Rename raw export
REM ============================================================
if !START_STEP! GTR 1 goto STEP2
echo.
echo [1/6] Renaming raw export to ACDC input structure...
echo       Source : !ROOT!!PHASE!_DATA
echo       Output : !ROOT!ACDC_IN_!PHASE!
echo.
if not exist "!ROOT!!PHASE!_DATA" (
  echo [ERROR] Data folder not found: !ROOT!!PHASE!_DATA
  pause
  exit /b 1
)
python scripts\rename_to_acdc_input.py !PHASE!
if errorlevel 1 ( echo [ERROR] rename_to_acdc_input.py failed. & pause & exit /b 1 )

:STEP2
REM ============================================================
REM  STEP 2 - Build segmentation_input
REM ============================================================
if !START_STEP! GTR 2 goto STEP3
echo.
echo [2/6] Building segmentation_input + position_map.csv ...
echo.
if not exist "!ROOT!ACDC_IN_!PHASE!" (
  echo [ERROR] ACDC_IN_!PHASE! not found. Run from step 1 first.
  pause
  exit /b 1
)
mkdir segmentation_input 2>nul
python scripts\build_segmentation_input.py !PHASE! segmentation_input
if errorlevel 1 ( echo [ERROR] build_segmentation_input.py failed. & pause & exit /b 1 )

:STEP3
REM ============================================================
REM  STEP 3 - Segmentation + tracking (Cell-ACDC)
REM ============================================================
if !START_STEP! GTR 3 goto STEP4
echo.
echo [3/6] Running segmentation + tracking...
echo.
if /I "!APPLY_PATCH!"=="y" (
  if exist tools\patch_trackpy_cli.py ( python tools\patch_trackpy_cli.py )
)
if /I "!MODE!"=="all" (
  python scripts\run_segm_track_from_index.py --exp_root segmentation_input --ini_dir inis --all
) else (
  python scripts\run_segm_track_from_index.py --exp_root segmentation_input --ini_dir inis --select !SELECT!
)
if errorlevel 1 ( echo [ERROR] run_segm_track_from_index.py failed. & pause & exit /b 1 )

:STEP4
REM ============================================================
REM  STEP 4 - TrackPy gap-closing linking
REM ============================================================
REM  WHY THIS STEP:
REM    Cell-ACDC may split one physical cell track into multiple short tracks
REM    when the segmentation misses the cell for 1-2 frames.  Imaris bridges
REM    these gaps and reports ONE long track.  This step re-links ACDC detections
REM    using TrackPy with gap-closing (memory=!TP_MEMORY!) so the exported
REM    Parent IDs match Imaris track extents.
REM
REM  OUTPUT:
REM    segmentation_input/Position_X/Images/<field>_<loc>_trackpy_tracks.csv
REM    Auto-detected by export_imaris_like_from_pipeline.py in step 6.
REM ============================================================
if !START_STEP! GTR 4 goto STEP5
echo.
echo [4/6] TrackPy gap-closing linking...
echo       search_range=!TP_RANGE!px  memory=!TP_MEMORY!  min_frames=!TP_MIN_FRAMES!
echo       (bridges up to !TP_MEMORY! consecutive missing frames per track)
echo.
call "!CONDA_BAT!" activate !TRACKPY_ENV!
if errorlevel 1 ( echo [ERROR] Could not activate conda env: !TRACKPY_ENV! & pause & exit /b 1 )
if /I "!MODE!"=="all" (
  python scripts\run_trackpy_linking.py ^
    --exp_root segmentation_input ^
    --all ^
    --search-range-px !TP_RANGE! ^
    --memory !TP_MEMORY! ^
    --min-frames !TP_MIN_FRAMES!
) else (
  python scripts\run_trackpy_linking.py ^
    --exp_root segmentation_input ^
    --select !SELECT! ^
    --search-range-px !TP_RANGE! ^
    --memory !TP_MEMORY! ^
    --min-frames !TP_MIN_FRAMES!
)
if errorlevel 1 ( echo [ERROR] run_trackpy_linking.py failed. & pause & exit /b 1 )
REM Switch back to ACDC env for step 5
call "!CONDA_BAT!" activate !ACDC_ENV!

:STEP5
REM ============================================================
REM  STEP 5 - Filter short tracks (optional)
REM ============================================================
if !START_STEP! GTR 5 goto STEP6
if "!MIN_FRAMES!"=="0" (
  echo.
  echo [5/6] Skipping track-length filter (MIN_FRAMES=0)
) else (
  echo.
  echo [5/6] Filtering tracks shorter than !MIN_FRAMES! frames...
  echo.
  if /I "!MODE!"=="all" (
    python scripts\filter_acdc_output_by_tracklen.py --exp_root segmentation_input --all --min_frames !MIN_FRAMES!
  ) else (
    python scripts\filter_acdc_output_by_tracklen.py --exp_root segmentation_input --select !SELECT! --min_frames !MIN_FRAMES!
  )
  if errorlevel 1 ( echo [WARN] Filtering failed - continuing without filtering. )
)

:STEP6
REM ============================================================
REM  STEP 6 - Export Imaris-like Excel
REM ============================================================
REM  NOTE: --trackpy-csv is NOT passed here because run_trackpy_linking.py
REM  already saved the CSVs inside each Position_X/Images/ folder.
REM  export_imaris_like_from_pipeline.py auto-detects them there.
REM ============================================================
echo.
echo [6/6] Exporting Imaris-like Excel workbooks...
echo       (TrackPy CSVs auto-detected from Position_X/Images/)
echo.
mkdir pipeline_output 2>nul
call "!CONDA_BAT!" activate !EXPORT_ENV!
if errorlevel 1 ( echo [ERROR] Could not activate conda env: !EXPORT_ENV! & pause & exit /b 1 )
if /I "!MODE!"=="all" (
  python scripts\export_imaris_like_from_pipeline.py ^
    --exp_root segmentation_input ^
    --out_dir pipeline_output ^
    --converter scripts\acdc_npz_tif_to_imaris_like.py ^
    --all ^
    --pixel-size-um !PX! ^
    --z-step-um !ZSTEP! ^
    --frame-interval-s !DT!
) else (
  python scripts\export_imaris_like_from_pipeline.py ^
    --exp_root segmentation_input ^
    --out_dir pipeline_output ^
    --converter scripts\acdc_npz_tif_to_imaris_like.py ^
    --select !SELECT! ^
    --pixel-size-um !PX! ^
    --z-step-um !ZSTEP! ^
    --frame-interval-s !DT!
)
if errorlevel 1 ( echo [ERROR] export_imaris_like_from_pipeline.py failed. & pause & exit /b 1 )

echo.
echo ========================================
echo   DONE - Pipeline completed successfully
echo   Output: !ROOT!pipeline_output\
echo ========================================
echo.
pause
