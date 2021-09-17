
:: Usage: run.bat <image> <tissue_type>
:: System: Windows
::
:: Script to automatically run each pathology classifier on an input image given the tissue type.
:: Creates a highlighted image for each pathology corresponding to a tissue type.
:: Note: Append the character "|" on each python line to run in parallel (requires lots of memory).
::
:: arg1: image location
:: arg2: tissue type     (testis, prostate, kidney)
:: arg3: optional argument for the highlighting type. If no argument is
::       given, then boxes will be drawn around tiles which have been marked positive for disease.
::       All possible arguments are the heatmap options described below.
::       For heatmap options, set the argument to any of the color map names from the
::       following link:
::       <https://matplotlib.org/stable/tutorials/colors/colormaps.html>
::       Recommendations: plasma, gray, cividis
::
:: Authors: Colin Greeley and Larry Holder, Washington State University


if "%2" == "testis" (
    python .\DTP\dtp.py %1 testis Atrophy 256 %3 
    python .\DTP\dtp.py %1 testis Maturation_Arrest 256 %3 
    python .\DTP\dtp.py %1 testis Vacuole 256 %3 
    )

if "%2" == "prostate" (
    python .\DTP\dtp.py %1 prostate Atrophy 256 %3 
    python .\DTP\dtp.py %1 prostate Collapsed_Prost 256 %3 
    python .\DTP\dtp.py %1 prostate Hyperplasia 256 %3 
    python .\DTP\dtp.py %1 prostate Vacuole 256 %3 
    )

if "%2" == "kidney" (
    python .\DTP\dtp.py %1 kidney Cyst 256 %3 
    python .\DTP\dtp.py %1 kidney Reduced_Glomeruli 256 %3 
    python .\DTP\dtp.py %1 kidney "Thickened_Bowman's_Capsule" 256 %3 
    )