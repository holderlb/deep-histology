
:: Usage: run.bat <image> <tissue_type>
:: System: Windows
::
:: Script to automatically run each pathology classifier on an input image given the tissue type.
:: Creates a highlighted image for each pathology corresponding to a tissue type.
:: Note: Append the character "|" on each python line to run in parallel (requires lots of memory).
::
:: arg1: image location
:: arg2: tissue type     (testis, prostate, kidney)


if "%2" == "testis" (
    python .\DTP\dtp.py %1 testis Atrophy 256 
    python .\DTP\dtp.py %1 testis Maturation_Arrest 256 
    python .\DTP\dtp.py %1 testis Vacuole 256 
    )

if "%2" == "prostate" (
    python .\DTP\dtp.py %1 prostate Atrophy 256 
    python .\DTP\dtp.py %1 prostate Collapsed_Prost 256 
    python .\DTP\dtp.py %1 prostate Hyperplasia 256 
    python .\DTP\dtp.py %1 prostate Vacuole 256 
    )

if "%2" == "kidney" (
    python .\DTP\dtp.py %1 kidney Cyst 256 
    python .\DTP\dtp.py %1 kidney Reduced_Glomeruli 256 
    python .\DTP\dtp.py %1 kidney "Thickened_Bowman's_Capsule" 256 
    )