@REM This script will start the program. Watch at console carefully
@echo OFF
@REM Run this script to capture/collect the dataset.
python .\dataset.py
@REM pause 
echo "User data collection complete. Starting training."
pause
@REM Run the below script to train the dataset.
python .\trainer.py
@REM pause 
echo "Training was completed. Continue with the program."
pause
@REM Run the below script to recognize and open/close the door.
python .\recognizer.py