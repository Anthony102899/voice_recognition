Speech Recognition System
Author: LYU, An

# WARNING:
	The code is only tested runnable in WINDOWS system. The reliability in other systems is not assured.

Description:
	This project realized a simple model that recognize voice through dynamic programming on MFCC dist between
	voices. The project is writen in Python.

Configuration on parameters:
	The restricted regions for accumulated dp matrix is set to 20% and 80 % on each border conditionally. 	

Project Dir:

	-voice-recognition
		-data
			-results: Where place all the .txt code files and image files (Please find all the results here).
			-test data: Where place all the .wav files for testing.
			-training data: Where place all the .wav files for training.
		-src
			-util
				-sound_util.py: Where holds the code for utility functions operating the sound.(Part1-4)
			-speech_recognition.py: Where holds the code for speech recognition system. (Part5) 
	

Dependencies:
	python 3.8
	numpy
	openpyxl
	pandas
	scipy
	matplotlib
	python-speech-features
	
How to run:
1. Install all the dependencies in local python environment.
2. Use either terminal or IDE to run/test the code.
		
	a. For Part5's code (speech_recognition.py):
		(i). Goes (cd) into {project_root_dir}\src
		(ii). Run 

			"{your python directory}\python.exe speech_recognition.py" 

			in terminal to generate the confusion table.
			OR,
			Run

			"{your python directory}\python.exe speech_recognition.py debug" 

			to also generate a .xlsx file which includes the accumulate dp matrix for s8a.wav and
		     s8b.wav (at the same directory of .py file).
		
	b. For Part1-4's code (sound_util.py):
		(i). Change the global variables TRAINING_DIR and TESTING_DIR to the absolute directory in your local PC.
			("{project_root_dir}\data\test data "or "training data")	
		(ii). Goes into {project_root_dir}\src\util in terminal.
		(iii). Run 

			"{your python directory}\python.exe .\sound_util.py {N}"

			There are 5 options for parameter 'N'.
			N=1: Show the plot of voice "s1a.wav".
			N=2: Show the end point detection's plot for "s1a.wav"
			N=3: Show the dft's diagram for "s1a.wav".
			N=4: Show the pre-emphasis's plots for "s1a.wav".
			N=5: Print the LPC-10 digits in the terminal for the signal after pre-emphasis for s1a.wav.
			
	
		