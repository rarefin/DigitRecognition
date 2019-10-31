# DigitRecognition
Digit Recognition from Natural Images

1) To pre-process the data, follow the steps:
    1) Go to the "src" directory
    2) Then run the script in "preprocess" directory: 
       python preprocess/convert_to_lmdb.py --data_dir "Direcotory of train, test and extra data"


2) To train the model:
    1) Change the paths and parameter in "config.json"
    2) run the "train.py" script:
        python train.py