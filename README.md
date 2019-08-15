# Part-of-Speech Embeddings for Portuguese (STIL 2019)

Work presented at STIL 2019. This repository contains
- dataset files: `data` directory
- scripts for dataset generation: `scripts` directory
- files used for model definition + training: `models`, `pos_tagger` and `postagger.py`
- logs of some training experiments (test accuracies too): `runs` directory
- pretrained models: `runs` directory
- pdf of the paper: `STIL2019.pdf`

### Requirements
- `python v3.6.3`  
- `pytorch v1.2`: https://pytorch.org/
- `tqdm`: https://github.com/tqdm/tqdm
- Very basic knowledge of `python` is needed in order to fill the `pos_tagger/parameters.py` file.

## Dataset generation
### Links for downloading the datasets
  - MacMorpho: http://nilc.icmc.usp.br/macmorpho/
  - GSD: https://github.com/UniversalDependencies/UD_Portuguese-GSD 
  - Bosque-UD: https://github.com/UniversalDependencies/UD_Portuguese-Bosque 
  - Bosque-LT: https://www.linguateca.pt/Floresta/ficheiros/Bosque_CP_8.0.ad.txt (at https://www.linguateca.pt/Floresta/corpus.html#bosque)

### Scripts for dataset generation
##### `ad2mm.py`
 Script for extracting sentences and their POS tags from a file with `ad` formatting and generating a new file with the extracted samples, following the Mac-Morpho formatting. In order to execute the script, run
 ```
 python ad2mm.py PATH_TO_AD_FILE PATH_TO_NEW_FILE
 ```
##### `conllu2mm.py`
 Script for extracting sentences and their POS tags from a file with `conllu` formatting and generating a new file with the extracted samples, following the Mac-Morpho formatting. In order to execute the script, run
 ```
  python ad2mm.py PATH_TO_CONLLU_FILE PATH_TO_NEW_FILE
 ```
##### `build_lgtc.py`
 Script for generating the `lgtc` (Linguateca) datasets. Since there can be a huge intersection between different sets (eg. train and test) between Bosque-UD and the generated Linguateca (Bosque-LT) splits, a more cautious split is needed. By generating the files with this script, there will only be intersections between the same sets (train-train, dev-dev, test-test).
 The parameters need to be hardcoded:
 - `DATA_PATH`: path to directory with input files
 - `UD_TRAIN_FILE`: Bosque-UD train file
 - `UD_DEV_FILE`: Bosque-UD dev file
 - `UD_TEST_FILE`: Bosque-UD test file
 - `FILE_LGTC`: Linguateca file with samples at the Mac-Morpho formatting
 - `DEST_LGTC_TRAIN`: Destination path to the Bosque-LT train file
 - `DEST_LGTC_DEV`: Destination path to the Bosque-LT dev file
 - `DEST_LGTC_TEST`: Destination path to the Bosque-LT test file
 Then run
  ```
  python build_lgtc.py
 ```

## Usage of the POS Tagger
#### `pos_tagger/parameters.py` file
Follow the instructions at the file to fill it.

### Executing
Execute the `main` file
```
python postagger.py
```

### Output
#### Terminal
Only log messages with `rank <= LOG_LEVEL` will be printed on the terminal.
- `rank=0` messages: erros, warnings, train and test output, tqdm
- `rank=1` messages: success messages, descriptive log

#### Log file (`output.txt`)
A log file with all the log messages will be generated.

#### Tagged samples
A file with the samples form the validation set, along with their tags prediction, will be generated for each dataset.
    
## Aditional scripts
##### `intersect.py`
 Script used for checking the intersection of sentences between all the files from the dataset. The path of the files must be hardcoded for the variable `FILES`. To execute the script, run
 ```
 python intersect.py
 ```
 
#### References
If you used our model, please cite our paper:
- LINK  TODO
```
 bibtex  TODO
```
