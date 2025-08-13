# BeHance-art-valuation

This repository contains the source code and dataset used in the study:

- Kim et al. **Combined influence of artwork and artist features in predicting user generated art valuation**.

**Authors**:  
Seunghwan Kim<sup>1+</sup>, Soomin Lee<sup>2+</sup>, Byunghwee Lee<sup>3*</sup>, Wonjae Lee<sup>1*</sup>  

<sup>1</sup> <sub>Graduate School of Culture Technology, Korea Advanced Institute of Science and Technology, Daejeon, Republic of Korea</sub>  
<sup>2</sup> <sub>Department of Computer Science and Engineering, Chungnam National University, Daejeon, Republic of Korea</sub>  
<sup>3</sup> <sub>Center for Complex Networks and Systems Research, Luddy School of Informatics, Computing, and Engineering, Indiana University, Bloomington, IN, USA</sub>  

<sup>+</sup> <sub>S.K. and S.L. contributed equally to this work.</sub>  <sup>*</sup> <sub>Corresponding author.</sub>  



## Introduction
 - This repository provides the source code necessary for reproducing the results presented in the paper.
 - The core implementation of experimental results is found in **`src/Main_results.ipynb`**.
 - The original data was collected from the Behance platform; however, due to copyright concerns, only the preprocessed data required for modeling is provided.



## Installation

Installation using [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html):

```bash
git clone https://github.com/sxxmin/BeHance-art-valuation.git
cd Belief-Embedding
conda create -y --name behance python=3.6.9
conda activate behance
pip install -r requirements.txt
```



## System requirements
* **Software dependencies**:
  * Supported platforms: MacOS and Ubuntu (with Python 3.6.9)
  * See requirements.txt for a complete list of necessary libraries.

* **Tested Versions** 
   * The following libraries have been tested with Python 3.6.9:
     * `statsmodels = 0.12.2`
     * `xgboost = 1.5.2`
     * `interpret = 0.6.2`
     * See `requirements.txt` for full list of necessary libraries. 



* **Quickstart**

  * In terminal, run:
  ```bash
  jupyter notebook
  ```
  * Select the `behance` kernel in Jupyter Notebook.
  * Before opening 'Main_result.ipynb', you should first run the training script in the terminal:
  ```bash
  python Trainer.py --model EBM --target view --engaging_group Artist --using_c_variable True
  ```
  - This command runs the Explainable Boosting Machine (`EBM`) model to predict the `view` target variable as valuation using only the `Artist` group and including the control variable.  
  - For detailed descriptions of all available arguments, please refer to the **Command-line Arguments** section below.



* **Hardware requirements**
  * A CPU is sufficient for training.



## Command-line Arguments

The script provides several command-line options to configure the model training and evaluation:

* `--model`  
  - Type: `str`  
  - Default: `'EBM'`  
  - Description: Specifies the model to use for prediction. Options include:  
    - `LR` : Linear Regression  
    - `NB` : Negative Binominal  
    - `XGB` : XGBoost  
    - `EBM` : Explainable Boosting Machine

* `--target`  
  - Type: `str`  
  - Default: `'view'`  
  - Description: Defines the target variable to predict. Options include:  
    - `appreciation` : Measures user appreciation  
    - `view` : Number of views  

* `--engaging_group`  
  - Type: `str`  
  - Default: `'All'`  
  - Description: Specifies which group to include in the analysis. Options:  
    - `Artist` : Only artist factors as feature  
    - `Artwork` : Only artwork factors as feature  
    - `All` : Include both artist and artwork factors as feature

* `--using_c_variable`  
  - Type: `bool` (`str2bool`)  
  - Default: `True`  
  - Description: Determines whether to include the control variable `C` in the model.

* `--window_opt`  
  - Type: `str`  
  - Default: `''`  
  - Description: Specifies the size of the time window for feature aggregation. Options:  
    - `'3'` : 3-unit window  
    - `'6'` : 6-unit window  
    - `''` (empty string) : Default window of 9 units
