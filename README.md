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
 - The original data was collected from the Behance platform; however, due to copyright concerns, the preprocessed data required for modeling is provided.



## Installation

You can set up this project using **Anaconda Navigator** without directly using the command line.

1. **Open Anaconda Navigator**.  
2. Go to the **Environments** tab and click **Create** to make a new environment named `behance` with **Python 3.6.9**.  
3. Once created, select the `behance` environment and click **Open Terminal**.  
4. In the terminal, run:  
   ```bash
   git clone https://github.com/sxxmin/BeHance-art-valuation.git
   cd BeHance-art-valuation
   pip install -r requirements.txt
   ```
5. To work with the project in a notebook, open **JupyterLab** from Anaconda Navigator (make sure the `behance` environment is selected).
6. You can run the code either:
  **In a Terminal** inside JupyterLab, or
  **In a Notebook** by creating a new Python 3 notebook linked to the `behance` environment.



## System requirements
* **Software dependencies**:
  * Supported platforms: MacOS and Ubuntu (with Python)
  * See requirements.txt for a complete list of necessary libraries.

* **Tested Versions** 
   * The following libraries have been tested with Python 3.6.9:
     * `statsmodels = 0.12.2`
     * `xgboost = 1.5.2`
     * `interpret = 0.6.2`
     * See `requirements.txt` for full list of necessary libraries. 



## Quickstart

You can run the project directly from **JupyterLab** (via Anaconda Navigator) or from the **Terminal** in the `behance` environment.

1. **Open JupyterLab** from Anaconda Navigator (with the `behance` environment selected).  
2. If you want to use the notebook interface, create or open a notebook and set the kernel to **`behance`**.  
3. Before opening `Main_result.ipynb`, you should first run the training script in the **Terminal** (either from Anaconda Navigator or inside JupyterLab):  
   ```bash
   python Trainer.py --model EBM --target view --engaging_group Artist --using_c_variable True
   ```
4. This command runs the **Explainable Boosting Machine (EBM)** model to predict the **view** target variable as valuation using only the **Artist** group and including the control variable.
5. For detailed descriptions of all available arguments, see the **Command-line Arguments section** below.

* **Hardware requirements**
  * A CPU is sufficient for training.



## Command-line Arguments

The script provides several command-line options to configure the model training and evaluation:

* `--model`  
  - Type: `str`  
  - Description: Specifies the model to use for prediction. Options include:  
    - `LR` : Linear Regression  
    - `NB` : Negative Binomial  
    - `XGB` : XGBoost  
    - `EBM` : Explainable Boosting Machine

* `--target`  
  - Type: `str`  
  - Description: Defines the target variable to predict. Options include:  
    - `appreciation` : Measures user appreciation  
    - `view` : Number of views  

* `--engaging_group`  
  - Type: `str`  
  - Description: Specifies which group to include in the analysis. Options:  
    - `Artist` : Only artist factors as feature  
    - `Artwork` : Only artwork factors as feature  
    - `All` : Include both artist and artwork factors as feature

* `--using_c_variable`  
  - Type: `bool` (`str2bool`)  
  - Description: Determines whether to include the control variable in the model.

* `--window_opt`  
  - Type: `str`  
  - Description: Specifies the size of the time window for feature aggregation. Options:  
    - `'3'` : 3-unit window  
    - `'6'` : 6-unit window  
    - `''` (empty string) : Default window of 9 units



## Additional Notes on Data and Model Files

The provided dataset is split into 10 separate files due to size limitations.
To reproduce the results, you must concatenate these files into a single variable table before use.

* Please note:  
  - The pre-trained model weights (`.models` files) are not included in the repository due to size constraints.  
  - These files will be generated and saved after training is completed.  
  - All related procedures, including dataset concatenation and model saving, are implemented in `Trainer.py`.  
  - In `Main_results.ipynb`, the variable `Data_acquisition_to_project_publication_interval` is used as a control variable. This feature does not belong to either the artist or artwork groups at Figure. 2. While it is transparently provided in the reproduction materials, it was intentionally omitted from the published paper as it does not align with the paper’s primary research objectives.


    
