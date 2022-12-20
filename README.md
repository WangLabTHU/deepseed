# DeepSEED: Deep flanking sequences engineering for efficient promoter design
## 
The code for official implementation of "Deep flanking sequences engineering for efficient promoter design"
This codebase provides:
1. The conditional generative adversarial networks (GANs) that generate promoter sequences when giving conditional motif sequences. (the folder "Generator")
2. The predictor that could evaluate promoter activities based on densenet-lstm model. (the folder "Predictor")
3. The optimizer that could design promoter sequences with high activities when giving conditional motif sequences. (the folder "Optimizer")

## Contents
1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Bibtex](#bibtex)
4. 
## Introduction <a name="introduction">
We introduced an AI-aided promoter design framework, DeepSEED, that employs both experts’ knowledge and deep learning together to efficiently design synthetic promoters of diverse desirable functions. DeepSEED incorporates the user-defined cis-regulatory sequences as ‘seed’ and generates flanking sequences that match the ‘seed’. We showed that DeepSEED could automatically capture a variety of weak patterns like k-mer frequencies and DNA shape features from active promoters in the training set, and efficiently optimize the flanking sequences in synthetic promoters to better match these properties.

<div align='center'><img align="middle" src="imgs/interface.png" width="70%" /><br></div>

We validated the effectiveness of this framework for diverse synthetic promoter design tasks in both prokaryotic and eukaryotic cells. DeepSEED successfully designed constitutive, IPTG-inducible, and doxycycline(Dox)-inducible promoters with significant performance improvements, suggesting DeepSEED as an efficient AI-aided flanking sequence optimization approach for promoter design that may greatly benefit synthetic biology applications.




## Environment Setup <a name="environment-setup">
**Env Requirements:** 
- MAC OS, Linux or Windows.
- Python 3.5+.
- PyTorch 1.7.1
- CUDA 1.10.0  if you need train deep learning model with gpu.

**Steps of using DeepSEED:** 

0. Install Python ref to [Download Python](https://www.python.org/downloads/)

1. Install DeepSEED in virtualenv to keep your environment clean:

    ##### on macOS/Linux
    ```
    pip install virtualenv  
    # or 
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple virtualenv 

    virtualenv --python=python3 DeepSEED
    cd DeepSEED
    source ./bin/activate
    ```
    Optional: After use, shutdown virtual environment with
    ```
    deactivate
    ```

    ##### on Windows (open cmd with administrator's permission)
    ```
    pip install virtualenv  
    # or 
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple virtualenv 
    virtualenv --python=python3 DeepSEED
    cd DeepSEED
    .\Scripts\activate.bat
    ```
    Optional: Shutdown virtual environment with
    ```
    .\Scripts\deactivate.bat
    ```

2. Install [Git](https://git-scm.com/), this step is optional if you does not install CellTracker by git clone. Clone the source codes with git. 
    ```
    git clone https://github.com/WangLabTHU/deepseed.git
    ```
3. or, download the source codes and extract files and put it in the virtual environment directory you defined. 

4. after 2/3, the directory of CellTracker should have the following structure:
    
    ```
    DeepSEED
        deepseed
            |-------
            |
            |-------data
                    |...
            |-------Generator
                    |...
            |-------Predictor
                    |...
            |-------Optimizer
                    |...
            |-------utlis
                    |...
            |-------design_demo
            |-------
            |...
        |...
    ```

5. After the extraction, download all dependencies with the following commend.
    ```
    cd deepseed
    pip install -r requirements.txt
    ```
    To boost download speeds, you could setup pip mirror such as
    ```
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt  
    # or
    pip install -i https://mirrors.aliyun.com/pypi/simple  -r requirements.txt
    ```
    It is recommended to use the official website to install pytorch.
    ```
    # CPU version
    pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
    # GPU version
    pip install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/torch_stable.html
    ```

6. Run deepseed with python and enjoy it with following steps:

## Using DeepSEED model to design promoter sequence data <a name="environment-setup">
We take design of 3-lacO IPTG-inducible promoters in *E. coli* as an example, to illustrate how to train the DeepSEED model and design the promoter sequences

### 1. Training the conditional GANs
    ```
    cd Generator
    python cGAN_training.py
    ```
    
### 2. Training the predictor
    ```
    cd Predictor
    python predictor_training.py
    ```

### 3. Design the promoter sequences with optimizer
    ```
    cd Optimizer
    python deepseed_optimizer.py
    ```
### 4. Check the synthetic promoter sequences!
    ```
    vi ./results/ecoli_3_laco.txt
    ```
## Bibtex<a name="bibtex">
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.

```
@article{,
  title={Deep flanking sequences engineering for efficient promoter design},
  author={Pengcheng Zhang, Haochen Wang, Hanwen Xu, Lei Wei, Zhirui Hu, Xiaowo Wang
},
  journal={},
  year={2022
}
```

## License
For academic use, this project is licensed under the GNU Lesser General Public License v3.0 - see the LICENSE file for details. For commercial use, please contact the authors. 


