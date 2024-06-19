# Mamba-based-MIL


## Usage 
### Setup 

* `pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117`
* `pip install packaging`
* `pip install timm==0.4.12`
* `pip install pytest chardet yacs termcolor`
* `pip install submitit tensorboardX`
* `pip install triton==2.0.0`
* `pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl`
* `pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl`
* `pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs`

### Datasets
    ./                           # parent directory
    ├── ./positives              # positive images of training data
    ├── ./positives_t            # positive images of validating data
    ├── ./negatives              # negative images of training data
    ├── ./negatives_t            # negative images of validating data
    ├── ./RGB                    # all images

### Dataset preprocessing
```
python 1_preprocess.py
```

### Model training
```
python 2_train.py
```

### Model inference
```
python 3_probmap.py
```

### Output evaluation indicators
```
python 4_report.py
```
### Output probmap and heatmap
```
python 4_heatmap.py
```


## Acknowledgement
Thanks [MedMamba](https://github.com/YubiaoYue/MedMamba.git). for serving as building blocks of our codes.
