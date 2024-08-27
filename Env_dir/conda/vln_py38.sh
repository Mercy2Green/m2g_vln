
conda activate vln_py38

conda env update -f /home/lujia/VLN_HGT/VLN_HGT/Env_dir/conda/environment_py38.yml

# Missing package
    # grpcio==1.65.0rc2
    # gym==0.21.0
    # habitat==0.1.7
    # habitat-sim==0.1.7

pip install -r requirements_py38.txt

conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

## You can add this to .bashrc
# export CUDACXX="/usr/local/cuda-11.8/bin/nvcc"

## For gym=0.21.0
# pip install setuptools==65.5.0 pip==21
# pip install wheel==0.38.0
# pip install gym==0.21.0

## install habitat
# pip uninstall typing


## Some package for finetune
# pip install omegaconf open_clip_torch supervision torch_scatter grpcio