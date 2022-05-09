# TFM
Research on how to build a GNN for recommendation

# Environment

conda create -n my-torch python=3.8 -y

conda activate my-torch

conda install pip

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

pip install markupsafe==2.0.1

~/anaconda3/envs/my-torch/bin/pip install torch-geometric==2.0.3 (da errores pero funciona)

~/anaconda3/envs/my-torch/bin/pip install torch-sparse

~/anaconda3/envs/my-torch/bin/pip install torch-scatter

conda install -c conda-forge huggingface_hub==0.2.1

conda install -c conda-forge sentence-transformers