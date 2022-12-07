sudo apt-get --assume-yes update
sudo apt-get --assume-yes install git man wget curl unzip zip fish htop tmux


echo "fish" >> .bashrc; fish

sudo apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*
    
sudo apt-get --assume-yes install ffmpeg libsm6 libxext6  -y

sudo apt-get update

##### If no conda
wget 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'
bash Miniconda3-latest-Linux-x86_64.sh


conda install -c conda-forge fish
conda create --name siimcovid python=3.9.2
conda activate siimcovid

# According to CUDA version (V100 on Google cloud, CUDA=11.6)
conda install libgcc gmp
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

pip3 install certifi
pip3 install -U albumentations
pip3 install madgrad opencv-python


# download dataset
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get install apt-transport-https ca-certificates gnupg 
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update && sudo apt-get install google-cloud-sdk 
apt-get install google-cloud-sdk-app-engine-python --assume-yes 
gcloud init 
gsutil cp gs://siimcovid19_jpg/siim_512_jpg.zip .

# gcloud auth login

## Clone newest repo
git clone https://github.com/psui3905/COMS4995-Project.git