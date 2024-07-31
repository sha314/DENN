# The installation is done using a package manager named 'conda'
Install Anaconda/Miniconda for mac and then use the following command to create the environemnt from .yml file

$ conda env create -f environment.yml

if you want to update an exsisting environment

$ conda env update --name myenv --file local.yml --prune

