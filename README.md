
# CryoGAN 

Methods to reconstruct 3D structure from cryoEM data using adversarial learning.


# Download

Run following to install the dependencies:

    conda env create --file environment-mac.yml (on mac)
    conda env create --file environment.yml (on non-mac)


To start a reconstruction, run the following. 
    
    python main_cryogan.py --config_path configs/betagal_simulated.yaml

The parameters of the reconstruction can be edited in the input config file.
For simualted the SNR of the ground truth data can also be given with:
    
    python main_cryogan.py --config_path configs/betagal_simulated.yaml --snr_val -5

