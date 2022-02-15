
# CryoGAN 

Methods to reconstruct 3D structure from cryoEM data using adversarial learning [1].

# Download

Run following to install the dependencies:

    conda env create --file environment-mac.yml (on mac)
    conda env create --file environment.yml (on non-mac)


To start a reconstruction, run the following. 
    
    python main_cryogan.py --config_path configs/betagal_simulated.yaml

The parameters of the reconstruction can be edited in the input config file.
For simualted the SNR of the ground truth data can also be given with:
    
    python main_cryogan.py --config_path configs/betagal_simulated.yaml --snr_val -5
    
[1] Gupta H, McCann MT, Donati L, Unser M. CryoGAN: a new reconstruction paradigm for single-particle cryo-EM via deep adversarial learning. IEEE Transactions on Computational Imaging. 2021 Jul 13;7:759-74. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9483649

