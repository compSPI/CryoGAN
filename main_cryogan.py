import argparse
import os
import sys
import traceback
import yaml
from utils import Dict_to_Obj, update_config
from wrapper_cryogan import CryoganWrapper

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(),"src"))

# TODO: add pbar
def init_config():
    # takes the cfg file with parameters and creates a variable called config with those parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="Specify config file", metavar="FILE")
    parser.add_argument("--snr_val", help="Specify snr value", type=float)

    args = parser.parse_args()

    if not os.path.isfile(args.config_path):
        raise FileNotFoundError("Please provide a valid .cfg file")
    
    with open(args.config_path, "r") as read:
            config_dict = yaml.safe_load(read)

    config = update_config(config_dict)
    config=Dict_to_Obj(config)
    config.config_path=args.config_path
    if args.snr_val is not None:
        config.snr_val=args.snr_val
    config.exp_name=config.exp_name+"_sidelen_"+str(config.side_len)+"_projsize_"+str(config.ProjectionSize)+"_snr_"+str(config.snr_val)

    return config


def main():
    config = init_config()
    cryogan_wrapper = CryoganWrapper(config)
    cryogan_wrapper.run()
    return 0, "Reconstruction Done."


if __name__ == "__main__":
    try:
        retval, status_message = main()
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = "Error: Reconstruction failed."

    print(status_message)
    exit(retval)
