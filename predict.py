import torch
import tqdm
import os
import argparse
import yaml
from inertial_poser.DynaIP.DynaIP import DynaIP
from inertial_poser.PNP.PNP import PNP
from exp.famework import run_pipeline
from submodule.detector.detector import BasicNormDetector, OurDetector
from submodule.corrector.corrector import CorrectorV2

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="exp/PNP/eskf9+det.yaml", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    
    results_dir = "results"
    measurement_dir = "datasets/MagIMU/raw_measurements"
    
    # load inertial poser
    inertial_poser_name = config.get('inertial_poser')
    if inertial_poser_name == 'PNP':
        net = PNP().eval().cuda()
    elif inertial_poser_name == 'DynaIP':
        net = DynaIP().eval().cuda()

    # load detector
    use_detector = config.get('use_detector')
    detector = OurDetector() if use_detector else BasicNormDetector()

    # load corrector
    use_corrector = config.get('use_corrector')
    corrector = CorrectorV2().eval().cuda() if use_corrector else None

    method = "eskf9"
    if use_detector: method += "+det"
    if use_corrector: method += "+cor"
    
    with torch.no_grad():
        save_dir = os.path.join(results_dir, inertial_poser_name, method)
        os.makedirs(os.path.join(results_dir, inertial_poser_name), exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        import shutil
        shutil.copy2(args.config, os.path.join(save_dir, 'config.yaml'))

        for f in tqdm.tqdm(os.listdir(measurement_dir), ncols=50):
            poses, trans = run_pipeline(
                os.path.join(measurement_dir, f),
                net,
                detector=detector,
                corrector=corrector
            )
            torch.save((poses, trans), os.path.join(save_dir, f))
