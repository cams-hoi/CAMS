from .cams_cvae import *

def model_entry(config):
    return globals()[config['type']](config['kwargs'])
cams_cvae
