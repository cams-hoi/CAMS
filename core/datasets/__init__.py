from .hoi4d_dataset import HOI4D

def dataset_entry(config):
    return globals()[config['data']['type']](config)
