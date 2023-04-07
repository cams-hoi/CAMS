import os
import yaml

class Config(object):

    def __init__(self, config_file):

        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        config['log_dir'] = os.path.dirname(config_file)
        self.config = config
