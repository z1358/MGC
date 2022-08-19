import json
import os.path as osp
import numpy as np
from collections import defaultdict, OrderedDict
from tensorboardX import SummaryWriter
import os
import sys
class Mylog(object):
    def __init__(self, save_path='', filename="log.txt"):
        self.terminal = sys.stdout
        save_path = save_path + '/log.txt'
        print(save_path)
        self.log = open(save_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

_log_path = None

def set_log_path(path, filename='log.txt'):
    global _log_path
    _log_path = path
    # fobj = open(os.path.join(_log_path, filename), 'w')
    # fobj.close()

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)

class Logger(object):
    def __init__(self, args, log_dir, **kwargs):
        self.logger_path = osp.join(log_dir, 'scalars.json')
        self.tb_logger = SummaryWriter(
                            logdir=osp.join(log_dir, 'tflogger'),
                            **kwargs,
                        )
        self.log_config(vars(args))

        self.scalars = defaultdict(OrderedDict)

    def add_scalar(self, key, value, counter):
        assert self.scalars[key].get(counter, None) is None, 'counter should be distinct'
        self.scalars[key][counter] = value
        self.tb_logger.add_scalar(key, value, counter)

    def log_config(self, variant_data):
        config_filepath = osp.join(osp.dirname(self.logger_path), 'configs.json')
        with open(config_filepath, "w") as fd:
            json.dump(variant_data, fd, indent=2, sort_keys=True, cls=ConfigEncoder)

    def dump(self):
        with open(self.logger_path, 'w') as fd:
            json.dump(self.scalars, fd, indent=2)