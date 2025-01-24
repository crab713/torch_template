import os

class AvgMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.reset()
        self.val = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))