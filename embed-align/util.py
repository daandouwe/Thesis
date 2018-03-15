import time

class Timer:
    def __init__(self):
        self.time0 = time.time()

    def elapsed(self):
        time1 = time.time()
        elapsed = time1 - self.time0
        self.time0 = time1
        return elapsed
