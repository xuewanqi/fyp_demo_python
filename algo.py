import random

class Algo:
    def __init__(self,data):
        self.data = data

    def move(self,thief,police):
        p = []
        for pol in police:
            p.append(random.choice(self.data[pol]))
        return p

