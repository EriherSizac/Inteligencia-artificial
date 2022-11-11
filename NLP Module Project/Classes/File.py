class File():
    def __init__(self, path):
        self.lines = open(path).readlines()
