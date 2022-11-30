class File():
    def __init__(self, path):
        with open(path) as file:
            self.lines = file.readlines()
