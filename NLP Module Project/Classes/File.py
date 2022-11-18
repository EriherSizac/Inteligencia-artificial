class File(): 
    def __init__(self, path):
        # I like the idea of a helper! however, you should use the "with" construction
        # mentioned here: https://www.programiz.com/python-programming/file-operation
        # so that the file is automatically closed. right now, you are leaving it open! 
        self.lines = open(path).readlines()
