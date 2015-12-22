# standard distribution imports
import os


def get_available_examples():
    path, fname = os.path.split(os.path.abspath(__file__))
    examples    = []
    for d in os.listdir( path ):
        d = os.path.join(path, d)
        if os.path.isdir(d): 
            examples.append(d)
    return examples
    