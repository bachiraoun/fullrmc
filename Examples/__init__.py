"""
.. py:function:: get_examples_path()

    Get distribution examples directory path in the standard distribution.
    
    :Returns:
        #. path (string): directory path in a form of a string.

        
.. py:function:: find_example(name)

    Return example's path given its name.
    
    :Parameters:
        #. name (string): The name of the example.
        
    :Returns:
        #. path (None, string): The path of the example if found. 
           None if example is not found.
        
        
.. py:function:: get_available_examples()

    Get all available examples as delivered in the standard distribution.
    
    :Returns:
        #. examples (list): List of examples paths.

        
.. py:function:: print_available_examples() 

    Print all available examples path in the standard distribution.       
"""

# standard distribution imports
import os

def get_examples_path():
    path, _ = os.path.split(os.path.abspath(__file__))
    return path  

def find_example(name):
    if not isinstance(name, basestring):
        return None
    name = str(name)
    if not len(name):
        return None
    path = get_examples_path()
    exPath = os.path.join(path, name)
    if os.path.isdir(exPath):
        return exPath
    else:
        return None
        
def get_available_examples():
    path = get_examples_path()
    # create examples list
    examples = []
    for d in os.listdir( path ):
        d = os.path.join(path, d)
        if os.path.isdir(d): 
            examples.append(d)
    return examples

def print_available_examples():
    for e in get_available_examples():
        print e


    