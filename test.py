# test.py file

def print_tree(levels):
    for i in range(levels):
        print(' ' * (levels - i - 1) + '*' * (2 * i + 1))
        
        
        
print_tree(5)