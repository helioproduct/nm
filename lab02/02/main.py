from newton import Newton
from iterations import Iterations


if __name__ == "__main__":


    '''
    
    
    '''

    eps = 1e-5

    a = [-1, -2]
    b = [2, 2]

    print("Iteration method:", Iterations(a, b, eps))
    print("Newton method:", Newton(a, b, eps))
