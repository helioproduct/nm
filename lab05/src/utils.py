def max_abs_error(A, B):
    
    # print(A.shape, B.shape)
    # assert A.shape == B.shape
    # return abs(A - B).max()

    assert len(A) == len(B)

    # yes 
    # diff = [abs(A[i] - B[i]) if B[i] != None and A[i] != None else 0 for i in range(len(A))]

    diff = [abs(A[i] - B[i]) for i in range(len(A))]

    return max(diff)


     