import numpy as np


test_idxs=np.array([31,41,17,20,33,40,26,43,19,23,25,47,38,7,34,29,46,13,15,39,10,2,9,3
,37,30,11,44,14,1,32,5,21,48,42,49,8,6,35,18,0,28,27,22,16])
test_err=np.array([False,False,True,True,False,True,False,False,True,False,False,True
,False,False,True,False,False,True,False,False,False,True,True,True
,False,True,False,True,False,True,False,False,False,False,False,True
,False,True,False,False,False,False,False,False,False])

print(test_idxs[test_err])