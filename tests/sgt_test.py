import sgt

try:
    str_mat = "0 1 0 0 1 1 0 0 0 0 0 0 1 0 1 0 0 0 0 1 1 0 1 1 0"
    anc = sgt.compute_anc(str_mat, 25, 8, 0)
    print(anc)
except Exception as err:
    print(err)
