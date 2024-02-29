import sgt_module

str_mat = "0 1 0 0 1 1 0 0 0 0 0 0 1 0 1 0 0 0 0 1 1 0 1 1 0"
anc = sgt_module.compute_anc(str_mat, 25, 8, 0)
print(anc)
