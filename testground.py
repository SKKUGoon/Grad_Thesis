"""
voting classifer = c1 c2 c3 c4

boosting - iteratively weight adjusting
1, 2, 3, 4

iteration - what classifer?
    c1 c2 c3 c4 ... c100
    c1
    iteration 100
    c10000 - predict error
    nobody use this

week classifer
1,2,3,4,5
c1 c1 c1 c1 c1

1) c1 iterate c1 c2 assign


Novel way
stochastically pick

c1 c2 c3
1/3 1/3 1/3 = pdist

c2 -> good predict -> rate of pick inc


classifier weight = (1/2) log ((1 - err_m) / err_m)
sign<classifer weight , classifer prediction>


probwc := probwc * exp(v1=0.01 * classifer weight) -> v1 should be iteratively controlled

pdist random choice

ensemble
error orthogonal -> prediction value
error diverse -> 1) use different model ->
                 2) data set - bootstrapping

"""