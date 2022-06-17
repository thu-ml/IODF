from core.coder.fast_ans.build_trt import fast_ans
mass_bits = 24

def encode(pmfs, cdfs, bs):
    # pmfs: a list of ndarrays, shape: [batchsize, num_elements]

    ans_coder = fast_ans.ANS(mass_bits, bs)

    for pmf, cdf in zip(pmfs, cdfs):

        ans_coder.encode_with_pmf_cdf(pmf, cdf)
    
    return ans_coder

def decode(model, ans_coder):

    ans_coder, z = model.prior.decode(ans_coder)

    x = model.flow.decode(ans_coder, z)

    x = model.normalize(x, reverse=True)

    return x 