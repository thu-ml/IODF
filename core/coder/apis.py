from core.coder.fast_ans.build import fast_ans
from .cdf import DiscretizedLogistic, MixtureDiscretizedLogistic

mass_bits = 24

def prepare_pmf_cdf(j, pj):
    if len(pj) == 3:
        # mixture 
        l = MixtureDiscretizedLogistic(*pj, mass_bits, 256)
    elif len(pj) == 2:
        l = DiscretizedLogistic(*pj, mass_bits, 256)
    else:
        raise RuntimeError
    pmf, cdf = l.pmf(j)

    return pmf.cpu().float().numpy().ravel(), cdf.cpu().float().numpy().ravel()

def encode(pz, z, pys, ys):
    # These tensors should be on GPU to speed up computation
    batchsize = z.size(0)

    ans_coder = fast_ans.ANS(mass_bits, batchsize)

    ys += (z, )
    pys += (pz, )

    for y, py in zip(ys, pys):

        pmf, cdf = prepare_pmf_cdf(y, py)

        ans_coder.encode_with_pmf_cdf(pmf, cdf)
    
    return ans_coder

def decode(model, ans_coder):

    ans_coder, z = model.prior.decode(ans_coder)

    x = model.flow.decode(ans_coder, z)

    x = model.normalize(x, reverse=True)

    return x 