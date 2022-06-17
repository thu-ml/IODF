import torch

def discretized_logistic_cdf(x, mean, logscale):
    scale = torch.exp(logscale)

    cdf = torch.sigmoid((x - mean) / scale)

    return cdf

def mixture_discretized_logistic_cdf(x, mean, logscale, pi):
    scale = torch.exp(logscale)

    x = x[..., None]

    cdfs = torch.sigmoid((x - mean) / scale)

    cdf = torch.sum(cdfs * pi, dim=-1)

    return cdf

class CdfBins():

    def __init__(self, mean:torch.Tensor, inverse_bin_width:int) -> None:
        self.mean = (mean * inverse_bin_width).round()
        self.inverse_bin_width = inverse_bin_width
        self.n_bins = 1 << 12
        self.low = self.mean - (self.n_bins >> 1)
        self.high = self.mean + (self.n_bins >> 1)
        self.num_syms = self.high - self.low
    
    def locate(self, z:torch.Tensor) -> torch.Tensor:
        Z = (z * self.inverse_bin_width).round()
        if (Z < self.low).any() or (Z > self.high).any():
            raise RuntimeError
        return Z - self.low

class DiscretizedLogistic():

    def __init__(self, mean:torch.Tensor, logscale:torch.Tensor, mass_bits:int, inverse_bin_width:int) -> None:
        self.mean = mean
        self.logscale = logscale
        self.total_mass = 1 << mass_bits
        self.inverse_bin_width = inverse_bin_width
        self.cdf_bins = CdfBins(mean, inverse_bin_width)
    
    def pmf(self, z:torch.Tensor) -> torch.Tensor:
        r_prob = self.cdf(z + 1 / self.inverse_bin_width)
        l_prob = self.cdf(z)
        if (r_prob < l_prob).any():
            raise RuntimeError
        return r_prob - l_prob, l_prob

    def cdf(self, z:torch.Tensor) -> torch.Tensor:
        c_prob = discretized_logistic_cdf(z - 0.5 / self.inverse_bin_width, self.mean, self.logscale)
        loc = self.cdf_bins.locate(z)
        return c_prob * (self.total_mass - self.cdf_bins.num_syms) + loc

class MixtureDiscretizedLogistic():

    def __init__(self, mean:torch.Tensor, logscale:torch.Tensor, pi_mix:torch.Tensor, mass_bits:int, inverse_bin_width:int) -> None:
        self.mean = mean
        self.n_mixtures = mean.shape[-1]
        self.logscale = logscale
        self.pi_mix = pi_mix
        self.total_mass = 1 << mass_bits
        self.inverse_bin_width = inverse_bin_width
        self.cdf_bins = CdfBins(mean[:,:,:,:,self.n_mixtures // 2], inverse_bin_width)
    
    def pmf(self, z:torch.Tensor) -> torch.Tensor:
        r_prob = self.cdf(z + 1 / self.inverse_bin_width)
        l_prob = self.cdf(z)
        if (r_prob < l_prob).any():
            raise RuntimeError
        return r_prob - l_prob, l_prob

    def cdf(self, z:torch.Tensor) -> torch.Tensor:
        c_prob = mixture_discretized_logistic_cdf(z - 0.5 / self.inverse_bin_width, self.mean, self.logscale, self.pi_mix)
        loc = self.cdf_bins.locate(z)
        return c_prob * (self.total_mass - self.cdf_bins.num_syms) + loc
        