import math

class RegenerationStrategy():
    def __init__(self, gamma, sigma, r):
        self.gamma = gamma
        self.sigma = sigma
        self.r = r

    def HealthRegained(self, investment):
        regain = int(self.gamma*((1 - math.exp((-1)*self.sigma*investment))/
                                 (1 + math.exp(((-1)*self.sigma*(investment - self.r))))))
        return regain