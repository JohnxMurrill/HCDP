import math

class LifeEnjoymentStrategy():
    def __init__(self, alpha, beta, mu, c):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.c = c

    def LifeEnjoyment(self, investment, currentHealth):
        enjoy = self.c*(self.beta*(currentHealth/100.0) + self.mu)*(1 - math.exp((-1)*self.alpha*investment))
        return enjoy
