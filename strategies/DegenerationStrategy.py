class DegenerationStrategy():
    def __init__(self, intercept, slope):
        self.intercept = intercept
        self.slope = slope

    def HealthDegeneration(self, currentHealth, currentRound):
        return max(currentHealth - int(round(self.intercept + (self.slope * currentRound))),0)