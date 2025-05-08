class HarvestStrategy():
    def __init__(self, maxHarvest):
        self.maxHarvest = maxHarvest

    def HarvestAmount(self, currentHealth):
        return int(round(self.maxHarvest*currentHealth/100))