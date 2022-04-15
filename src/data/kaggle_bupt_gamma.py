from data import BuptGammaWithValidation
from data import IsaacDataset

class KaggleBuptGamma(IsaacDataset):
    def __init__(self, datasets=[]) -> None:
        datasets.extend([BuptGammaWithValidation()])
        super().__init__(datasets)
