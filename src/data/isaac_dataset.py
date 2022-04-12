from data import Combination
from data import BuzzFeed
from data import FaKes
from data import FakeAndTrue
from data import Politifact

class IsaacDataset(Combination):
    def __init__(self, datasets=[]) -> None:
        datasets.extend([BuzzFeed(), FaKes(), FakeAndTrue(), Politifact()])
        super().__init__(datasets)
