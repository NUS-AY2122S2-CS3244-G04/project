from data.dataset import Dataset
from data.dataset_using_dataframe import DatasetUsingDataframe
from data.bupt_gamma import BuptGamma
from data.bupt_gamma_with_validation import BuptGammaWithValidation
from data.buzzfeed import BuzzFeed
from data.fa_kes import FaKes
from data.fake_and_true import FakeAndTrue
from data.politifact import Politifact
from data.combination import Combination
from data.isaac_dataset import IsaacDataset
from data.janelle_dataset import JanelleDataset
from data.janelle_tokenized_dataset import JanelleTokenizedDataset
from data.singapore_test import SingaporeTest

datasets = {
    'bupt_gamma': BuptGamma,
    'bupt_gamma_with_validation': BuptGammaWithValidation,
    'buzzfeed': BuzzFeed,
    'fa_kes': FaKes,
    'fake_and_true': FakeAndTrue,
    'politifact': Politifact,
    'isaac_dataset': IsaacDataset,
    'janelle_dataset': JanelleDataset,
    'janelle_tokenized_dataset': JanelleTokenizedDataset,
    'singapore_test': SingaporeTest
}
