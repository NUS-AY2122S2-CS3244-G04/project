from data.dataset import Dataset
from data.dataset_using_dataframe import DatasetUsingDataframe
from data.bupt_gamma import BuptGamma
from data.bupt_gamma_with_validation import BuptGammaWithValidation
from data.buzzfeed import BuzzFeed
from data.fa_kes import FaKes
from data.fake_and_true import FakeAndTrue

datasets = {
    'bupt_gamma': BuptGamma,
    'bupt_gamma_with_validation': BuptGammaWithValidation,
    'buzzfeed': BuzzFeed,
    'fa_kes': FaKes,
    'fake_and_true': FakeAndTrue
}
