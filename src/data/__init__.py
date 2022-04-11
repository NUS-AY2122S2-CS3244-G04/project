from data.dataset import Dataset
from data.dataset_using_dataframe import DatasetUsingDataframe
from data.bupt_gamma import BuptGamma
from data.bupt_gamma_with_validation import BuptGammaWithValidation
from data.buzzfeed import BuzzFeed

datasets = {
    'bupt_gamma': BuptGamma,
    'bupt_gamma_with_validation': BuptGammaWithValidation,
    'buzzfeed': BuzzFeed
}
