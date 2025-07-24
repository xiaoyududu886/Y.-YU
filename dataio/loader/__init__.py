import json
from dataio.loader.test_dataset import TestDataset


def get_dataset(name):
    return {'acdc_sax': ISPRSDataset, 'test_sax': TestDataset},[name]


def get_dataset_path(dataset_name, opts):
    return getattr(opts, dataset_name)
