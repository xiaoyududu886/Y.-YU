import json
from dataio.transformation.transforms import Augmentation


def get_dataset_augmentation(name, opts=None):

    trans_obj = Augmentation(name)
    if opts: trans_obj.initialise(opts)

    trans_obj.print()

    return trans_obj.get_augmentation()
