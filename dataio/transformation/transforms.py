import torchsample.transforms as ts
from pprint import pprint


class Augmentation:

    def __init__(self, name):
        self.name = name
        self.scale_size = (192, 192, 1)
        self.patch_size = (128, 128, 1)
        self.shift_val = (0.1, 0.1)
        self.rotate_val = 15.0
        self.scale_val = (0.7, 1.3)
        self.inten_val = (1.0, 1.0)
        self.random_flip_prob = 0.0

        # Divisibility factor for testing
        self.division_factor = (16, 16, 1)

    def get_augmentation(self):
        return {
            'isprs_sax': self.isprs_3d_sax_transform,
            'hms_sax':  self.isprs_sax_transform,
            'test_sax': self.test_3d_sax_transform
        }[self.name]()

    def print(self):
        print(vars(self))

    def initialise(self, opts):
        t_opts = getattr(opts, self.name)

        if hasattr(t_opts, 'scale_size'): self.scale_size = t_opts.scale_size
        if hasattr(t_opts, 'patch_size'): self.patch_size = t_opts.patch_size
        if hasattr(t_opts, 'shift_val'): self.shift_val = t_opts.shift
        if hasattr(t_opts, 'rotate_val'): self.rotate_val = t_opts.rotate
        if hasattr(t_opts, 'scale_val'): self.scale_val = t_opts.scale
        if hasattr(t_opts, 'inten_val'): self.inten_val = t_opts.intensity
        if hasattr(t_opts, 'random_flip_prob'): self.random_flip_prob = t_opts.random_flip_prob
        if hasattr(t_opts, 'division_factor'): self.division_factor = t_opts.division_factor


    def isprs_sax_transform(self):
        train_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      ts.RandomFlip(h=True, v=True, p=self.random_flip_prob),
                                      ts.RandomAffine(rotation_range=self.rotate_val, translation_range=self.shift_val,
                                                      zoom_range=self.scale_val, interp=('bilinear', 'nearest')),
                                      ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      ts.RandomCrop(size=self.patch_size),
                                      ts.TypeCast(['float', 'long'])
                                ])

        valid_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      ts.SpecialCrop(size=self.patch_size, crop_type=0),
                                      ts.TypeCast(['float', 'long'])
                                ])
        return {'train': train_transform, 'valid': valid_transform}


    def isprs_3d_sax_transform(self):
        train_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      ts.RandomFlip(h=True, v=True, p=self.random_flip_prob),
                                      ts.RandomAffine(rotation_range=self.rotate_val, translation_range=self.shift_val,
                                                      zoom_range=self.scale_val, interp=('bilinear', 'nearest')),
                                      #ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      ts.NormalizeMedic(norm_flag=(True, False)),
                                      ts.ChannelsLast(),
                                      ts.AddChannel(axis=0),
                                      ts.RandomCrop(size=self.patch_size),
                                      ts.TypeCast(['float', 'long'])
                                ])

        valid_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      #ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      ts.NormalizeMedic(norm_flag=(True, False)),
                                      ts.ChannelsLast(),
                                      ts.AddChannel(axis=0),
                                      ts.SpecialCrop(size=self.patch_size, crop_type=0),
                                      ts.TypeCast(['float', 'long'])
                                ])

        return {'train': train_transform, 'valid': valid_transform}

    def test_3d_sax_transform(self):
        test_transform = ts.Compose([ts.PadFactorNumpy(factor=self.division_factor),
                                     ts.ToTensor(),
                                     ts.ChannelsFirst(),
                                     ts.TypeCast(['float']),
                                     ts.NormalizeMedic(norm_flag=True),
                                     ts.ChannelsLast(),
                                     ts.AddChannel(axis=0),
                                     ])

        return {'test': test_transform}