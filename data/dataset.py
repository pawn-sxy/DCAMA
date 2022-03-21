r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torch.utils.data.distributed import DistributedSampler as Sampler
from torch.utils.data import DataLoader
from torchvision import transforms

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold,
                                          transform=cls.transform,
                                          split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        train_sampler = Sampler(dataset) if split == 'trn' else None
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, sampler=train_sampler, num_workers=nworker,
                                pin_memory=True)

        return dataloader
