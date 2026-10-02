import torch
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import ToTensor


class DownstreamClassificationTrainTransformViT(torch.nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        min_ratio: float = 0.2,
    ):
        super().__init__()
        transforms_list = []

        if min_ratio < 1.0:
            transforms_list.append(
                transforms.RandomResizedCrop(
                    img_size,
                    scale=(min_ratio, 1.0),
                )
            )
        else:
            assert min_ratio == 1.0, "min_ratio must be <= 1.0"

            imagenet_crop_ratio = 224 / 256
            transforms_list.append(
                transforms.Resize(
                    int(img_size / imagenet_crop_ratio),
                )
            )
            transforms_list.append(
                transforms.CenterCrop(img_size),
            )

        transforms_list += [
            transforms.RandomHorizontalFlip(),
            ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]

        self.transform = transforms.Compose(transforms_list)

    def forward(self, sample):
        return self.transform(sample)


class DownstreamClassificationTestTransform(torch.nn.Module):
    def __init__(self, img_size: int = 224):
        super().__init__()
        imagenet_crop_ratio = 224 / 256

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    int(img_size / imagenet_crop_ratio),
                ),
                transforms.CenterCrop(img_size),
                ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )

    def forward(self, sample):
        return self.transform(sample)
