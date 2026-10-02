import torchvision.transforms as transforms
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MAEPretextTransform(torch.nn.Module):
    def __init__(self, img_size=224, img_crop_min_ratio: float = 0.2):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size, scale=(img_crop_min_ratio, 1.0), interpolation=3  # type: ignore
                ),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )

    def forward(self, sample) -> torch.Tensor:
        return self.transform(sample)
