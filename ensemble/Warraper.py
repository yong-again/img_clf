import numpy as np
import albumentations as A

class AlbumentationsWithAugMix:
    """
    A wrapper class that applies AugMix augmentation to an image after
    a set of base transformations.

    AugMix works by creating several augmented versions of an image and mixing
    them together, then combining this mixture with the original image. This
    can lead to more robust models.

    Args:
        base_transform (A.Compose): A composition of Albumentations transforms
            that are applied to the image before the AugMix process. These are
            standard augmentations like resizing, flipping, etc.
        normalize_transform (A.Compose): A composition of Albumentations transforms
            that are applied last, typically normalization and conversion to a tensor.
        severity (int): This parameter is part of the original AugMix paper but is
            not used in this specific implementation. It's kept for API consistency.
        mixture_width (int): The number of augmented images to mix together.
        alpha (float): The hyperparameter for the Dirichlet and Beta distributions,
            controlling the mixing weights and intensity.
    """
    def __init__(
            self,
            base_transform: A.Compose,
            normalize_transform: A.Compose,
            severity: int = 3,
            mixture_width: int = 3,
            alpha: float = 1.0
    ):
        # base_transform: Resize, RandomResizedCrop, Flip, etc (np.ndarray → np.ndarray)
        # normalize_transform: Normalize + ToTensorV2 (np.ndarray → Tensor)
        self.base_transform = base_transform
        self.normalize_transform = normalize_transform
        self.severity = severity
        self.mixture_width = mixture_width
        self.alpha = alpha

    def __call__(self, image: np.ndarray):
        orig = image.copy()
        orig_t = self.base_transform(image=orig)['image'].astype(np.float32)

        # Dirichlet (벡터) → numpy array
        ws = np.random.dirichlet([self.alpha] * self.mixture_width).astype(np.float32)
        # Beta (스칼라) → float or np.float32
        m = np.float32(np.random.beta(self.alpha, self.alpha))

        mix = np.zeros_like(orig_t, dtype=np.float32)
        for i in range(self.mixture_width):
            aug = image.copy()
            aug_t = self.base_transform(image=aug)['image'].astype(np.float32)
            mix += ws[i] * aug_t

        mixed = (1 - m) * orig_t + m * mix
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)

        return self.normalize_transform(image=mixed)