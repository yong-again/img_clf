import torch
import torch.nn as nn
import timm

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='convnext_base'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.backbone(x)
        if x.dim() == 4:
            x = self.pool(x)
            x = x.flatten(1)
        return x

def extract_features(model, dataloader, device, writer=None, tag_prefix="train"):
    model.eval()
    features, targets = [], []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            feat = model(images)
            features.append(feat.cpu())
            targets.append(labels.cpu())

            if writer and i == 0:
                writer.add_image(f"{tag_prefix}/sample", images[0].cpu(), 0)

    return torch.cat(features).numpy(), torch.cat(targets).numpy()
