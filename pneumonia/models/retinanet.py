import torch
import torchvision

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels, out_channels, 1)
        )
    def forward(self, x):
        return self.layers(x)

class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x, 1).view(x.shape[0], -1)

class SCSEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_gate = torch.nn.Sequential(
            GlobalAvgPool2d(),
            torch.nn.Linear(in_channels, in_channels // reduction),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_channels // reduction, in_channels),
            torch.nn.Sigmoid()
        )

        self.spatial_gate = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 1, 1, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.spatial_gate(x) * x + self.channel_gate(x).view(x.shape[0], -1, 1, 1) * x

class SCSEDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        )

        self.downsampler = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        self.relu = torch.nn.ReLU(inplace=True)
        self.scse = SCSEBlock(out_channels)

    def forward(self, x):
        return self.scse(self.relu(self.layers(x) + self.downsampler(x)))


class Retinanet(torch.nn.Module):
    def __init__(self, num_classes, num_anchors):
        super().__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)

        self.encoders = torch.nn.ModuleList([
            torch.nn.Sequential(
                self.resnet.conv1,
                self.resnet.bn1,
                self.resnet.relu,
                self.resnet.maxpool
            ),
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ])

        self.decoders = torch.nn.ModuleList([
            Decoder(512, 256),
            Decoder(256, 256),
            Decoder(128, 256),
            Decoder(64, 256)
        ])

        self.upscalers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 256, 2, stride=2, groups=256, bias=False),
                torch.nn.Conv2d(256, 256, 1, bias=False)
            ),
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 256, 2, stride=2, groups=256, bias=False),
                torch.nn.Conv2d(256, 256, 1, bias=False)
            ),
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 256, 2, stride=2, groups=256, bias=False),
                torch.nn.Conv2d(256, 256, 1, bias=False)
            ),
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 256, 2, stride=2, groups=256, bias=False),
                torch.nn.Conv2d(256, 256, 1, bias=False)
            )
        ])

        self.bbox_regressor = torch.nn.Sequential(
            SCSEDecoder(256, 128),
            torch.nn.Conv2d(128, num_anchors * 4, 1)
        )

        self.bbox_classifier = torch.nn.Sequential(
            SCSEDecoder(256, 128),
            torch.nn.Conv2d(128, num_anchors * num_classes, 1)
        )

    def forward(self, x):
        # (Pdb) x1.shape
        # torch.Size([2, 64, 56, 56])
        # (Pdb) x2.shape
        # torch.Size([2, 128, 28, 28])
        # (Pdb) x3.shape
        # torch.Size([2, 256, 14, 14])
        # (Pdb) x4.shape
        # torch.Size([2, 512, 7, 7])
        x = self.encoders[0](x)
        x1 = self.encoders[1](x)  # sees objects of 4 pixels and larger
        x2 = self.encoders[2](x1) # sees objects of 8 pixels and larger
        x3 = self.encoders[3](x2) # sees objects of 16 pixels and larger
        x4 = self.encoders[4](x3) # sees objects of 32 pixels and larger

        d1 = self.decoders[0](x4)
        d2 = self.decoders[1](x3) + self.upscalers[0](d1)
        d3 = self.decoders[2](x2) + self.upscalers[0](d2)
        d4 = self.decoders[3](x1) + self.upscalers[0](d3)
        bbox_logits = [self.bbox_classifier(d).view(x.shape[0], -1, 1) for d in [d1, d2, d3, d4]]
        bbox_deltas = [self.bbox_regressor(d).view(x.shape[0], -1, 4) for d in [d1, d2, d3, d4]]
        return (torch.cat(bbox_logits, dim=1), torch.cat(bbox_deltas, dim=1))


if __name__ == '__main__':
    batch = torch.rand((2, 3, 224, 224))
    model = Retinanet(1, 9)
    output = model(batch)
    import pdb; pdb.set_trace()
