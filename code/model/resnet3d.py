import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit
from collections import OrderedDict

# simple 5-layer 3D CNN with skip connection
class Resnet3D(nn.Module):
    '''
    Layer 1: conv
    Layer 2,3: 1 residual block
    Layer 4,5: 1 residual block
    Layer 6: fc layer
    Args:
        dropout_block = [0,0,0]. Dropout for layer 2-3, 4-5, fc 6 layer
    '''
    def __init__(self, in_channels, num_class, dropout_block = [0,0,0]):
        super(Resnet3D, self).__init__()
        init_features =16 # number of filters in the first convolution layer.

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv3d(in_channels, init_features, kernel_size=3, stride=1, padding=1)),
                    ("norm0", nn.BatchNorm3d(init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool3d(kernel_size=2, stride=2)),
                ]
            )
        )
        in_channels = init_features
        for i, num_channel in enumerate([32, 64]):
            block = ResidualUnit(
                subunits=2,
                dimensions=3,
                in_channels=in_channels,
                out_channels=num_channel,
                act='relu',
                norm='batch',
                dropout=dropout_block[i],
            )
            self.features.add_module(f"residualblock{i + 1}", block)
            in_channels = num_channel

        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("norm", nn.AdaptiveAvgPool3d(1)),
                    ("flatten", nn.Flatten(1)),
                    ("class", nn.Linear(in_channels, num_class)),
                ]
            )
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.class_layers(x)
        return x
