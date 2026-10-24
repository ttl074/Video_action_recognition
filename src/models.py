"""
Module: models.py

This module defines two models for video classification.

The LRCN (Long-term Recurrent Convolutional Network) combines a 2D CNN backbone (e.g.,
ResNet) for spatial feature extraction from individual frames with an LSTM to capture
temporal dynamics across frames. An additional fully-connected layer is used to output
the final class predictions. Backbone stages can be frozen.

The R3D-18 model is a 3D CNN that learns spatial and temporal features together. It can be
initialized from ImageNet ResNet18 weights inflated along the time axis, so no
video-pretrained weights are needed.

Classes:
    LRCNConfig: Hyperparameters for building an LRCN model.
    Identity: A helper module that returns the input unchanged. It is used to replace the
        fully-connected layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization,
        and a final fully-connected layer to produce class logits.

Functions:
    key_translation: Map an R3D-18 state-dict key to the matching ResNet18 key.
    inflate_conv_weight: Turn a 2D conv weight into a 3D one by repeating it along time
        and dividing by the temporal size.
    build_r3d18: Build an R3D-18 classifier, optionally initialized from inflated ImageNet
        ResNet18 weights.
"""

from dataclasses import dataclass
from torch import nn
from torchvision import models

@dataclass
class LRCNConfig:
    """
    Hyperparameters for building an LRCN model.

    Attributes:
        hidden_size (int): Number of features in the LSTM hidden state.
        n_layers (int): Number of stacked LSTM layers.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool): If True, load ImageNet-1k V1 weights for the CNN backbone.
        cnn_model (str): ResNet variant for the backbone ('resnet18' ... 'resnet152').
        freeze_layers (tuple): Names of ResNet stages whose weights are frozen and
                                BatchNorm statistics are also held fixed
    """
    hidden_size: int
    n_layers: int
    dropout_rate: float
    n_classes: int
    pretrained: bool = True
    cnn_model: str = 'resnet34'
    freeze_layers: tuple = ()

class Identity(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.

    This module is used to replace the fully-connected (fc) layer in the ResNet backbone,
    effectively making the backbone output the raw features before classification.

    Example:
        >>> identity = Identity()
        >>> output = identity(input_tensor)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass that returns the input as is.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: The same tensor x.
        """
        return x

class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.

    This model uses a ResNet backbone as a 2D CNN to extract spatial features from each
    video frame. An LSTM network is then used to model the temporal dynamics across the
    sequence of frame features. Dropout is applied before the final fully-connected layer
    that produces class logits.

    Args:
        config (LRCNConfig): Model hyperparameters (LSTM hidden size and number of layers,
            dropout rate, number of classes, whether to use ImageNet-pretrained weights,
            which ResNet variant to use as the backbone, and which backbone stages to
            freeze). See LRCNConfig for details.

    Raises:
        ValueError: If config.cnn_model is not a supported ResNet variant.
    """

    def __init__(self, config):
        super().__init__()
        weights_v1 = "IMAGENET1K_V1" if config.pretrained else None
        weights_v2 = "IMAGENET1K_V2" if config.pretrained else None
        # Set up the ResNet backbone as a 2D CNN feature extractor.
        if config.cnn_model == 'resnet18':
            base_cnn = models.resnet18(weights=weights_v1)
        elif config.cnn_model == 'resnet34':
            base_cnn = models.resnet34(weights=weights_v1)
        elif config.cnn_model == 'resnet50':
            base_cnn = models.resnet50(weights=weights_v2)
        elif config.cnn_model == 'resnet101':
            base_cnn = models.resnet101(weights=weights_v2)
        elif config.cnn_model == 'resnet152':
            base_cnn = models.resnet152(weights=weights_v2)
        else:
            raise ValueError('The input CNN backbone is not supported, '
                             'please choose a valid ResNet variant.')

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features

        # Replace the original fc layer with an identity mapping so that raw features are returned.
        base_cnn.fc = Identity()
        self.base_model = base_cnn
        # freeze layers if specified
        self.freeze_layers = config.freeze_layers
        for layer in self.freeze_layers:
            for param in getattr(self.base_model, layer).parameters():
                param.requires_grad = False
        # LSTM over the sequence of frame features; expects (batch, time, features).
        self.rnn = nn.LSTM(num_features, config.hidden_size, config.n_layers, batch_first=True)

        # Define dropout for regularization.
        self.dropout = nn.Dropout(config.dropout_rate)

        # Final fully-connected layer to produce logits for each class.
        self.fc = nn.Linear(config.hidden_size, config.n_classes)

    def forward(self, x):
        """
        Forward pass for the LRCN model.

        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)

        All B*T frames go through the CNN in one batch, the features are reshaped to (B,T,F),
        and the LSTM(batch-first) processes each clip's frame sequence independently. The output
        from the last time step is then passed through dropout and the final fully-connected layer
        to produce the class logits.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        bs, ts, c, h, w = x.shape  # batch_size, time_steps, channels, height, width
        # (B*T, C, H, W)
        x_merge = x.reshape(bs*ts,c,h,w)
        # (B*T, F)
        y_merge = self.base_model(x_merge)
        # (B, T, F)
        y = y_merge.reshape(bs,ts,-1)
        # (B, T, Hidden)
        out,_ = self.rnn(y)
        # Apply dropout to the output of the final time step.
        out = self.dropout(out[:, -1])
        # Pass the final output through the fully-connected layer to get class logits.
        out = self.fc(out)
        return out

    def train(self, mode=True):
        """
        Set training or evaluation mode, keeping frozen backbone stages in eval mode.

        Calls nn.Module.train(mode) on the whole model, then switches every stage listed in
        self.freeze_layers back to eval mode. Frozen stages have requires_grad=False, but
        BatchNorm layers still update their running statistics whenever they are in training
        mode; without this override, model.train() at the start of each epoch would let the
        frozen stages' BatchNorm statistics drift away from their ImageNet values.

        Args:
            mode (bool): True for training mode, False for evaluation mode. model.eval()
                calls this method with mode=False.

        Returns:
            LRCN: The model itself, so calls can be chained (e.g. model.train().to(device)).
        """
        super().train(mode)
        for freeze_layer in self.freeze_layers:
            layer = getattr(self.base_model, freeze_layer)
            layer.eval()
        return self


def key_translation(key):
    """
    Translate an R3D-18 state-dict key to its ResNet18 equivalent.

    R3D-18 wraps each conv and its BatchNorm in an nn.Sequential, so they appear as
    'convN.0' and 'convN.1' (and 'stem.0', 'stem.1'), while ResNet18 names them 'convN' and
    'bnN' (and 'conv1', 'bn1'). Keys that need no change, such as the downsample keys, are
    returned as they are.

    Args:
        key (str): State-dict key from R3D-18, e.g. 'layer1.0.conv1.1.weight'.

    Returns:
        str: The matching ResNet18 key, e.g. 'layer1.0.bn1.weight'.
    """
    translations = {
        'stem.0.':'conv1.',
        'stem.1.':'bn1.',
        '.conv1.0.':'.conv1.',
        '.conv1.1.':'.bn1.',
        '.conv2.0.':'.conv2.',
        '.conv2.1.':'.bn2.'
    }
    for r3d, resnet in translations.items():
        key = key.replace(r3d, resnet)
    return key


def inflate_conv_weight(weight, t):
    """
    Inflate a 2D ResNet conv weight along a new time axis for R3D.

    The 2D kernel is repeated t times along a new time dimension and divided by t. On a clip
    where every frame is the same, the 3D conv then gives the same response as the 2D conv
    on that frame, which keeps the copied BatchNorm running statistics valid. The input
    weight is not modified.

    Args:
        weight (Tensor): 2D conv weight of shape (out, in, k, k).
        t (int): Temporal size of the matching 3D kernel (3 for the stem and 3x3x3 convs,
            1 for the downsample convs).

    Returns:
        Tensor: Inflated weight of shape (out, in, t, k, k), each time slice equal to the
            2D kernel divided by t.
    """
    temporal_weight = weight.unsqueeze(2)
    repeated = temporal_weight.repeat(1, 1, t, 1, 1)
    norm_weight = repeated / t
    return norm_weight


def build_r3d18(n_classes, dropout, pretrained):
    """
    Build an R3D-18 video classifier, optionally initialized from ImageNet ResNet18.

    When pretrained is True, every R3D-18 weight is copied from the ImageNet-pretrained
    ResNet18: BatchNorm tensors are copied as they are, and each 2D conv kernel is inflated
    along time by repeating it t times and dividing by t, so the network's response to a
    static clip matches ResNet18's response to that frame. No video-pretrained (Kinetics)
    weights are used. The original classification head is replaced with dropout followed
    by a linear layer with n_classes outputs.

    Args:
        n_classes (int): Number of output classes.
        dropout (float): Dropout probability applied before the final linear layer.
        pretrained (bool): Whether to initialize from inflated ImageNet ResNet18 weights.

    Returns:
        VideoResNet: The R3D-18 model with a new classification head.

    Raises:
        ValueError: If an R3D-18 weight cannot be matched to a ResNet18 weight of a
            compatible shape.
    """
    model = models.video.r3d_18(weights=None)
    if pretrained:
        weights = "IMAGENET1K_V1"
        resnet2d = models.resnet18(weights=weights)
        state_3d = model.state_dict()
        state_2d = resnet2d.state_dict()
        weights_3d = {}
        for key, weight in state_3d.items():
            if key.startswith('fc.'):
                weights_3d[key] = weight
                continue
            translation = key_translation(key)
            weight_2d = state_2d[translation]
            if weight.shape == weight_2d.shape:
                weights_3d[key] = weight_2d
            elif weight.ndim == 5 and weight_2d.ndim == 4:
                weights_3d[key] = inflate_conv_weight(weight_2d, weight.shape[2])
            else:
                raise ValueError(f"Cannot map {key} to {translation}: "
                                 f"3D shape {tuple(weight.shape)}, "
                                 f"2D shape {tuple(weight_2d.shape)}")
        model.load_state_dict(weights_3d)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, n_classes)
    )
    return model
