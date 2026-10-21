"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature
extraction from individual frames with an LSTM to capture temporal dynamics across frames.
An additional fully-connected layer is used to output the final class predictions.

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace the
            fully-connected layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization,
             and a final fully-connected layer to produce class logits.
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
    """
    hidden_size: int
    n_layers: int
    dropout_rate: float
    n_classes: int
    pretrained: bool = True
    cnn_model: str = 'resnet34'


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
            and which ResNet variant to use as the backbone). See LRCNConfig for details.

    Raises:
        ValueError: If config.cnn_model is not a supported ResNet variant.
    """

    def __init__(self, config):
        super().__init__()
        weights = "IMAGENET1K_V1" if config.pretrained else None

        # Set up the ResNet backbone as a 2D CNN feature extractor.
        if config.cnn_model == 'resnet18':
            base_cnn = models.resnet18(weights=weights)
        elif config.cnn_model == 'resnet34':
            base_cnn = models.resnet34(weights=weights)
        elif config.cnn_model == 'resnet50':
            base_cnn = models.resnet50(weights=weights)
        elif config.cnn_model == 'resnet101':
            base_cnn = models.resnet101(weights=weights)
        elif config.cnn_model == 'resnet152':
            base_cnn = models.resnet152(weights=weights)
        else:
            raise ValueError('The input CNN backbone is not supported, '
                             'please choose a valid ResNet variant.')

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features

        # Replace the original fc layer with an identity mapping so that raw features are returned.
        base_cnn.fc = Identity()
        self.base_model = base_cnn

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
        from the last time stepis then passed through dropout and the final fully-connected layer
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
