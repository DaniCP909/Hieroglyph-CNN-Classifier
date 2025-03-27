from torch import nn
from torch.nn import Sequential

from GlyphnetModelBlocks import FirstBlock, InnerBlock, FinalBlock

class Glyphnet(nn.Module):

    def __init__(self, in_channels=1,
                 num_classes=1071,
                 first_conv_out=64,
                 last_sconv_out=512,
                 sconv_seq_outs=(128, 128, 256, 256),
                 dropout_rate=0.15):
        """
        :param in_channels: the number of channels in the input image
        :param num_classes: the number of labels for prediction
        :param first_conv_out:
        :param last_sconv_out:
        :param sconv_seq_outs:
        """

        super(Glyphnet, self).__init__()
        self.first_block = FirstBlock(in_channels, first_conv_out)
        in_channels_sizes = [first_conv_out] + list(sconv_seq_outs)
        self.inner_blocks = Sequential(*(InnerBlock(in_channels=i, sconv_out=o)
                                         for i, o in zip(in_channels_sizes, sconv_seq_outs)))
        self.final_block = FinalBlock(in_channels=in_channels_sizes[-1], out_size=num_classes,
                                      sconv_out=last_sconv_out, dropout_rate=dropout_rate)

    def forward(self, image_input_tensor):
        x = self.first_block(image_input_tensor)
        x = self.inner_blocks(x)
        x = self.final_block(x)

        return x