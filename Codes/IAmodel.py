import torch
import torch.nn as nn

class SharedEncoderDecoderBlock(nn.Module):
    def __init__(self, in_channels):
        super(SharedEncoderDecoderBlock, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()

        # Max pooling
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

        # Average pooling
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        # Shared Encoder-Decoder Block
        self.shared_encoder_decoder = SharedEncoderDecoderBlock(in_channels)

    def forward(self, x):
        # Max pooling
        max_pooled = self.max_pooling(x)

        # Average pooling
        avg_pooled = self.avg_pooling(x)

        # Shared Encoder-Decoder for pooled results
        max_pooled_encoded_decoded = self.shared_encoder_decoder(max_pooled)
        avg_pooled_encoded_decoded = self.shared_encoder_decoder(avg_pooled)

        # Element-wise summation of encoded-decoded results
        pooled_sum = max_pooled_encoded_decoded + avg_pooled_encoded_decoded

        # Element-wise activation
        pooled_sum = nn.functional.relu(pooled_sum)

        # Broadcast into H*W*C
        pooled_sum_broadcasted = pooled_sum.expand_as(x)

        # Element-wise multiplication with the original input
        attention_out = x * pooled_sum_broadcasted

        return attention_out

class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()

        # Concatenation and convolution layer
        self.concat_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Max pooling
        max_pooled_final, _ = torch.max(x, dim=1, keepdim=True)
        
        # Average pooling
        avg_pooled_final = torch.mean(x, dim=1, keepdim=True)
        
        # Concatenate max and average pooled results
        final_concat = torch.cat([max_pooled_final, avg_pooled_final], dim=1)

        # Convolution layer
        final_conv_result = self.concat_conv(final_concat)

        # Element-wise activation
        final_conv_result = nn.functional.relu(final_conv_result)

        # Broadcast into H*W*C
        final_conv_result_broadcasted = final_conv_result.expand_as(x)

        # Element-wise multiplication with the attention_out
        final_result = x * final_conv_result_broadcasted

        return final_result

class InceptionAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionAttentionBlock, self).__init__()

        # Path 1: 1x1 convolution followed by pooling
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # Path 2: 1x1 convolution followed by 3x3 convolution then pooling
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # Path 3: 1x1 convolution followed by 5x5 convolution then pooling
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # Path 4: Pooling followed by 1x1 convolution
        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        

    def forward(self, x):
        # Path 1
        path1 = self.path1(x)

        # Path 2
        path2 = self.path2(x)

        # Path 3
        path3 = self.path3(x)

        # Path 4
        path4 = self.path4(x)

        # Concatenate the results of all paths
        inception_output = torch.cat([path1, path2, path3, path4], dim=1)

        return inception_output

class SequentialInceptionAttentionBlocks(nn.Module):
    def __init__(self, num_blocks=5):
        super(SequentialInceptionAttentionBlocks, self).__init__()

        # Initial block with input size 256*256*1
        self.initial_block = InceptionAttentionBlock(1)

        # Sequential Inception-Attention blocks
        self.inception_attention_blocks = nn.Sequential(
            *[nn.Sequential(InceptionAttentionBlock(4), ChannelAttentionBlock(4), SpatialAttentionBlock(4)) for _ in range(num_blocks - 1)]
        )

        self.fc1 = nn.Linear(256 * 256 * 4, 1024)
        self.fc2 = nn.Linear(1024, 250)
        self.fc3 = nn.Linear(250, 2)

    def forward(self, x):
        # Initial block
        x = self.initial_block(x)

        # Sequential Inception-Attention blocks
        x = self.inception_attention_blocks(x)

        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

