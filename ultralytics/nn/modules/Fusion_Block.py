import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Inception Module (Figure 4b and 4c)
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
        
        # Branch 2: 3x3 conv (k=9)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
        
        # Branch 3: 5x5 conv (k=9)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
        
        # Branch 4: MaxPool + 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # Concatenate the outputs of the four branches
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)

# Define the Attention Fusion Module (Figure 4a)
class AttentionFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels_per_branch):
        super(AttentionFusionModule, self).__init__()
        self.inception_ir = InceptionModule(in_channels, out_channels_per_branch)
        self.inception_vis = InceptionModule(in_channels, out_channels_per_branch)
    
    def forward(self, ir_feature, vis_feature):
        # Pass infrared and visible features through their respective Inception modules
        ir_enhanced = self.inception_ir(ir_feature)
        vis_enhanced = self.inception_vis(vis_feature)
        
        # Element-wise multiplication (attention mechanism)
        ir_enhanced = ir_enhanced * vis_enhanced
        vis_enhanced = vis_enhanced * ir_enhanced
        
        return ir_enhanced, vis_enhanced

# Define the Feature Fusion Module (Figure 4c)
class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        # Define the dilated convolution branches for each feature
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_d1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv3x3_d3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv5x5_d5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=10, dilation=5)
        self.conv7x7_d7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=21, dilation=7)
        
        # Final 1x1 convolution after concatenation and shuffle
        # We concatenate ir_sum and vis_sum, each with out_channels, so total channels = out_channels * 2
        self.final_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
    
    def forward(self, ir_feature, vis_feature):
        # Process infrared feature
        ir_1x1 = self.conv1x1(ir_feature)
        ir_3x3_d1 = self.conv3x3_d1(ir_feature)
        ir_3x3_d3 = self.conv3x3_d3(ir_feature)
        ir_5x5_d5 = self.conv5x5_d5(ir_feature)
        ir_7x7_d7 = self.conv7x7_d7(ir_feature)
        ir_sum = ir_1x1 + ir_3x3_d1 + ir_3x3_d3 + ir_5x5_d5 + ir_7x7_d7
        
        # Process visible feature
        vis_1x1 = self.conv1x1(vis_feature)
        vis_3x3_d1 = self.conv3x3_d1(vis_feature)
        vis_3x3_d3 = self.conv3x3_d3(vis_feature)
        vis_5x5_d5 = self.conv5x5_d5(vis_feature)
        vis_7x7_d7 = self.conv7x7_d7(vis_feature)
        vis_sum = vis_1x1 + vis_3x3_d1 + vis_3x3_d3 + vis_5x5_d5 + vis_7x7_d7
        
        # Concatenate the summed features
        concat_features = torch.cat([ir_sum, vis_sum], dim=1)
        
        # Shuffle operation (simulated by a 1x1 conv to mix channels)
        mixed_features = self.final_conv(concat_features)
        
        return mixed_features

# Combine both parts into a single model
class AttentionFusionNetwork(nn.Module):
    def __init__(self, in_channels, inception_out_channels_per_branch, fusion_out_channels):
        super(AttentionFusionNetwork, self).__init__()
        # Part 1: Attention Fusion Module
        self.attention_fusion = AttentionFusionModule(in_channels, inception_out_channels_per_branch)
        
        # Part 2: Feature Fusion Module
        # The input channels to the fusion module will be 4 * inception_out_channels_per_branch
        # because the Inception module concatenates 4 branches
        fusion_in_channels = 4 * inception_out_channels_per_branch
        self.feature_fusion = FeatureFusionModule(fusion_in_channels, fusion_out_channels)
    
    def forward(self, ir_feature, vis_feature):
        # Part 1: Get enhanced features
        ir_enhanced, vis_enhanced = self.attention_fusion(ir_feature, vis_feature)
        
        # Part 2: Fuse the enhanced features
        fused_feature = self.feature_fusion(ir_enhanced, vis_enhanced)
        
        return fused_feature