import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.layer1_conv = nn.Conv2d(1,64,4,2,padding=1)
        self.layer2_conv = nn.Conv2d(64,128,4,2,padding=1)
        self.layer3_conv = nn.Conv2d(128,256,4,2,padding=1)
        self.layer4_conv = nn.Conv2d(256,512,4,2,padding=1)
        self.layer5_conv = nn.Conv2d(512,512,4,2,padding=1)
        self.layer6_conv = nn.Conv2d(512,512,4,2,padding=1)
        self.layer7_conv = nn.Conv2d(512,512,4,2,padding=1)
        self.layer8_conv = nn.Conv2d(512,512,4,2,padding=1)

        self.layer1_transconv = nn.ConvTranspose2d(512, 512, 4, 2, padding=1, output_padding=0)
        self.layer2_transconv = nn.ConvTranspose2d(1024, 512, 4, 2, padding=1, output_padding=0)
        self.layer3_transconv = nn.ConvTranspose2d(1024, 512, 4, 2, padding=1, output_padding=0)
        self.layer4_transconv = nn.ConvTranspose2d(1024, 512, 4, 2, padding=1, output_padding=0)
        self.layer5_transconv = nn.ConvTranspose2d(1024, 256, 4, 2, padding=1, output_padding=0)
        self.layer6_transconv = nn.ConvTranspose2d(512, 128, 4, 2, padding=1, output_padding=0)
        self.layer7_transconv = nn.ConvTranspose2d(256, 64, 4, 2, padding=1, output_padding=0)
        self.layer8_transconv = nn.ConvTranspose2d(128, 1, 4, 2, padding=1, output_padding=0)

        self.relu = nn.ReLU()
        self.Tanh = nn.Tanh()
        #self.concat = torch.cat()

    def forward(self,x):
        conv1 = self.layer1_conv(x)
        conv1= self.relu(conv1)

        conv2 = self.layer2_conv(conv1)
        conv2 = self.relu(conv2)

        conv3 = self.layer3_conv(conv2)
        conv3 = self.relu(conv3)

        conv4 = self.layer4_conv(conv3)
        conv4 = self.relu(conv4)

        conv5 = self.layer5_conv(conv4)
        conv5 = self.relu(conv5)

        conv6 = self.layer6_conv(conv5)
        conv6 = self.relu(conv6)

        conv7 = self.layer7_conv(conv6)
        conv7 = self.relu(conv7)

        conv8 = self.layer8_conv(conv7)
        conv8 = self.relu(conv8)

        transconv1 = self.layer1_transconv(conv8)
        transconv1 = torch.cat([transconv1,conv7], dim=1)
        transconv1 = self.relu(transconv1)

        transconv2 = self.layer2_transconv(transconv1)
        transconv2 = torch.cat([transconv2, conv6], dim=1)
        transconv2 = self.relu(transconv2)

        transconv3 = self.layer3_transconv(transconv2)
        transconv3 = torch.cat([transconv3, conv5], dim=1)
        transconv3 = self.relu(transconv3)

        transconv4 = self.layer4_transconv(transconv3)
        transconv4 = torch.cat([transconv4, conv4], dim=1)
        transconv4 = self.relu(transconv4)

        transconv5 = self.layer5_transconv(transconv4)
        transconv5 = torch.cat([transconv5, conv3], dim=1)
        transconv5 = self.relu(transconv5)

        transconv6 = self.layer6_transconv(transconv5)
        transconv6 = torch.cat([transconv6, conv2], dim=1)
        transconv6 = self.relu(transconv6)

        transconv7 = self.layer7_transconv(transconv6)
        transconv7 = torch.cat([transconv7, conv1], dim=1)
        transconv7 = self.relu(transconv7)

        transconv8 = self.layer8_transconv(transconv7)
        transconv8 = self.Tanh(transconv8)

        return transconv8

net = Unet()