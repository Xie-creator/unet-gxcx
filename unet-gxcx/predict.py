import torch
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np

trans = transforms.Compose([transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(r"G:\python project\unet-gxcx\unet1.path") #加载模型
img_path = '17441_00569_41_30_wf.jpg'
img = cv2.imread(img_path,flags=0)

img0 = cv2.copyMakeBorder(img, 96, 96, 96, 96, cv2.BORDER_CONSTANT) #填充

img_tensor = trans(img)
img_tensor = img_tensor.unsqueeze(0) #升维
img_tensor = img_tensor.to(device)
output_tensor = model(img_tensor)
output_tensor.to("cpu")
output_tensor = output_tensor.squeeze(dim=1) #
output_tensor = output_tensor.squeeze(dim=0)
print(output_tensor.size())

output = output_tensor.data.cpu().numpy()
output*=255
picture = Image.fromarray(output)
picture = picture.convert("L")
picture.save('output.jpg')