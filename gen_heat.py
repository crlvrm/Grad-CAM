import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import get_model
from heat_utils import GradCAM, show_cam_on_image, create_AugTransforms
# from utils.augment import create_AugTransforms
import torchvision.transforms as T
import argparse

def image_process(path: str, transforms: T.Compose):
    img = Image.open(path).convert('RGB')
    return transforms(img)

def parse_opt():
    parsers = argparse.ArgumentParser()
    parsers.add_argument('--weight_path', type=str, default='./run/mobilenetv2_30epoch/best.pt')
    parsers.add_argument('--img_path', type=str, default='./7.jpg')
    parsers.add_argument('--target_category', type=int, default=1)
    parsers.add_argument('--imgsz', default='[[720, 720], [224, 224]]', type=str)
    parsers.add_argument('--transforms', type=str, default='centercrop_resize to_tensor_without_div')
    args = parsers.parse_args()
    return args

def main(opt):
    imgsz = opt.imgsz
    weight_path = opt.weight_path
    target_category = opt.target_category 
    img_path = opt.img_path
    transforms = opt.transforms

    '''构建模型，将模型权重导入'''
    model = get_model('mobilenet_v2', width_mult = 0.25)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 7)
    weight = torch.load(weight_path, map_location='cpu')['model']
    model.load_state_dict(weight)
    '''选取features中的最后一层'''
    target_layers = [model.features[-1]]

    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = image_process(img_path, create_AugTransforms(transforms, eval(imgsz)))
    
    # 调用算法求出权重
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=img.unsqueeze(0), target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]

    # 对图像进行resize操作，再和热力图融合输出
    transforms1 = transforms.split(' ')[0]
    img1 = np.array(image_process(img_path, create_AugTransforms(transforms1, eval(imgsz))), dtype=np.uint8)

    visualization = show_cam_on_image(img1.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
