'''
Date: 2023-05-08 16:34:17
FirstEditors: wangjl 446976612@qq.com
LastEditors: wangjl 446976612@qq.com
LastEditTime: 2023-05-09 17:38:40
FilePath: /active_vision_eval_model/dataset.py
'''



from torch.utils.data import DataLoader, Dataset
import os
import torch
import torchvision.transforms as transforms
from PIL import Image 
import numpy as np
from pathlib import Path 
from typing import Union, Callable, Optional, List
import cv2 
import open3d as o3d

MAX_DEPTH = 800 # 拍摄范围 0-60cm，有发现超过600的深度值，改成800

class ImitationDataset(Dataset):

    label2classid = {'OK': 0, 'NG': 1}
    print('label2classid: ', label2classid)
    def __init__(
            self, roots: List[Union[Path, str]], transform_funs: Optional[Callable], 
            imgsz: Optional[List[int]]=[224,224], img_channel: str='rgbd', 
        ) -> None:

        self.roots = roots
        
        self.data_list = self._bulid_data_list()
        self.transform_funs = transform_funs
        self.imgsz = imgsz
        
        if img_channel not in ['rgb', 'depth', 'rgbd']:
            raise ValueError("channel must be in ['rgb', 'depth', 'rgbd'].")
        self.img_channel = img_channel

    def _bulid_data_list(self,):
        data_list = []
        for root in self.roots:
            root = Path(root)
            n = 0 
            for one_class_data_dir in root.iterdir():
                if one_class_data_dir.is_dir():
                    for jpg_f in one_class_data_dir.iterdir():
                        if jpg_f.suffix in ['.jpg']:
                            pcd_f = jpg_f.with_suffix('.pcd')
                            class_id = self.label2classid[jpg_f.parent.name]
                            if pcd_f.exists():
                                data_list.append((str(jpg_f), str(pcd_f), int(class_id)))
                                n += 1
                            else:
                                print(f"{jpg_f} does not have a corresponding pcd file.")
            print(f">>>{root} has {n} images.")
        return data_list
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        jpg_f, pcd_f, label = self.data_list[idx]
        
        if self.img_channel == 'rgbd':
            rgb_img = Image.open(jpg_f).convert('RGB')
            pcd = o3d.io.read_point_cloud(pcd_f)
            depth_img = self.pcd2depth(pcd, rgb_img.size[::-1])
            rgb_img = rgb_img.resize(self.imgsz)
            depth_img = cv2.resize(depth_img, self.imgsz)
            rgb_img = self.transform_funs(rgb_img)
            depth_img = transforms.ToTensor()((depth_img/MAX_DEPTH).astype(np.float32))
            data = torch.cat([rgb_img, depth_img])
        
        elif self.img_channel == 'rgb':
            rgb_img = Image.open(jpg_f).convert('RGB').resize(self.imgsz)
            data = self.transform_funs(rgb_img)
        
        elif self.img_channel == 'depth':
            rgb_img = Image.open(jpg_f)
            pcd = o3d.io.read_point_cloud(str(pcd_f))
            depth_img = self.pcd2depth(pcd, rgb_img.size[::-1])
            depth_img = cv2.resize(depth_img, self.imgsz)
            data = transforms.ToTensor()((depth_img/MAX_DEPTH).astype(np.float32))
        
        else:
            raise ValueError("channel must be in ['rgb', 'depth', 'rgbd'].")
        return data, label

    @staticmethod
    def pcd2depth(pcd, img_shape):
        pcd_array = np.asarray(pcd.points)[:,2].reshape(img_shape)
        mask = np.isnan(pcd_array)
        pcd_array[mask] = 0
        return pcd_array



def load_dataset(root,batch_size):
    transform_funs = transforms.Compose([
        # transforms.ColorJitter(0.3, 0.3, 0.3),
        # transforms.GaussianBlur(5),
        transforms.ToTensor(),
        # transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    train_iter = DataLoader(ImitationDataset(root, transform_funs), batch_size, shuffle=True)
    return train_iter

if __name__ == '__main__':
    print(os.getcwd())
    roots=['/media/datum/wangjl/data/active_vision_dataset/eval_model_dataset/batch1/images']
    batch_size=32
    transform_funs = transforms.Compose([
        # transforms.ColorJitter(0.3, 0.3, 0.3),
        # transforms.GaussianBlur(5),
        transforms.ToTensor(),
        # transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    dataset = ImitationDataset(roots, transform_funs, img_channel='rgbd')
    dataloader = DataLoader(dataset, batch_size=32)
    for i, (x, y) in enumerate(dataloader):
        print(x.shape, y.shape)
        img2d = transforms.ToPILImage()(x[1, :3, ...])
        img3d = transforms.ToPILImage()(x[1, 3, ...])
        img2d.save('img2d.png')
        img3d.save('img3d.png')
        break
