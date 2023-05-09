
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from importlib import import_module
import os
import shutil
from omegaconf import OmegaConf

from model import EvalModel
from dataset import ImitationDataset

def write_log(file, content):
    with open(file, "a") as f:
        f.write(content) 

class ActiveVisionEvalModel(object):
    def __init__(self, cfg_file) -> None:
        self.cfg_file = cfg_file
        self.cfg = OmegaConf.load(cfg_file)
        self.save_root = self.cfg.save_root
        os.makedirs(self.save_root, exist_ok=True)
        self.log = os.path.join(self.save_root, "train_log.log")
        self.device = self.cfg.device
        self.build_model()
        self.criterion = nn.CrossEntropyLoss()

    def build_model(self,):
        model_cfg = self.cfg.model
        model_params = model_cfg.get("model_params", None)
        if model_params is not None:
            self.model = EvalModel(getattr(import_module("model"), model_cfg.build_backbone_fun), **model_cfg.params)
        else:
            self.model = EvalModel(getattr(import_module("model"), model_cfg.build_backbone_fun))
        self.model.to(self.device)
        
        ckpt_file = model_cfg.get("ckpt", None)
        if ckpt_file:
            ckpt = torch.load(ckpt_file, map_location=self.device)
            self.model.load_state_dict(ckpt)
            write_log(self.log, f"load {ckpt_file} success!")
        


    def build_dataloader(self, data_cfg):
        transform_funs = []
        transforms_cfg = OmegaConf.to_container(data_cfg.transforms)

        for f, params in transforms_cfg.items():
            if params is None:
                fun = getattr(transforms, f)()
            elif isinstance(params, list):
                fun = getattr(transforms, f)(*params)
            elif isinstance(params, dict):
                fun = getattr(transforms, f)(**params)
            else:
                raise TypeError("params must be in [null, list, dict]...")
            transform_funs.append(fun)
        transform_funs = transforms.Compose(transform_funs)
        dataset = ImitationDataset(roots=data_cfg.roots, transform_funs=transform_funs)
        loader = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=data_cfg.shuffle, num_workers=data_cfg.num_workers)
        return loader


    
    def build_optimizer(self, train_cfg):
        optimizer_cfg = train_cfg.optimizer
        optimizer = getattr(optim, optimizer_cfg.name)(self.model.parameters(), **optimizer_cfg.params)

        scheduler_cfg = train_cfg.scheduler
        scheduler = getattr(optim.lr_scheduler, scheduler_cfg.name)(optimizer, **scheduler_cfg.params)

        return optimizer, scheduler

    def train(self,):

        shutil.copy2(self.cfg_file, self.save_root)

        train_cfg = self.cfg.train
        epochs = train_cfg.epoch
        train_loader = self.build_dataloader(train_cfg.train_data)
        val_loader = self.build_dataloader(train_cfg.val_data)
        
        optimizer, scheduler = self.build_optimizer(train_cfg)


        best_acc = 0
        for epoch in range(1, 1+epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            content = f"epoch={epoch}, lr={optimizer.state_dict()['param_groups'][0]['lr']:.6f}, loss={train_loss:.6f}, train_acc={correct/total:.6f} \n"
            print(content)
            write_log(self.log, content)

            acc = self.valid(val_loader)
            if acc >= best_acc:
                best_acc = acc
                save_name = os.path.join(self.save_root, "best.pth")
                torch.save(self.model.state_dict() , save_name)
                write_log(self.log, "save best model... \n")
            
            scheduler.step()
            write_log(self.log, "\n")

        save_name = os.path.join(self.save_root, "last.pth")
        torch.save(self.model.state_dict() , save_name)
        
    def valid(self, val_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0    
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)

                correct += predicted.eq(targets).sum().item()
                
                
        acc = 100.*correct/total
        content = f"Val Acc: {acc:.6f}, {correct}/{total}, val_loss={test_loss:.6f} \n"
        write_log(self.log, content)
        
        return acc

    @staticmethod
    def inference(model, img, device, transform_cfg):
        pass





if __name__ == '__main__':

    
    myclassfier = ActiveVisionEvalModel("config.yaml")
    myclassfier.train()



