import numpy as np
import torch
from tqdm import tqdm
import pickle
import argparse
import matplotlib
import matplotlib.pyplot as plt
import PIL
import torchvision.utils as vutils

class PatchAttacker():
    def __init__(self,model,device,image_size=(3,224,224),noise_percentage=0.03,sign=False):
        self.model = model
        self.device = device
        self.sign = sign
        self.image_size = image_size
        self.patch = self.patch_initialization(image_size=image_size,noise_percentage=noise_percentage)

    def patch_initialization(self, image_size=(3, 224, 224), noise_percentage=0.03):
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
        patch = torch.from_numpy(patch).to(self.device)
        return patch

    def mask_generation(self, patch, image_size=(3, 224, 224),random_rotate=True):
        applied_patch = torch.zeros(image_size).to(self.device)
        mask = torch.zeros(image_size).to(self.device)
        if random_rotate:
            # patch rotation
            rotation_angle = np.random.choice(4)
            patch = torch.rot90(patch,rotation_angle,[len(patch.shape)-2,len(patch.shape)-1])
        x_location, y_location = np.random.randint(low=0, high=image_size[1]-patch.shape[1]), np.random.randint(low=0, high=image_size[2]-patch.shape[2])
        applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
        mask[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = 1.0
        return applied_patch, mask, x_location, y_location

    def show_image(self,image,name):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        plt.imshow(np.clip(np.transpose(image.cpu().squeeze().numpy(), (1, 2, 0)) * std + mean, 0, 1))
        plt.axis('off')
        plt.savefig(name)

    def save_patch(self,name):
        self.show_image(self.patch,name+'.image.png')
        torch.save(self.patch,name)
        return self.patch

    def load_patch(self,name):
        self.patch = torch.load(name)
        self.patch = self.patch.to(self.device)
        return self.patch

    def train_patch(self,loader,epochs=20,max_iter=200,lr=0.075,target=None,random_init=False,random_rotate=True,clamp_bound=(-3,3)):
        if random_init:
            self.patch = self.patch_initialization(image_size=image_size,noise_percentage=noise_percentage)

        # for epoch in tqdm(range(epochs),desc = 'train patch'):
        for epoch in range(epochs):
            org_success, train_total, train_success = 0, 0, 0
            loss_tot = 0
            for (image, label) in tqdm(loader,desc = 'train patch'):
                train_total += label.shape[0]
                image = image.to(self.device)
                label = label.to(self.device)
                output = self.model(image)
                _, predicted = torch.max(output, 1)  
                org_success+=torch.sum(predicted==label).item()     

                applied_patch, mask, x_location, y_location = self.mask_generation(self.patch, image_size=self.image_size,random_rotate=random_rotate)
                applied_patch.requires_grad=True
                for it in range(max_iter):
                    # print(image.shape,mask.shape,(1-mask).shape,applied_patch.shape)
                    perturbated_image = image*(1-mask) + applied_patch*mask
                    output = self.model(perturbated_image)

                    if target is None:
                        loss = torch.nn.functional.cross_entropy(output, label, reduction='mean') 
                    else:
                        targets = torch.LongTensor([target]).expand(label.shape)
                        targets = targets.to(self.device)
                        loss = - torch.nn.functional.cross_entropy(output, targets, reduction='mean')
                    loss_tot+=loss.item()
                    loss.backward()
                    patch_grad = applied_patch.grad.clone()
                    if self.sign:
                        patch_grad.data = patch_grad.data.sign()
                    applied_patch.data = lr * patch_grad.data + applied_patch.data
                    applied_patch.data = torch.clamp(applied_patch.data, min=clamp_bound[0], max=clamp_bound[1])
                    applied_patch.grad.data.zero_()

                self.patch.data = applied_patch[:, x_location:x_location + self.patch.shape[1], y_location:y_location + self.patch.shape[2]].data
                perturbated_image = image*(1-mask) + applied_patch*mask
                output = self.model(perturbated_image)
                _, predicted = torch.max(output.data, 1)
                if target is None:
                    train_success += torch.sum(predicted!=label).data.cpu().numpy()
                else:
                    train_success += torch.sum(predicted==targets).data.cpu().numpy()

            success_rate=train_success/train_total
            org_rate = org_success/train_total
            print("in epoch {}, success rate is {}, org rate is {}, loss_tot is {}"\
                .format(epoch,success_rate,org_rate,loss_tot/max_iter))
        return self.patch

    def attack(self,image,label,max_iter=10,random_rotate=True,target=None):
        max_success = -1
        best_image = image
        for it in max_iter:
            applied_patch, mask, x_location, y_location = self.mask_generation(self.patch, image_size=self.image_size,random_rotate=random_rotate)
            perturbated_image = image*(1-mask) + applied_patch*mask
            output = self.model(perturbated_image)
            _, predicted = torch.max(output.data, 1)
            if target is None:
                success = torch.sum(predicted!=label).data.cpu().numpy()
            else:
                success = torch.sum(predicted==target).data.cpu().numpy()

            if success < max_success:
                best_image = perturbated_image
                max_success=success
        return best_image

