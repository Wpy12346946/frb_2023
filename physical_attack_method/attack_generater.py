import os
import torch
from data import get_dataloader, store_dataloader
from config import config
from utils import load_mask


def generate_black_mask():
    test_loader = get_dataloader('./data/gallery', config['batch_size'])
    labels = sorted(os.listdir('./data/gallery'))

    for batch_idx, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        mask = load_mask('facemask').cuda()
        X_adv = X * (1 - mask)
        y_adv = y

    adv_dataset = torch.utils.data.TensorDataset(X_adv.cpu().detach(), y_adv.cpu().detach())
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False)
    store_dataloader(adv_loader, './data/test', labels)


def generate_white_mask():
    test_loader = get_dataloader('./data/gallery', config['batch_size'])
    labels = sorted(os.listdir('./data/gallery'))

    for batch_idx, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        mask = load_mask('facemask').cuda()
        X_adv = X * (1 - mask) + mask
        y_adv = y

    adv_dataset = torch.utils.data.TensorDataset(X_adv.cpu().detach(), y_adv.cpu().detach())
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False)
    store_dataloader(adv_loader, './data/test', labels)


def generate_mask_with_options(name):
    test_loader = get_dataloader('./data/gallery', config['batch_size'])
    labels = sorted(os.listdir('./data/gallery'))

    for batch_idx, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        mask = load_mask('facemask').cuda()
        filler = load_mask(name).cuda()
        X_adv = X * (1 - mask) + filler
        y_adv = y

    adv_dataset = torch.utils.data.TensorDataset(X_adv.cpu().detach(), y_adv.cpu().detach())
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False)
    store_dataloader(adv_loader, './data/test', labels)


def generate_cheater():
    test_loader = get_dataloader('./data/gallery', config['batch_size'])
    labels = sorted(os.listdir('./data/gallery'))

    for batch_idx, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        mask = load_mask('sticker').cuda()
        X_adv = X * (mask)
        y_adv = y

    adv_dataset = torch.utils.data.TensorDataset(X_adv.cpu().detach(), y_adv.cpu().detach())
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False)
    store_dataloader(adv_loader, './data/test', labels)


def generate_black_glasses():
    test_loader = get_dataloader('./data/gallery', config['batch_size'])
    labels = sorted(os.listdir('./data/gallery'))

    for batch_idx, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        mask = load_mask('eyeglass').cuda()
        X_adv = X * (1 - mask)
        y_adv = y

    adv_dataset = torch.utils.data.TensorDataset(X_adv.cpu().detach(), y_adv.cpu().detach())
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False)
    store_dataloader(adv_loader, './data/test', labels)


def generate_white_glasses():
    test_loader = get_dataloader('./data/gallery', config['batch_size'])
    labels = sorted(os.listdir('./data/gallery'))

    for batch_idx, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        mask = load_mask('eyeglass').cuda()
        X_adv = X * (1 - mask) + mask
        y_adv = y

    adv_dataset = torch.utils.data.TensorDataset(X_adv.cpu().detach(), y_adv.cpu().detach())
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False)
    store_dataloader(adv_loader, './data/test', labels)


def generate_glasses_with_options(name):
    test_loader = get_dataloader('./data/gallery', config['batch_size'])
    labels = sorted(os.listdir('./data/gallery'))

    for batch_idx, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        mask = load_mask('eyeglass').cuda()
        filler = load_mask(name).cuda()
        X_adv = X * (1 - mask) + filler
        y_adv = y

    adv_dataset = torch.utils.data.TensorDataset(X_adv.cpu().detach(), y_adv.cpu().detach())
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False)
    store_dataloader(adv_loader, './data/test', labels)


def generate_black_sticker():
    test_loader = get_dataloader('./data/gallery', config['batch_size'])
    labels = sorted(os.listdir('./data/gallery'))

    for batch_idx, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        mask = load_mask('sticker').cuda()
        X_adv = X * (1 - mask)
        y_adv = y

    adv_dataset = torch.utils.data.TensorDataset(X_adv.cpu().detach(), y_adv.cpu().detach())
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False)
    store_dataloader(adv_loader, './data/test', labels)


def generate_white_sticker():
    test_loader = get_dataloader('./data/gallery', config['batch_size'])
    labels = sorted(os.listdir('./data/gallery'))

    for batch_idx, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        mask = load_mask('sticker').cuda()
        X_adv = X * (1 - mask) + mask
        y_adv = y

    adv_dataset = torch.utils.data.TensorDataset(X_adv.cpu().detach(), y_adv.cpu().detach())
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False)
    store_dataloader(adv_loader, './data/test', labels)


def generate_stickers_with_options(name):
    test_loader = get_dataloader('./data/gallery', config['batch_size'])
    labels = sorted(os.listdir('./data/gallery'))

    for batch_idx, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        mask = load_mask('sticker').cuda()
        filler = load_mask(name).cuda()
        X_adv = X * (1 - mask) + filler
        y_adv = y

    adv_dataset = torch.utils.data.TensorDataset(X_adv.cpu().detach(), y_adv.cpu().detach())
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False)
    store_dataloader(adv_loader, './data/test', labels)


def main():
    # Please select the attack form you need
    # Form1: mask
    # Form1.1: black mask
    # generate_black_mask()
    # Form1.2: white mask
    # generate_white_mask()
    # Form1.3: other masks
    # Parameters can be: "cheater_mask" or "colorful_mask" or "n95_mask" or "medical_mask"
    # generate_mask_with_options("cheater_mask")
    # Form2: glasses
    # Form2.1: black glasses
    # generate_black_glasses()
    # Form2.2: white mask
    # generate_white_glasses()
    # Form1.3: other masks
    # Parameters can be: "colorful_glass" or "skin_glass"
    generate_glasses_with_options("colorful_glass")
    # Form3: sticker
    # Form3.1: black sticker
    # generate_black_sticker()
    # Form3.2: white sticker
    # generate_white_sticker()
    # Form3.3: other stickers
    # Parameters can be: "colorful_sticker" or "skin_sticker" or "eyes_sticker"
    # generate_stickers_with_options("colorful_sticker")


if __name__ == '__main__':
    main()
