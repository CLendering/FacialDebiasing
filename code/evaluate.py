"""
In this file the evaluation of the network is done
"""

import torch

import vae_model
import argparse
from setup import config
from torch.utils.data import DataLoader
from dataset import make_eval_loader

from setup import config
import utils

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def eval_model(model, data_loader):
    """
    perform evaluation of a single epoch
    """

    model.eval()

    count = 0
    correct_count = 0

    for i, batch in enumerate(data_loader):
        # print(f"batch number {i}")
        count += 1
        images_list, _, _ , _= batch

        for images in images_list:
            if len(images.shape) == 5:
                images = images.squeeze(dim=0)

            batch_size = images.size(0)
            # print(f"size: {batch_size}")

            images = images.to(config.device)

            pred = model.forward_eval(images)

            
            if (pred > 0).any():
                # print("CORRECT")
                correct_count += 1
                break

    print(f"Amount of labels:{count}, Correct labels:{correct_count}")

    return (correct_count/count)*100

def interpolate_images(model, amount):
    eval_loader: DataLoader = make_eval_loader(
        batch_size=config.batch_size,
        nr_windows=config.eval_nr_windows
    )

    image_1 = []
    image_2 = []
    for i, batch in enumerate(eval_loader):
        _, _, _, img = batch

        if i == 0:
            image_1 = img.view(3, 64, 64)

        if i == 1:
            image_2 = img.view(3, 64, 64)

        if i > 1:
            break
    
    images = torch.stack((image_1, image_2)).to(config.device)
    recon_images = model.interpolate(images, amount)

    fig=plt.figure(figsize=(16, 16))

    ax = fig.add_subplot(2, 2, 1)
    grid = make_grid(images[1,:,:,:].view(1,3,64,64), 1)
    plt.imshow(grid.permute(1,2,0).cpu())
    utils.remove_frame(plt)

    ax = fig.add_subplot(2, 2, 2)
    grid = make_grid(images[0,:,:,:].view(1,3,64,64), 1)
    plt.imshow(grid.permute(1,2,0).cpu())
    utils.remove_frame(plt)

    ax = fig.add_subplot(2, 1, 2)
    grid = make_grid(recon_images.reshape(amount,3,64,64), amount)
    plt.imshow(grid.permute(1,2,0).cpu())
    utils.remove_frame(plt)
    
    plt.show()



def main():
    gender_list = [[], ["Female"], ["Male"], ["Female"], ["Male"]]
    skin_list = [[], ["lighter"], ["lighter"], ["darker"], ["darker"]]
    name_list = ["all", "dark man", "dark female", "light man", "light female"]

    # Load model
    model = vae_model.Db_vae(z_dim=config.zdim, device=config.device).to(config.device)

    if not config.path_to_model:
        raise Exception('Load up a model using --path_to_model')

    model.load_state_dict(torch.load(f"results/{config.path_to_model}/model.pt"))
    model.eval()

    # interpolate_images(model, 20)
    # return
    
    losses = []
    accs = []
    for i in range(5):
        eval_loader: DataLoader = make_eval_loader(
            batch_size=config.batch_size,
            filter_exclude_skin_color=skin_list[i],
            filter_exclude_gender=gender_list[i],
            nr_windows=config.eval_nr_windows
        )

        acc = eval_model(model, eval_loader)

        print(f"{name_list[i]} => acc:{acc:.3f}")

        with open(f"results/{config.path_to_model}/evaluation_results.txt", 'a+') as wf:
            wf.write(f"{name_list[i]} => acc:{acc:.3f}\n")

        accs.append(acc)

    print(f"Accuracy => all:{accs[0]:.3f}, dark male: {accs[1]:.3f}, dark female: {accs[2]:.3f}, white male: {accs[3]:.3f}, white female: {accs[4]:.3f}")

    print(f"Variance => {(torch.Tensor(accs[1:5])).var().item():.3f}")

    with open(f"results/{config.path_to_model}/evaluation_results.txt", 'a+') as wf:
        wf.write(f"\nVariance => {(torch.Tensor(accs[1:5])).var().item():.3f}\n")


#################### NEGATIVE SAMPLING ####################
    # eval_loader: DataLoader = make_eval_loader(
    #         batch_size=config.batch_size,
    #         filter_exclude_skin_color=skin_list[i],
    #         filter_exclude_gender=gender_list[i],
    #         nr_windows=config.eval_nr_windows
    #     )

    # loss, acc = eval_model(model, eval_loader)
    # with open(f"results/{config.path_to_model}/evaluation_results.txt", 'a+') as wf:
    #     wf.write(f"\nNegative score => loss:{loss:.3f}, acc:{acc:.3f}\n")

    return

if __name__ == "__main__":
    print("start evaluation")

    main()
