import os
import wget
import zipfile
import json
import imageio 
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def dataset_download(DRIVE_DIR: str, download: bool = False):
    r"""Try to download dataset and save it to `drive_dir`.
    
    Expected repo structure:
    drive_dir/
      | tiny_nerf_data.npz
      | data/
        | nerf_example_data.zip
        | nerf_llff_data/
        | nerf_synthetic/
    """
    if not download:  # If not first time download
        return

    os.makedirs(DRIVE_DIR, exist_ok=True)
    wget.download(
        "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz",
        out=DRIVE_DIR,
    )

    DATA_DIR = os.path.join(DRIVE_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    wget.download(
        "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip",
        out=DATA_DIR,
    )

    # Extract zip file:
    PATH_TO_ZIP = os.path.join(DATA_DIR, "nerf_example_data.zip")
    with zipfile.ZipFile(PATH_TO_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)


def load_blender_data(BASE_DIR: str, dev_res: int = 4, skip: int = 4, dtype=np.float32, device="cuda"):
    r"""The function loads images and gets corresponding transform matrices from 
    json files. We skip some of the images and reduce the resolution of input
    images so that we could train the model on Google Colab with reasonable
    times, but the performance would definitely drop.

    Reference: https://github.com/yenchenlin/nerf-pytorch

    Args:
      BASE_DIR: path to the data folder
      dev_res: constant used for reducing the image resolution
      skip: constant used for picking part of the images

    Returns:
      imgs (torch.tensor, torch.float32): resized whole images set (4D np.array)
      poses (torch.tensor, torch.float32): transform matrix (4D np.array), which
        also represents the camera poses correspond to each image
      [H, W, focal]: length and width of the image, and camera focal length
      i_split: [train_index, val_index, test_index] (it's a list of np.array)

    Transform json details:
      camera_angle_x: The FOV in x dimension
      frames: List of dictionaries that contain the camera transform matrices 
              for each image.
              [{'file_path':..., 'rotation':..., 'transform_matrix':...}]
    """

    # Read json files for 'train', 'val' and 'test' dataset, details covered above.
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(BASE_DIR, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    # Get all the images and transform matrix from the folder
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []

        # Skip some images to make the dataset small enough to train on Google
        # Colab, the default skip value is set to 4, which downsize the total 
        # number of images from 400 to 100 (for LEGO)
        if s is not 'train':
            for frame in meta['frames'][::skip]:
                fname = os.path.join(BASE_DIR, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname)) # (800, 800, 4)
                poses.append(np.array(frame['transform_matrix'])) # append 4*4 matrix
        else:
            for frame in meta['frames']: # load all training images
                  fname = os.path.join(BASE_DIR, frame['file_path'] + '.png')
                  imgs.append(imageio.imread(fname)) # (800, 800, 4)
                  poses.append(np.array(frame['transform_matrix'])) # append 4*4 matrix

        # Transfer RGBA values from 0-255 to 0-1, notice that we have 4 channels
        imgs = (np.array(imgs) / 255.).astype(dtype=dtype)
        poses = np.array(poses).astype(dtype=dtype)

        # Indicators of seperation between train/val/test dataset
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    # Calculate the focal length of the camera
    H, W = imgs[0].shape[:2] # 800, 800
    camera_angle_x = float(meta['camera_angle_x']) # train/val/test use the same camera
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    # [train_index, val_index, test_index]
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    # Concatenate train/val/test set to one large set, and we'll use
    # i_split to access different parts later
    imgs = np.concatenate(all_imgs, 0) # (N, 800, 800, 4)
    poses = np.concatenate(all_poses, 0) # (N, 4, 4)

    # Reduce image resolution so that we could train on Colab
    if dev_res > 1:
        if H % dev_res != 0:
            raise ValueError(
                f"""The value H is not dividable by dev_res. Please select an
                    appropriate value.""")
        H = int(H // dev_res)
        W = int(W // dev_res)
        focal = focal / float(dev_res)
        imgs_reduce = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_reduce[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_reduce

    # Print data shape
    print("Images shape: ", imgs.shape)
    print("Poses shape: ", poses.shape)

    # Convert useful variables to tensors
    imgs = torch.from_numpy(np.asarray(imgs)).to(device=device, dtype=torch.float32)
    poses = torch.from_numpy(np.asarray(poses)).to(device=device, dtype=torch.float32)
    focal = torch.from_numpy(np.asarray(focal)).to(device=device, dtype=torch.float32)

    return imgs, poses, [H, W, focal], i_split
