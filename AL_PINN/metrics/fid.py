import numpy as np
import torch
from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
from scipy.linalg import sqrtm
import os
from PIL import Image
from tqdm import tqdm

def calculate_fid(model, images1, images2):
    # Move images to the device where model is
    images1 = images1.to(model.device)
    images2 = images2.to(model.device)
    
    # Calculate activations
    with torch.no_grad():
        act1 = model(images1)
        act2 = model(images2)

    # Move activations back to CPU for numpy operations
    act1 = act1.cpu().numpy()
    act2 = act2.cpu().numpy()
    
    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # Calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def preprocess_images(image_paths, target_size=(299, 299)):
    images = []
    for img_path in tqdm(os.listdir(image_paths)):
        img = Image.open(os.path.join(image_paths,img_path)).convert('RGB')
        img = TF.resize(img, target_size)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images.append(img)
    images = torch.stack(images)
    return images

def load_and_calculate_fid(dataset1, dataset2):
    # Load pre-trained InceptionV3 model
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Modify the model to return features before the final fully connected layer
    model.eval()  # Set model to evaluation mode
    model.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(model.device)

    # Assume dataset1 and dataset2 are lists of image paths
    images1 = preprocess_images(dataset1)
    images2 = preprocess_images(dataset2)
    fid = calculate_fid(model, images1, images2)
    print(f"FID: {fid}")


if __name__ == "__main__":
    dataset1 = '/data1/lzengaf/fyp/Validation_Dataset/validation_result_Search_5'
    dataset2 = '/data1/lzengaf/fyp/Validation_Dataset/validation_result_Search_7'
    load_and_calculate_fid(dataset1, dataset2)
