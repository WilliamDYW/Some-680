import os
from torchvision import transforms
from PIL import Image
# Function to perform image flipping, rotation, and shearing
def augment_image(image_path, output_path, flip=True, rotate=True, shear=True):
    # Define transformations
    transform = transforms.Compose([])
    if flip:
        transform.transforms.append(transforms.RandomHorizontalFlip(p=1.0))
    if rotate:
        transform.transforms.append(transforms.RandomRotation(degrees=30))
    if shear:
        transform.transforms.append(transforms.RandomAffine(degrees=0, shear=30))
    # Apply transformations
    image = Image.open(image_path)
    augmented_image = transform(image)
    # Save augmented image
    augmented_image.save(output_path)
# Function to apply image flipping to all files in a directory
def flip_images_in_directory(input_directory, output_directory, keyword):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    # List all files in the input directory
    files = os.listdir(input_directory)
    for file in files:
        if keyword in file:
            # Construct input and output paths
            input_path = os.path.join(input_directory, file)
            
            # Apply image flipping
            transform = transforms.Compose([])
            output_path = os.path.join(output_directory, "aug_h" + file)
            transform.transforms.append(transforms.RandomHorizontalFlip(p=1.0))
            image = Image.open(input_path)
            augmented_image = transform(image)
            augmented_image.save(output_path)

            transform = transforms.Compose([])
            output_path = os.path.join(output_directory, "aug_v" + file)
            transform.transforms.append(transforms.RandomVerticalFlip(p=1.0))
            image = Image.open(input_path)
            augmented_image = transform(image)
            augmented_image.save(output_path)

            transform = transforms.Compose([])
            output_path = os.path.join(output_directory, "aug_r1" + file)
            transform.transforms.append(transforms.RandomRotation((90,90)))
            image = Image.open(input_path)
            augmented_image = transform(image)
            augmented_image.save(output_path)

            transform = transforms.Compose([])
            output_path = os.path.join(output_directory, "aug_r2" + file)
            transform.transforms.append(transforms.RandomRotation((180,180)))
            image = Image.open(input_path)
            augmented_image = transform(image)
            augmented_image.save(output_path)
            transform = transforms.Compose([])

            output_path = os.path.join(output_directory, "aug_r2" + file)
            transform.transforms.append(transforms.RandomRotation((270,270)))
            image = Image.open(input_path)
            augmented_image = transform(image)
            augmented_image.save(output_path)
# Example usage:
input_directory = '../2019/train/hotspot'
output_directory = '../2019/train/hotspot'
keyword_to_flip = 'MX'
flip_images_in_directory(input_directory, output_directory, keyword_to_flip)