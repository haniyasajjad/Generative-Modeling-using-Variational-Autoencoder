"""
Fashion-MNIST Data Preprocessing Script
Runs locally to prepare data before uploading to Colab
"""

import numpy as np
import gzip
import urllib.request
import os
from pathlib import Path
import pickle

def download_fashion_mnist(data_dir='./data/fashion'):
    """Download Fashion-MNIST dataset"""
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = {
        'train-images-idx3-ubyte.gz': 'train_images',
        'train-labels-idx1-ubyte.gz': 'train_labels',
        't10k-images-idx3-ubyte.gz': 'test_images',
        't10k-labels-idx1-ubyte.gz': 'test_labels'
    }
    
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    for filename in files.keys():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
            print(f"Downloaded {filename}")
        else:
            print(f"{filename} already exists, skipping download")

def load_mnist_images(filename):
    """Load images from Fashion-MNIST format"""
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    """Load labels from Fashion-MNIST format"""
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def preprocess_data(data_dir='./data/fashion', output_dir='./preprocessed_data'):
    """
    Load, preprocess, and save Fashion-MNIST data
    Preprocessing includes:
    - Loading raw data
    - Normalizing to [0, 1]
    - Splitting training into train/validation (90/10 split)
    - Saving as numpy arrays for easy upload
    """
    
    print("Loading raw data...")
    # Load training data
    train_images = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    
    # Load test data
    test_images = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_labels = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    
    print(f"Original shapes:")
    print(f"Train images: {train_images.shape}, Train labels: {train_labels.shape}")
    print(f"Test images: {test_images.shape}, Test labels: {test_labels.shape}")
    
    # Normalize to [0, 1]
    print("\nNormalizing images...")
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    # Split training into train and validation (90/10)
    print("\nSplitting into train/validation...")
    np.random.seed(42)
    indices = np.random.permutation(len(train_images))
    
    train_size = int(0.9 * len(train_images))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    X_train = train_images[train_idx]
    y_train = train_labels[train_idx]
    X_val = train_images[val_idx]
    y_val = train_labels[val_idx]
    X_test = test_images
    y_test = test_labels
    
    print(f"\nFinal splits:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save preprocessed data
    print(f"\nSaving preprocessed data to {output_dir}...")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save metadata
    metadata = {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'image_shape': (28, 28),
        'num_classes': 10,
        'class_names': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
        'normalization': '[0, 1]'
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\nPreprocessing complete!")
    print(f"Files saved in {output_dir}/")
    print("\nYou can now upload these files to Google Colab:")
    print("- X_train.npy, y_train.npy")
    print("- X_val.npy, y_val.npy")
    print("- X_test.npy, y_test.npy")
    print("- metadata.pkl")
    
    # Print statistics
    print("\nData Statistics:")
    print(f"Train set mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    print(f"Val set mean: {X_val.mean():.4f}, std: {X_val.std():.4f}")
    print(f"Test set mean: {X_test.mean():.4f}, std: {X_test.std():.4f}")
    
    return metadata

if __name__ == "__main__":
    print("=" * 60)
    print("Fashion-MNIST Data Preprocessing")
    print("=" * 60)
    
    # Download data
    download_fashion_mnist()
    
    # Preprocess data
    metadata = preprocess_data()
    
    print("\n" + "=" * 60)
    print("Done! Ready for Colab training.")
    print("=" * 60)
