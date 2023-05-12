# torch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# other
from tqdm import tqdm

# utils
from network import CNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

"""
Data Processing
"""

def download_mnist_datasets():
    """Download handwritten digit MNIST dataset"""
    train_data = datasets.MNIST(
        root="main/data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    validation_data = datasets.MNIST(
        root="main/data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_data, validation_data

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

"""
Model Training
"""
def train(model, dataloader, loss_fn, optimizer, device, epochs):
  for i in tqdm(range(epochs), "Training model..."):
    print(f"Epoch {i + 1}")

    train_epoch(model, dataloader, loss_fn, optimizer, device)

    print (f"----------------------------------- \n")
  
  print("---- Finished Training ----")
  

def train_epoch(model, dataloader, loss_fn, optimizer, device):
  for x, y in dataloader:
    x, y = x.to(device), y.to(device)

    # calculate loss
    pred = model(x)
    loss = loss_fn(pred, y)

    # backprop and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  print(f"Loss: {loss.item()}") 
    
if __name__ == "__main__":
  # assign device
  if torch.cuda.is_available():
      device = "cuda"
  else:
      device = "cpu"

  print(f"Using {device} device. \n")

  # download dataset and create dataloader
  train_data, _ = download_mnist_datasets()
  train_dataloader = create_data_loader(train_data, BATCH_SIZE)

  # init model to device
  model = CNNetwork().to(device)
  print(f"Model Info: \n {model}")

  # init loss function and optimizer
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


  # train model
  train(model, train_dataloader, loss_fn, optimizer, device, EPOCHS)

  # save model
  torch.save(model.state_dict(), "mnist_model.pth")
  print("Trained feed forward net saved at mnist_model.pth")

  