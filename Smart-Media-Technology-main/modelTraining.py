"""
This script trains the model and saves the trained parameters to a file.

Run this from terminal to start Tensorboard:
    tensorboard --logdir=runs
Note: Safari won't open the page, you need chrome/firefox...!

You dont need to run this if you just want to turn your hand gestures into midi.
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import model_class
import time
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

#### HYPERPARAMETERS ###
num_epochs = 100
learning_rate = 0.001
batch_size = 32

# Create writer for logging
writer = SummaryWriter(log_dir=f'runs/{time.asctime()}')

# Custom Dataset class for PyTorch
class HandGestureDataset(Dataset):
    def __init__(self, data):
        num_landmarks = 21  # Number of landmarks
        self.X = []
        for row in data.itertuples(index=False):
            landmarks = []
            for i in range(1, num_landmarks + 1):
                x, y, z = map(float, row[i].split(','))
                landmarks.extend([x, y, z])
            self.X.append(landmarks)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(data['gesture'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Path to the directory containing the CSV files
data_directory = 'Smart-Media-Technology-main/hand_data/'

# Get a list of all CSV files in the directory
csv_files = glob.glob(data_directory + 'hand_gesture_data_*.csv')

# Read and concatenate all CSV files into a single DataFrame
data_frames = [pd.read_csv(file) for file in csv_files]
data = pd.concat(data_frames, ignore_index=True)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create DataLoader for PyTorch
train_dataset = HandGestureDataset(train_data)
test_dataset = HandGestureDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function and optimizer
model = model_class.HandGestureModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
epoch_loss_train, epoch_loss_test = [], []  # store epoch loss
epoch_acc_train, epoch_acc_test = [], [] # store epoch accuracy
for epoch in range(num_epochs):

    #TRAIN
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    #epoch_loss_train.append(loss.item())
    writer.add_scalar('Loss/Train', loss.item(), epoch)

    #TEST
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        #epoch_loss_test.append(loss.item())
        writer.add_scalar('Loss/Test', loss.item(), epoch)

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'hand_gesture_model.pth')
print("Model training completed and saved.")

landmarks, labels = next(iter(train_loader))
writer.add_graph(model, landmarks)

writer.add_hparams({"training files": str(csv_files),
                    "num samples": train_dataset.__len__(),
                    "epochs": num_epochs,
                    "batch size": batch_size,
                    "learning rate": learning_rate},
                   {"loss": loss.item()})
writer.flush()
writer.close()