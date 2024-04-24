import visualkeras
import torchsummary
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import random

# Define the Convolutional Autoencoder model
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=(1,3), padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=7),


        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=7),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=(1,3), padding=(1,0)),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Define a custom dataset for generating synthetic circle images
class CircleDataset(Dataset):
    def __init__(self, x_size, y_size, num_samples):
        self.x_size = x_size
        self.y_size = y_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = np.zeros((self.y_size, self.x_size), dtype=np.float32)

        for i in range(random.randint(1, 3)):
          # Randomly generate circle parameters
          min_len = min(self.y_size, self.x_size)
          radius = random.randint(min_len // 10, min_len // 7)
          center_x = random.randint(radius, self.x_size - radius)
          center_y = random.randint(radius, self.y_size - radius)

          # Draw the circle on the image
          y, x = np.ogrid[:self.y_size, :self.x_size]
          mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
          image[mask] = 1.0

        for i in range(random.randint(1, 3)):
        #for i in range(random.randint(0, 0)):
          # Randomly generate line parameters
          min_len = min(self.y_size, self.x_size)
          is_horizontal = random.choice([True, False])
          length = self.y_size
          height = self.x_size
          if is_horizontal:
            length = self.x_size
            height = self.y_size
          width = min_len // 12 + 1
          line_start = random.randint(width, height - 1)
          line_end_1 = random.randint(0, length - 1)
          line_end_2 = random.randint(0, length - 1)
          if line_end_1 > line_end_2:
            temp = line_end_1
            line_end_1 = line_end_2
            line_end_2 = temp

          # Set the pixels for the line
          if is_horizontal:
              image[line_start - width : line_start, line_end_1 : line_end_2] = 1.0
          else:
              image[line_end_1 : line_end_2, line_start - width : line_start] = 1.0


        # Convert image to tensor and add channel dimension
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)

        return image


#################################
######### FIRST TRAINING  #######
#################################


circle_dataset = CircleDataset(x_size=300, y_size = 100, num_samples=10000)
circle_loader = DataLoader(circle_dataset, batch_size=128, shuffle=False)

# Run This if you need to load weights after the first training
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model's parameters
model = ConvAutoencoder().to(device)

train_loader = DataLoader(circle_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(circle_dataset, batch_size=128, shuffle=False)

# Initialize the model
# model = ConvAutoencoder().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
num_epochs = 40

for epoch in range(num_epochs):
    if epoch == num_epochs // 2:
      optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    train_loss = 0.0
    for data in train_loader:

        #images, _ = data
        images = data
        images = images.to(device)


        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

# Test the model
model.eval()
test_loss = 0.0

with torch.no_grad():
    for data in test_loader:
        images = data
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)
        test_loss += loss.item() * images.size(0)

test_loss /= len(test_loader.dataset)
print(f'Test Loss: {test_loss:.4f}')

# Save the weights after the first training
torch.save(model.state_dict(), '/content/drive/MyDrive/FYP/circle_line_model_300_100_smaller.pth')





######################################################################################
################## Get the scaling parpameters, i.e., mean and std ###################
######################################################################################

# Set parameters for the custom dataset
num_samples = 1000

# Create the custom dataset
circle_dataset = CircleDataset(x_size=300, y_size = 100, num_samples=10000)
# Create a data loader for the dataset
batch_size = 128
circle_loader = DataLoader(circle_dataset, batch_size=batch_size, shuffle=False)

encoder = model.encoder
# Get the output vectors from the encoder for the training data
encoder.eval()
output_vectors = []

with torch.no_grad():
    for data in circle_loader:
        images = data.to(device)
        encoded = encoder(images)
        encoded = encoded.view(encoded.size(0), -1)
        output_vectors.append(encoded)

output_vectors = torch.cat(output_vectors, dim=0)

# Calculate the mean and standard deviation of each output vector dimension
mean = torch.mean(output_vectors, dim=0)
std = torch.std(output_vectors, dim=0)
output_vectors = 0

torch.save(mean, '/content/drive/MyDrive/FYP/mean_smaller.pt')
torch.save(std, '/content/drive/MyDrive/FYP/std_smaller.pt')






#################################
######### SECOND TRAINING #######
#################################

# Copy the decoder
import copy
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model's parameters
decoder = model.decoder
decoder_s = copy.deepcopy(decoder) # Keep the original copy of the decoder
decoder_s.eval()

# Define a custom dataset for generating random vectors
class VectorDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        random_input = torch.randn((256, 1, 1), dtype=torch.float32).to(device)
        random_input = (random_input * std + mean).to(device)
        old_output = decoder_s(random_input)
        return random_input, old_output

vector_dataset = VectorDataset(num_samples=10000)
vector_loader = DataLoader(vector_dataset, batch_size=10, shuffle=True)

# Second training
# Define the custom loss function
class ImageLoss(nn.Module):
    def __init__(self):
        super(ImageLoss, self).__init__()

    def forward(self, image_1, image_2):
        # Calculate the sum of (x(1-x))^2 values
        loss = torch.sum((image_1 * (1 - image_1)) ** 2) / image_1.numel()
        loss += ((torch.sum(image_1) - torch.sum(image_2)) / image_1.numel()) ** 2

        return loss

# Create an instance of the custom loss function
loss_fn = ImageLoss()

num_epoch = 2

optimizer = optim.Adam(decoder.parameters(), lr=0.00001)

for epoch in range(num_epoch):
  # Generate random input vector following the normal distribution
  for vector, y in vector_loader:
    # Generate the output image
    output = decoder(vector)

    # Calculate the loss
    loss = loss_fn(output, y)  # Loss is the distance to all zeros

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')

torch.save(decoder.state_dict(), '/content/drive/MyDrive/FYP/circle_line_decoder_300_100_smaller_clear.pth')