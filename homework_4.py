import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Set the number of samples
n_samples = 1_000_000

def random_walk(n_steps, min_val, max_val):
    # Initialize the starting point randomly within the given range
    init_val = np.random.uniform(min_val, max_val + 1)
    
    # Generate random steps for the walk using a normal distribution
    steps = np.random.normal(0, (max_val - min_val) / 10, n_steps) # Use 10 as a design choice such that each setp will on average be 1/10 the size of the total range
    
    # Initialize the walk with the first value
    walk = np.zeros(n_steps)
    walk[0] = init_val
    print(walk)
    
    # Perform the random walk, ensuring it stays within the given bounds
    for i in range(1, n_steps):
        walk[i] = np.clip(walk[i-1] + steps[i], min_val, max_val)
    
    return walk

random_walk_F1 = random_walk(n_samples, -10000, 10000)
random_walk_F2 = random_walk(n_samples, -5000, 5000)
random_walk_alpha2 = random_walk(n_samples, -180, 180)
random_walk_F3 = random_walk(n_samples, -5000, 5000)
random_walk_alpha3 = random_walk(n_samples, -180, 180)


data = np.stack([random_walk_F1, random_walk_F2, random_walk_alpha2, random_walk_F3, random_walk_alpha3])

print(data)
print(data.shape)

def compute_generalized_force(data, l1, l2, l3, l4): 
    # Separate individual commands
    F1, F2, alpha2, F3, alpha3 = data[0,:], data[1,:], data[2,:], data[3,:], data[4,:]

    # Convert angles to radians
    alpha2_rad = np.radians(alpha2)
    alpha3_rad = np.radians(alpha3)

    B_0 = np.zeros_like(alpha2_rad)
    B_1 = np.ones_like(alpha2_rad)
    B_3 = l2 * np.ones_like(alpha2_rad)

    B_row_1 = np.stack([B_0, np.cos(alpha2_rad), np.cos(alpha3_rad)], axis=0)
    B_row_2 = np.stack([B_1, np.sin(alpha2_rad), np.sin(alpha3_rad)], axis=0)
    B_row_3 = np.stack([B_3, l1*np.sin(alpha2_rad) - l3*np.cos(alpha2_rad), l1*np.sin(alpha3_rad) - l4*np.cos(alpha3_rad)], axis=0)

    B = np.stack([B_row_1, B_row_2, B_row_3], axis=0)

    # Create F matrix
    F = np.stack([F1, F2, F3], axis=0)
    
    # Create tau matrix
    tau = np.einsum('ijk,jk->ik', B, F) 

    return tau

def compute_generalized_force_tensor(data, l1, l2, l3, l4):
    # Separate individual commands
    data = data.cpu().detach().numpy()
    F1, F2, alpha2, F3, alpha3 = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]

    # Convert angles to radians
    alpha2_rad = np.radians(alpha2)
    alpha3_rad = np.radians(alpha3)

    B_0 = np.zeros_like(alpha2_rad)
    B_1 = np.ones_like(alpha2_rad)
    B_3 = l2 * np.ones_like(alpha2_rad)

    B_row_1 = np.stack([B_0, np.cos(alpha2_rad), np.cos(alpha3_rad)], axis=0)
    B_row_2 = np.stack([B_1, np.sin(alpha2_rad), np.sin(alpha3_rad)], axis=0)
    B_row_3 = np.stack([B_3, l1*np.sin(alpha2_rad) - l3*np.cos(alpha2_rad), l1*np.sin(alpha3_rad) - l4*np.cos(alpha3_rad)], axis=0)

    B = np.stack([B_row_1, B_row_2, B_row_3], axis=0) 

    # Create F matrix
    F = np.stack([F1, F2, F3], axis=0)
    
    # Create tau matrix
    tau = np.einsum('ijk,jk->ik', B, F) 

    tau_return = torch.tensor(tau, dtype=torch.float32, requires_grad=True).to(device)

    return tau_return

# Define l1, l2, l3, and l4 from paper
l1 = -14
l2 = 14.5
l3 = -2.7
l4 = 2.7

u_1_max = 30000
u_2_max = 60000 
u_3_max = 60000
angle_2_max = 180
angle_3_max = 180
u_l_max = torch.tensor(np.stack([u_1_max, u_2_max, angle_2_max, u_3_max, angle_3_max]), dtype=torch.float32, requires_grad=False).to(device) # not sure if this is correct? Should it contain angles?
delta_u_l_max = torch.tensor([1000,1000,10,1000,10])

k_0 = 10 ** 0
k_1 = 10 ** 0
k_2 = 10 ** -1
k_3 = 10 ** -7
k_4 = 10 ** -7
k_5 = 10 ** -1

alpha_bar_1_c = 100
alpha_bar_0_c = 80
alpha_sub_1_c = -80
alpha_sub_0_c = -100

tau = compute_generalized_force(data, l1, l2, l3, l4)
X_encoder, y_encoder, y_decoder = np.transpose(tau), np.transpose(data), np.transpose(tau)


# Define the model
class EncoderDecoderModel(nn.Module):
    def __init__(self):
        super(EncoderDecoderModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        ).to(device)

    def forward(self, x):
        latent_space =  self.encoder(x)
        decoded_output = self.decoder(latent_space)

        return latent_space, decoded_output
    
model = EncoderDecoderModel()

# Loss function and optimizer
loss_fn_encoder = nn.MSELoss()  # Mean Square Error
loss_fn_decoder = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

split_ratio = 0.8
n_train = int(X_encoder.shape[0] * split_ratio)
X_train = X_encoder[:n_train, :]
y_train = y_encoder[:n_train, :]
X_test = X_encoder[n_train:, :]
y_test = y_encoder[n_train:, :]

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)  # Move data to device
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)  # Move data to device
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)  # Move data to device
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)  # Move data to device

# Regularization: Normalize Data using Standardization
mean_X_train = X_train.mean(dim=0)
std_X_train = X_train.std(dim=0)
X_train = (X_train - mean_X_train) / std_X_train

y_train_before_norm = y_train[0]
y_test_before_norm = y_test[0]

mean_y_train = y_train.mean(dim=0)
std_y_train = y_train.std(dim=0)
y_train = (y_train - mean_y_train) / std_y_train

mean_X_test = X_test.mean(dim=0)
std_X_test = X_test.std(dim=0)
X_test = (X_test - mean_X_test) / std_X_test

mean_y_test = y_test.mean(dim=0)
std_y_test = y_test.std(dim=0)
y_test = (y_test - mean_y_test) / std_y_test

train_scale_factor = y_train[0]/y_train_before_norm
test_scale_factor = y_test[0]/y_test_before_norm

u_l_max_train = u_l_max * train_scale_factor
u_l_max_test = u_l_max * test_scale_factor

# Training parameters
n_epochs = 20  # Number of epochs to run
batch_size = 1024 # Size of each batch
print(len(X_train))
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_loss = np.inf
best_weights = None
history_eval = []
history_loss_0 = []
history_loss_1 = []
history_loss_2 = []
history_loss_3 = []
history_loss_4 = []
history_loss_5 = []
history_loss_6 = []
history_train = []

# Hold Previous u
prev_u = 0

# Training loop
for epoch in range(n_epochs):
    model.train()
    print(f"Epoch {epoch+1}/{n_epochs}")
    for start in batch_start:

        # Take a batch
        X_batch = X_train[start:start+batch_size]
        X_shift = X_train[start+1:start+1+batch_size]
        y_batch = y_train[start:start+batch_size, :]
        
        # Forward pass
        u_pred, y_pred_decoder = model.forward(X_batch)
        u_shift = model.forward(X_shift)[0]
  
        y_pred = compute_generalized_force_tensor(u_pred, l1, l2, l3, l4)
        ground_truth = ground_truth = compute_generalized_force_tensor(y_batch, l1, l2, l3, l4)

        # Calculate Loss
        loss_0 = loss_fn_encoder(y_pred, ground_truth)
 
        loss_1 = loss_fn_decoder(y_pred_decoder, X_batch)

        loss_2 = 0 
        u_pred_scaled = u_pred * std_y_train + mean_y_train
        excess_force = torch.abs(u_pred_scaled) - u_l_max
        penalty = torch.clamp(excess_force, min=0)
        loss_2 = torch.mean(penalty)

        loss_3 = 0 
        if(u_shift.shape[0] == u_pred.shape[0]):
            temp = torch.abs(u_pred_scaled - u_shift) - delta_u_l_max
            penalty = torch.clamp(temp, min=0)
            loss_3 = torch.mean(penalty)

        loss_4 = torch.mean((abs(u_pred[:,0]) ** 3/2) + (abs(u_pred[:,1]) ** 3/2) + (abs(u_pred[:,3]) ** 3/2))
        
        loss_5_sub = torch.sum(((u_pred_scaled[:,2] < alpha_sub_1_c) & (u_pred_scaled[:,2] > alpha_sub_0_c)).int() + ((u_pred_scaled[:,4] < alpha_sub_1_c) & (u_pred_scaled[:,4] > alpha_sub_0_c)).int())
        loss_5_bar = torch.sum(((u_pred_scaled[:,2] < alpha_bar_1_c) & (u_pred_scaled[:,2] > alpha_bar_0_c)).int() + ((u_pred_scaled[:,4] < alpha_bar_1_c) & (u_pred_scaled[:,4] > alpha_bar_0_c)).int())

        loss_5 = loss_5_sub + loss_5_bar

        loss = k_0 * loss_0 + k_1 * loss_1 + k_2 * loss_2 + k_3 * loss_3 + k_4 * loss_4 + k_5 * loss_5

        history_train.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress for each batch
        # print(f" epoch: {epoch+1}/{n_epochs},  Batch {start//batch_size+1}/{len(batch_start)}, Encoder Loss: {loss_encoder.item():.6f}, Decoder Loss: {loss_decoder.item():.6f}")
    print(f" Epoch: {epoch+1}/{n_epochs},  Batch {start//batch_size+1}/{len(batch_start)}, Loss: {loss.item():.6f}") 

    
    # Evaluate accuracy at the end of each epoch
    model.eval()
    with torch.no_grad():

        u_pred, y_pred_decoder = model.forward(X_test)
        X_shift = X_test[1:2000]
        u_shift = model.forward(X_shift)[0]
        y_pred = compute_generalized_force_tensor(u_pred, l1, l2, l3, l4)
        ground_truth = ground_truth = compute_generalized_force_tensor(y_test, l1, l2, l3, l4)

        # Calculate Loss
        loss_0 = loss_fn_encoder(y_pred, ground_truth)

        loss_1 = loss_fn_decoder(y_pred_decoder, X_test)

        loss_2 = 0

        u_pred_scaled = u_pred * std_y_train + mean_y_train
        excess_force = torch.abs(u_pred_scaled) - u_l_max
        penalty = torch.clamp(excess_force, min=0)
        loss_2 = torch.mean(penalty)

        loss_3 = 0
        if(u_shift.shape[0] == 1024):
            temp = torch.abs(u_pred_scaled - u_shift) - delta_u_l_max
            penalty = torch.clamp(temp, min=0)
            loss_3 = torch.mean(penalty)

        loss_4 = torch.mean((abs(u_pred[:,0]) ** 3/2) + (abs(u_pred[:,1]) ** 3/2) + (abs(u_pred[:,3]) ** 3/2))
        
        loss_5_sub = torch.sum(((u_pred_scaled[:,2] < alpha_sub_1_c) & (u_pred_scaled[:,2] > alpha_sub_0_c)).int() + ((u_pred_scaled[:,4] < alpha_sub_1_c) & (u_pred_scaled[:,4] > alpha_sub_0_c)).int())
        loss_5_bar = torch.sum(((u_pred_scaled[:,2] < alpha_bar_1_c) & (u_pred_scaled[:,2] > alpha_bar_0_c)).int() + ((u_pred_scaled[:,4] < alpha_bar_1_c) & (u_pred_scaled[:,4] > alpha_bar_0_c)).int())
        loss_5 = loss_5_sub + loss_5_bar

        loss_test = (k_0 * loss_0 + k_1 * loss_1 + k_2 * loss_2 + k_3 * loss_3 + k_4 * loss_4 + k_5 * loss_5).item()

        history_eval.append(loss_test)
        history_loss_0.append(loss_0)
        history_loss_1.append(loss_1)
        history_loss_2.append(loss_2)
        history_loss_3.append(loss_3)
        history_loss_4.append(loss_4)
        history_loss_5.append(loss_5)

        if loss_test < best_loss:
            best_loss = loss_test
            best_weights = copy.deepcopy(model.state_dict())


# Restore model and return best accuracy
model.load_state_dict(best_weights)

print(f"Best MSE: {best_loss:.2f}")
print(f"Best RMSE: {np.sqrt(best_loss):.2f}")

plt.figure(1)
plt.plot(history_eval)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs in eval")

plt.figure(2)
plt.plot(history_train)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss in train")

plt.figure(3)
plt.plot(history_loss_0)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs Loss 0")

plt.figure(4)
plt.plot(history_loss_1)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs Loss 1")

plt.figure(5)
plt.plot(history_loss_2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs Loss 2")

plt.figure(6)
plt.plot(history_loss_3)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs Loss 3")

plt.figure(7)
plt.plot(history_loss_4)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs Loss 4")

plt.figure(8)
plt.plot(history_loss_5)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs Loss 5")

plt.show()