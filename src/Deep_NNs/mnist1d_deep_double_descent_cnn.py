import pylab as plt
import numpy as np
import time
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Figure style (loading from the web to make running in a Colab easier)
plt.style.use('https://github.com/greydanus/mnist1d/raw/master/notebooks/mpl_style.txt')

# Load MNIST-1D
# (loading from the web to make running in a Colab easier)

from urllib.request import urlopen
import pickle

url = 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
data = pickle.load(urlopen(url))

print(data['x'].shape)
print(data['x_test'].shape)

# The experiment works with the default frozen dataset as well,
# but having a larger test set makes the test performance curve smoother.
# So here we generate MNIST-1D with 4000 + 12000 samples.
# Requires installed mnist1d package (pip install mnist1d)

# If running in Google Colab, you can uncomment this line:
# !pip install mnist1d

from data import make_dataset, get_dataset_args

args = get_dataset_args()
args.num_samples = 16_000
args.train_split = 0.25

data = make_dataset(args)

print(data['x'].shape)
print(data['x_test'].shape)

# Add 15% noise to training labels

import copy

data_with_label_noise = copy.deepcopy(data)

for i in range(len(data['y'])):
    if np.random.random_sample() < 0.15:
        data_with_label_noise['y'][i] = np.random.randint(0, 10)

def fit_model(model, data, n_epoch=500):
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    x_train = torch.tensor(data["x"].astype("float32"))
    y_train = torch.tensor(data["y"].transpose().astype("long"))
    x_test = torch.tensor(data["x_test"].astype("float32"))
    y_test = torch.tensor(data["y_test"].astype("long"))

    data_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=100, shuffle=True
    )

    # Training progress monitoring variables
    best_test_error = float('inf')
    patience = 20  # Early stopping patience value
    patience_counter = 0
    
    # For recording training information for each epoch
    training_history = {
        'train_loss': [],
        'test_loss': [],
        'train_error': [],
        'test_error': [],
        'epochs': []
    }
    
    print(f"Starting training with {n_epoch} epochs...")
    
    # Train the model
    for epoch in range(n_epoch):
        # Training phase
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for i, batch in enumerate(data_loader):
            x_batch, y_batch = batch
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_function(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Evaluate model performance every epoch
        model.eval()
        with torch.no_grad():
            pred_train = model(x_train)
            pred_test = model(x_test)
            _, predicted_train_class = torch.max(pred_train.data, 1)
            _, predicted_test_class = torch.max(pred_test.data, 1)
            errors_train = 100 * (predicted_train_class != y_train).float().mean()
            errors_test = 100 * (predicted_test_class != y_test).float().mean()
            losses_train = loss_function(pred_train, y_train).item()
            losses_test = loss_function(pred_test, y_test).item()
        
        # Record training history
        training_history['epochs'].append(epoch + 1)
        training_history['train_loss'].append(losses_train)
        training_history['test_loss'].append(losses_test)
        training_history['train_error'].append(errors_train)
        training_history['test_error'].append(errors_test)
        
        # Show detailed progress every 10 epochs, or for the last few epochs
        if (epoch + 1) % 10 == 0 or epoch >= n_epoch - 5:
            # Show training progress
            avg_loss = total_loss / num_batches
            print(
                f"Epoch {epoch + 1:5d}/{n_epoch}, "
                f"avg loss: {avg_loss:.6f}, "
                f"train loss: {losses_train:.6f}, train error: {errors_train:3.2f}%, "
                f"test loss: {losses_test:.6f}, test error: {errors_test:3.2f}%"
            )
            
            # Early stopping check
            if errors_test < best_test_error:
                best_test_error = errors_test
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered! Test error did not improve for {patience} epochs")
                break
        else:
            # Show simple progress
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1:5d}/{n_epoch}, avg loss: {avg_loss:.6f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred_train = model(x_train)
        pred_test = model(x_test)
        _, predicted_train_class = torch.max(pred_train.data, 1)
        _, predicted_test_class = torch.max(pred_test.data, 1)
        errors_train = 100 * (predicted_train_class != y_train).float().mean()
        errors_test = 100 * (predicted_test_class != y_test).float().mean()
        losses_train = loss_function(pred_train, y_train).item()
        losses_test = loss_function(pred_test, y_test).item()
    
    print(
        f"Training completed! Final results - "
        f"train loss: {losses_train:.6f}, train error: {errors_train:3.2f}%, "
        f"test loss: {losses_test:.6f}, test error: {errors_test:3.2f}%"
    )

    return errors_train, errors_test, losses_train, losses_test, training_history

# Define CNN architecture

def get_model_cnn(channels):
    return nn.Sequential(
        nn.Conv1d(1, channels, 5, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv1d(channels, channels, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv1d(channels, channels, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(channels * 5, 10),
    )

# With label noise
# Need to reshape the data arrays to make them compatible with the conv1d layers

ddata = data_with_label_noise.copy()
ddata['x'] = data['x'][:, np.newaxis, :]
ddata['x_test'] = data['x_test'][:, np.newaxis, :]

channels = np.concatenate((np.arange(2, 30, 2), np.arange(30, 101, 5)))

errors_train_cnn = np.zeros_like(channels)
errors_test_cnn = np.zeros_like(channels)
losses_train_cnn = np.zeros_like(channels, dtype=float)
losses_test_cnn = np.zeros_like(channels, dtype=float)

# Create directory for saving models
models_dir = "saved_models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created model save directory: {models_dir}")

print(f"Starting training of CNN models with different channel numbers, total {len(channels)} models...")
start_time = time.time()

for i, size in enumerate(channels):
    model_start_time = time.time()
    print(f'\n{"="*50}')
    print(f'Training model {i+1}/{len(channels)}: {size:3d} channels')
    print(f'Start time: {datetime.now().strftime("%H:%M:%S")}')
    print(f'{"="*50}')
        
    model = get_model_cnn(size)
    errors_train, errors_test, loss_train, loss_test, training_history = fit_model(model, ddata, n_epoch=500)
    errors_train_cnn[i] = errors_train
    errors_test_cnn[i] = errors_test
    losses_train_cnn[i] = loss_train
    losses_test_cnn[i] = loss_test
    
    # Save model parameters and training history
    model_filename = f"model_cnn_channels_{size}.pth"
    model_path = os.path.join(models_dir, model_filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'channels': size,
        'train_error': errors_train,
        'test_error': errors_test,
        'train_loss': loss_train,
        'test_loss': loss_test,
        'model_architecture': str(model),
        'training_history': training_history  # Add training history
    }, model_path)
    print(f'Model parameters and training history saved to: {model_path}')
    
    model_time = time.time() - model_start_time
    print(f'Model {size} completed - Train error: {errors_train:.2f}%, Test error: {errors_test:.2f}%')
    print(f'Model training time: {model_time:.1f} seconds')
    
    # Show overall progress
    elapsed_time = time.time() - start_time
    remaining_models = len(channels) - (i + 1)
    if remaining_models > 0:
        avg_time_per_model = elapsed_time / (i + 1)
        estimated_remaining = avg_time_per_model * remaining_models
        print(f'Overall progress: {i+1}/{len(channels)} ({100*(i+1)/len(channels):.1f}%)')
        print(f'Estimated remaining time: {estimated_remaining/60:.1f} minutes')

# Training completion statistics
total_time = time.time() - start_time
print(f'\n{"="*50}')
print(f'All CNN models training completed!')
print(f'Total training time: {total_time/60:.1f} minutes')
print(f'Average time per model: {total_time/len(channels):.1f} seconds')
print(f'Completion time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'All model parameters saved to directory: {models_dir}')
print(f'{"="*50}')

# Save training results to JSON file
import json
results = {
    'channels': channels.tolist(),
    'errors_train_cnn': errors_train_cnn.tolist(),
    'errors_test_cnn': errors_test_cnn.tolist(),
    'losses_train_cnn': losses_train_cnn.tolist(),
    'losses_test_cnn': losses_test_cnn.tolist(),
    'training_time_minutes': total_time/60,
    'completion_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open('training_results_cnn.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Training results saved to: training_results_cnn.json')

# Plot results
fig, ax = plt.subplots(figsize=(6, 4), dpi=125)

ax.plot(channels, errors_test_cnn, '-', label='Test')
ax.plot(channels, errors_train_cnn, '--', label='Train', zorder=0)

ax.set_ylim(0, 70)
ax.set_xlim(0, 100)
ax.set_xlabel('Num. of channels')
ax.set_ylabel('Classification error (%)')
ax.set_title('CNN with Label Noise')
ax.legend()
ax.plot([0, 100], [15, 15], ':', zorder=-1, color='#aaaaaa')

ax.spines.left.set_position(('outward', 3))
ax.spines.bottom.set_position(('outward', 3))

fig.tight_layout()

# Create figures directory (if it doesn't exist)
figures_dir = "figures"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"Created figures save directory: {figures_dir}")

# Save figures
fig.savefig(os.path.join(figures_dir, 'cnn_double_descent.png'), dpi=300)
fig.savefig(os.path.join(figures_dir, 'cnn_double_descent.pdf'))
print(f"Figures saved to: {figures_dir}/")

plt.show()