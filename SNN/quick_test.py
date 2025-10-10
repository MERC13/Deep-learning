import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

num_steps = 100 # number of time steps
batch_size = 1
beta = 0.5  # neuron decay rate
spike_grad = surrogate.fast_sigmoid() # surrogate gradient

net = nn.Sequential(
      nn.Conv2d(1, 8, 5),
      nn.MaxPool2d(2),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
      nn.Conv2d(8, 16, 5),
      nn.MaxPool2d(2),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
      nn.Flatten(),
      nn.Linear(16 * 4 * 4, 10),
      snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True, threshold=1e-1)
      )

data_in = torch.rand(num_steps, batch_size, 1, 28, 28)  # boosted random input data
spike_recording = [] # record spikes over time
utils.reset(net) # reset/initialize hidden states for all neurons

for step in range(num_steps): # loop over time
    spike, state = net(data_in[step]) # one time step of forward-pass
    spike_recording.append(spike) # record spikes in list

# Convert spike_recording to a tensor for easier processing
import matplotlib.pyplot as plt
spike_recording = torch.stack(spike_recording)  # shape: [num_steps, batch_size, 10]

# Print spike counts for each output neuron
spike_counts = spike_recording.sum(dim=0)  # shape: [batch_size, 10]
print("Spike counts for each output neuron:")
print(spike_counts)

# Print total number of spikes
total_spikes = spike_counts.sum().item()
print(f"Total number of spikes: {total_spikes}")

# Plot spike raster for the first sample in the batch
plt.figure(figsize=(10, 4))
for neuron in range(spike_recording.shape[2]):
      spike_times = torch.where(spike_recording[:, 0, neuron] > 0)[0]
      plt.scatter(spike_times.cpu(), [neuron]*len(spike_times), s=10)
plt.xlabel('Time step')
plt.ylabel('Output neuron')
plt.title('Spike Raster Plot (Sample 0)')
plt.yticks(range(spike_recording.shape[2]))
plt.show()