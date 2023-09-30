import torch
import numpy as np


#inputs and constants
inputs= np.array([[73,67,43],
                 [91,88,64],
                 [87,134,58],
                 [102,43,37],
                 [69,96,70]], dtype = 'float32'
                )
targets = np.array([[56,70],
                    [81,101],
                    [119,133],
                    [22,37],
                    [103,119]],dtype = 'float32')

# Weights and biases (randomly generated)
w = np.array([[ 0.6802, -0.3004,  0.9199],
        [-0.1704,  0.7874,  1.3520]], dtype = 'float32')
b = np.array([0.8349, 0.5365], dtype = 'float32')

# Convert arrays to tensors

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Convert arrays to tensors for weight and biases and make them derivable.
w = torch.from_numpy(w).clone().detach().requires_grad_()
b = torch.from_numpy(b).clone().detach().requires_grad_()
#print(w.t())
#print(b)
#print(targets)

# defining a predictions function:
def model(x):
    return x @ w.t() + b

# defining loss calculation function:

def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff)/diff.numel()

m = 10000


for i in range(m):
    preds = model(inputs)
    loss = mse(preds, targets)
    # calculating the loss
    # and the gradian of our weights and biases
    # if its positive we remove a little portion of the weight/biases (it's |grad * 10^-5|)
    # if its negative we add a little portion of the weight/biases (it's |grad * 10^-5|)
    # This way we continously to dicrease the loss.
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

        # Tracking "print" the progress in diff levels of our optimization.
    if i == (0):
        print(f'#{i}preds : {preds}  | targets: {targets} |  loss : {loss}')
    elif i == round(m * 0.25) :
        print(f'#{i}preds : {preds}  | targets: {targets} |  loss : {loss}')
    elif i == round(m * 0.5):
        print(f'#{i}preds : {preds}  | targets: {targets} |  loss : {loss}')
    elif i == round(m * 0.75) :
        print(f'#{i}preds : {preds}  | targets: {targets} |  loss : {loss}')
    elif i == m:
        print(f'#{i}preds : {preds}  | targets: {targets} |  loss : {loss}')
