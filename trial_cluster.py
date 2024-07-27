import torch

# Assuming 'output' and 'label' are your tensors:
# output, label = model(batch_data)

# Create masks based on the conditions

condition1 = (output > 0.5) & (label == 1.0)
condition2 = (output < 0.5) & (label == 0)

# Combine conditions
correct_predictions = condition1 | condition2

# Count the correct predictions
num_correct = correct_predictions.sum().item()

print("Number of correct labels:", num_correct)
