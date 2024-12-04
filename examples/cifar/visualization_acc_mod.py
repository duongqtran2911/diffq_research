import json
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import ticker, transforms, scale
from matplotlib.scale import ScaleBase, register_scale

model_name = 'SeNet'
name = model_name.lower()

# Define paths to your JSON files (update these paths)
file_paths = {
    32: f'/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model={name}/history.json',  # Group size 32 (normal model)
    4: f'/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model={name},quant.group_size=4,quant.penalty=5/history.json',
    8: f'/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model={name},quant.group_size=8,quant.penalty=5/history.json',
    16: f'/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model={name},quant.group_size=16,quant.penalty=5/history.json'
}

## CUSTOM SCALE FOR MODEL SIZE
"""
# Define directory to save the plot
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Dictionary to hold accuracy and model size data for each group size
model_data = {}

# Load data from each file
for group_size, file_path in file_paths.items():
    with open(file_path, 'r') as f:
        data = json.load(f)
        final_epoch = data[-1]  # Assuming the last entry is the final epoch
        model_data[group_size] = {
            "valid_acc": final_epoch["valid_acc"],
            "best_acc": final_epoch["best_acc"],
            "model_size": final_epoch["model_size"]
        }

# Extract group sizes, accuracies, and model sizes for plotting
group_sizes = sorted(model_data.keys())
valid_accs = [model_data[gs]["valid_acc"] for gs in group_sizes]
best_accs = [model_data[gs]["best_acc"] for gs in group_sizes]
model_sizes = [model_data[gs]["model_size"] for gs in group_sizes]

# Define a custom scaling function
class CustomScale(scale.ScaleBase):
    name = 'custom'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)  # Pass the axis argument to the superclass

    def get_transform(self):
        return self.CustomTransform()

    class CustomTransform(transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, x):
            # Linear for 0 to 0.5, then logarithmic for higher values
            return np.where(x <= 0.5, x, 0.5 + np.log10(x + 0.5))

        def inverted(self):
            return CustomScale.InvertedCustomTransform()

    class InvertedCustomTransform(transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, x):
            return np.where(x <= 0.5, x, 10**(x - 0.5) - 0.5)

# Register the custom scale
register_scale(CustomScale)

# Plotting Accuracy vs Model Size with Custom Scale on X-Axis
plt.figure(figsize=(8, 6))

# Choose which accuracy metric to plot (e.g., valid_acc or best_acc)
plt.plot(model_sizes, valid_accs, marker='o', label="Validation Accuracy")
plt.plot(model_sizes, best_accs, marker='o', label="Best Accuracy")

plt.xlabel("Model Size (MB)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Model Size, ResNet")

# Use custom scale on x-axis
plt.gca().set_xscale('custom')
plt.legend()
plt.grid(True, which="both", linestyle="--")  # Show grid for both major and minor ticks

# Save the plot
plt.savefig(os.path.join(output_dir, "accuracy_vs_model_size_custom_scale.png"))

# Show plot (optional, can be removed if only saving is needed)
plt.show()
"""

## LINEAR SCALE FOR MODEL SIZE
"""
# Define directory to save the plot
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Dictionary to hold accuracy and model size data for each group size
model_data = {}

# Load data from each file
for group_size, file_path in file_paths.items():
    with open(file_path, 'r') as f:
        data = json.load(f)
        final_epoch = data[-1]  # Assuming the last entry is the final epoch
        model_data[group_size] = {
            "valid_acc": final_epoch["valid_acc"],
            "best_acc": final_epoch["best_acc"],
            "model_size": final_epoch["model_size"]
        }

# Extract group sizes, accuracies, and model sizes for plotting
group_sizes = sorted(model_data.keys())
valid_accs = [model_data[gs]["valid_acc"] for gs in group_sizes]
best_accs = [model_data[gs]["best_acc"] for gs in group_sizes]
model_sizes = [model_data[gs]["model_size"] for gs in group_sizes]

# Plotting Accuracy vs Model Size
plt.figure(figsize=(8, 6))

# Choose which accuracy metric to plot (e.g., valid_acc or best_acc)
plt.plot(model_sizes, valid_accs, marker='o', label="Validation Accuracy")
plt.plot(model_sizes, best_accs, marker='^', label="Best Accuracy")

plt.xlabel("Model Size (MB)")
plt.ylabel("Accuracy (%)")
plt.title(f"Accuracy vs Model Size, {model_name}")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(os.path.join(output_dir, "accuracy_vs_model_size.png"))

# Show plot (optional, can be removed if only saving is needed)
plt.show()
"""



## SYMLOG FOR MODEL SIZE
# """
# Define directory to save the plot
output_dir = f"plots/{model_name}"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Dictionary to hold accuracy and model size data for each group size
model_data = {}

# Load data from each file
for group_size, file_path in file_paths.items():
    with open(file_path, 'r') as f:
        data = json.load(f)
        final_epoch = data[-1]  # Assuming the last entry is the final epoch
        model_data[group_size] = {
            "valid_acc": final_epoch["valid_acc"],
            "best_acc": final_epoch["best_acc"],
            "model_size": final_epoch["model_size"]
        }

# Extract group sizes, accuracies, and model sizes for plotting
group_sizes = sorted(model_data.keys())
valid_accs = [model_data[gs]["valid_acc"] for gs in group_sizes]
best_accs = [model_data[gs]["best_acc"] for gs in group_sizes]
model_sizes = [model_data[gs]["model_size"] for gs in group_sizes]

# Plotting Accuracy vs Model Size with a Custom Non-Linear Scale on X-Axis
plt.figure(figsize=(8, 6))

# Choose which accuracy metric to plot (e.g., valid_acc or best_acc)
plt.plot(model_sizes, valid_accs, marker='o', label="Validation Accuracy")
plt.plot(model_sizes, best_accs, marker='^', label="Best Accuracy")

plt.xlabel("Model Size (MB)")
plt.ylabel("Accuracy (%)")
plt.title(f"Accuracy vs Model Size, {model_name}")

# Set x-axis to a custom scaling with SymLogNorm to be dense at the higher end
plt.xscale('symlog', linthresh=3)  # Adjust `linthresh` to control where the transition occurs

# Configure custom ticks for better spacing
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.legend()
plt.grid(True, which="both", linestyle="--")  # Show grid for both major and minor ticks

# Save the plot
plt.savefig(os.path.join(output_dir, "accuracy_vs_model_size_custom_scale.png"))

# Show plot (optional, can be removed if only saving is needed)
plt.show()
# """

## LOG-SCALE FOR MODEL SIZE
"""
# Define directory to save the plot
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Dictionary to hold accuracy and model size data for each group size
model_data = {}

# Load data from each file
for group_size, file_path in file_paths.items():
    with open(file_path, 'r') as f:
        data = json.load(f)
        final_epoch = data[-1]  # Assuming the last entry is the final epoch
        model_data[group_size] = {
            "valid_acc": final_epoch["valid_acc"],
            "best_acc": final_epoch["best_acc"],
            "model_size": final_epoch["model_size"]
        }

# Extract group sizes, accuracies, and model sizes for plotting
group_sizes = sorted(model_data.keys())
valid_accs = [model_data[gs]["valid_acc"] for gs in group_sizes]
best_accs = [model_data[gs]["best_acc"] for gs in group_sizes]
model_sizes = [model_data[gs]["model_size"] for gs in group_sizes]

# Plotting Accuracy vs Model Size with Logarithmic Scale on X-Axis
plt.figure(figsize=(8, 6))

# Choose which accuracy metric to plot (e.g., valid_acc or best_acc)
plt.plot(model_sizes, valid_accs, marker='o', label="Validation Accuracy")
plt.plot(model_sizes, best_accs, marker='o', label="Best Accuracy")

plt.xlabel("Model Size (MB)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Model Size, ResNet")
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.legend()
plt.grid(True, which="both", linestyle="--")  # Show grid for both major and minor ticks

# Save the plot
plt.savefig(os.path.join(output_dir, "accuracy_vs_model_size_log_scale.png"))

# Show plot (optional, can be removed if only saving is needed)
plt.show()
"""


