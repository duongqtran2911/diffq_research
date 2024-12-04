import json
import matplotlib.pyplot as plt
import os
import numpy as np

model_name = 'DLA_Simple'
name = model_name.lower()

# Define paths to your JSON files (update these paths)
file_paths = {
    32: f'/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model={name}/history.json',  # Group size 32 (normal model)
    4: f'/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model={name},quant.group_size=4,quant.penalty=5/history.json',
    8: f'/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model={name},quant.group_size=8,quant.penalty=5/history.json',
    16: f'/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model={name},quant.group_size=16,quant.penalty=5/history.json'
}


mode = 1

if mode == 0:
    # Define directory to save the plots
    output_dir = f"plots/{model_name}"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Dictionary to hold accuracy data for each group size
    accuracy_data = {}

    # Load data from each file
    for group_size, file_path in file_paths.items():
        with open(file_path, 'r') as f:
            data = json.load(f)
            accuracy_data[group_size] = {
                "train_acc": [epoch_data["train_acc"] for epoch_data in data],
                "valid_acc": [epoch_data["valid_acc"] for epoch_data in data],
                "best_acc": [epoch_data["best_acc"] for epoch_data in data]
            }

    # Extract group sizes
    group_sizes = sorted(accuracy_data.keys())

    # Plotting Final Epoch Accuracy vs Group Size
    plt.figure(figsize=(12, 6))
    final_train_accs = [accuracy_data[gs]["train_acc"][-1] for gs in group_sizes]
    final_valid_accs = [accuracy_data[gs]["valid_acc"][-1] for gs in group_sizes]
    final_best_accs = [accuracy_data[gs]["best_acc"][-1] for gs in group_sizes]
    final_test_accs = [92.41, 92.28, 92.13, 95.03]

    plt.subplot(1, 2, 1)
    plt.scatter(group_sizes, final_train_accs, marker='o', label="Train Accuracy")
    plt.scatter(group_sizes, final_valid_accs, marker='^', label="Validation Accuracy")
    plt.scatter(group_sizes, final_best_accs, marker='s', label="Best Accuracy")
    plt.scatter(group_sizes, final_test_accs, marker='h', label="Test Accuracy")
    plt.xlabel("Group Size")
    plt.ylabel("Final Accuracy (%)")
    plt.title(f"Final Accuracy vs Group Size, {model_name}")
    plt.legend()
    plt.grid(True)

    # Save the first plot
    plt.savefig(os.path.join(output_dir, "accuracy_vs_group_size.png"))

    # Plotting Accuracy vs Epochs for Each Group Size
    plt.subplot(1, 2, 2)
    for group_size in group_sizes:
        plt.plot(accuracy_data[group_size]["valid_acc"], label=f"Group Size {group_size}")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title(f"Validation Accuracy vs Epochs for Different Group Sizes, {model_name}")
    plt.legend()
    plt.grid(True)

    # Save the second plot
    plt.savefig(os.path.join(output_dir, "accuracy_vs_epochs.png"))

    # Show plots (optional, can be removed if only saving is needed)
    plt.tight_layout()
    plt.show()

elif mode == 1:

    # Dictionary to hold final accuracy data for each group size
    accuracy_data = {
        'train_acc': [],
        'valid_acc': [],
        'best_acc': [],
        'test_acc': []  # Assuming test accuracy is available; otherwise, exclude this
    }
    group_sizes = []

    # Load data from each file
    for group_size, file_path in file_paths.items():
        with open(file_path, 'r') as f:
            data = json.load(f)
            final_epoch = data[-1]  # Assuming the last entry is the final epoch
            group_sizes.append(group_size)
            accuracy_data['train_acc'].append(final_epoch["train_acc"])
            accuracy_data['valid_acc'].append(final_epoch["valid_acc"])
            accuracy_data['best_acc'].append(final_epoch["best_acc"])
            # Replace 'test_acc' with the correct field if test accuracy is available
            # accuracy_data['test_acc'].append(final_epoch.get("test_acc", 0))  # Set 0 if not available

    final_test_accs = [92.41, 92.28, 92.13, 95.03]

    # Convert group_sizes to a sorted list for ordered bar groups
    group_sizes = sorted(group_sizes)
    n_groups = len(group_sizes)

    # Define bar width and positions for each group
    bar_width = 0.15
    index = np.arange(n_groups)

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define colors for each category
    pastel1 = plt.get_cmap("tab20c")

    train_color = pastel1(0)
    valid_color = pastel1(1)
    best_color = pastel1(2)
    test_color = pastel1(3)

    # Plot bars for each accuracy type
    train_bars = ax.bar(index - 1.5 * bar_width, accuracy_data['train_acc'], 
                        bar_width, label="Train Accuracy", color=train_color)
    valid_bars = ax.bar(index - 0.5 * bar_width, accuracy_data['valid_acc'], 
                        bar_width, label="Validation Accuracy", color=valid_color)
    best_bars = ax.bar(index + 0.5 * bar_width, accuracy_data['best_acc'], 
                    bar_width, label="Best Accuracy", color=best_color)
    test_bars = ax.bar(index + 1.5 * bar_width, final_test_accs, 
                    bar_width, label="Test Accuracy", color=test_color)

    # Add values on top of each bar
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                height + 3,  # Positioning the text slightly above the bar
                f'{height:.2f}', 
                ha='center', 
                va='center',
                fontsize=8,  # Smaller font size
                rotation=45
            )

    add_labels(train_bars)
    add_labels(valid_bars)
    add_labels(best_bars)
    add_labels(test_bars)

    # Set labels, title, and x-ticks
    ax.set_xlabel('Group Size')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Final Accuracy vs Group Size')
    ax.set_xticks(index)
    ax.set_xticklabels(group_sizes)
    ax.legend(loc="lower right")

    # Display grid for readability
    ax.grid(True, linestyle="--", alpha=0.7)

    # Save the plot
    output_dir = f"plots/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "final_accuracy_vs_group_size.png"))

    # Show plot (optional, can be removed if only saving is needed)
    plt.show()



