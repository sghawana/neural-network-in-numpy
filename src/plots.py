import matplotlib.pyplot as plt
import numpy as np


### FOR REGRESSION
def plot_multilinear_regression_results_train(actual_values, predicted_values):
    n_dimensions = actual_values.shape[0]
    if n_dimensions == 1:
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed
        actual_values_flat = actual_values.flatten()
        predicted_values_flat = predicted_values.flatten()
        wc, bc = np.polyfit(predicted_values_flat, actual_values_flat, 1)
        ax.scatter(actual_values_flat, predicted_values_flat, color='r', edgecolor='black', marker='o')
        ax.plot(actual_values_flat, actual_values_flat, color='black', label='Perfect Prediction')
        x_range = np.linspace(min(actual_values_flat), max(actual_values_flat), 100)
        ax.plot(x_range, wc * x_range + bc, color='skyblue', label='Line of Best Fit')
        ax.set_xlabel('Target Actual')
        ax.set_ylabel('Target Predicted')
        ax.set_title('Regression Results')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        fig, axs = plt.subplots(1, n_dimensions, figsize=(5*n_dimensions, 5))
        for i in range(n_dimensions):
            wc, bc = np.polyfit(predicted_values[i, :], actual_values[i, :], 1)
            ax = axs[i]
            ax.scatter(actual_values[i, :], predicted_values[i, :], color='r', edgecolor='black', marker='o')
            ax.plot(actual_values[i, :], actual_values[i, :], color='black', label='Perfect Prediction')
            x_range = np.linspace(min(actual_values[i, :]), max(actual_values[i, :]), 100)
            ax.plot(x_range, wc * x_range + bc, color='skyblue', label='Line of Best Fit')
            ax.set_xlabel('Target Actual')
            ax.set_ylabel('Target Predicted')
            ax.set_title(f'Dimension {i+1}')
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        plt.show()
        
        
def plot_multilinear_regression_results_val(actual_values, predicted_values):
    n_dimensions = actual_values.shape[0]
    if n_dimensions == 1:
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed
        actual_values_flat = actual_values.flatten()
        predicted_values_flat = predicted_values.flatten()
        wc, bc = np.polyfit(predicted_values_flat, actual_values_flat, 1)
        ax.scatter(actual_values_flat, predicted_values_flat, color='skyblue', edgecolor='black', marker='o')
        ax.plot(actual_values_flat, actual_values_flat, color='black', label='Perfect Prediction')
        x_range = np.linspace(min(actual_values_flat), max(actual_values_flat), 100)
        ax.plot(x_range, wc * x_range + bc, color='yellow', label='Line of Best Fit')
        ax.set_xlabel('Target Actual')
        ax.set_ylabel('Target Predicted')
        ax.set_title('Regression Results')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        fig, axs = plt.subplots(1, n_dimensions, figsize=(5*n_dimensions, 5))
        for i in range(n_dimensions):
            wc, bc = np.polyfit(predicted_values[i, :], actual_values[i, :], 1)
            ax = axs[i]
            ax.scatter(actual_values[i, :], predicted_values[i, :], color='skyblue', edgecolor='black', marker='o')
            ax.plot(actual_values[i, :], actual_values[i, :], color='black', label='Perfect Prediction')
            x_range = np.linspace(min(actual_values[i, :]), max(actual_values[i, :]), 100)
            ax.plot(x_range, wc * x_range + bc, color='yellow', label='Line of Best Fit')
            ax.set_xlabel('Target Actual')
            ax.set_ylabel('Target Predicted')
            ax.set_title(f'Dimension {i+1}')
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        plt.show()
        
def plot_loss(train_loss, val_loss):
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'black', label='Training loss')
    plt.plot(epochs, val_loss, 'red', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
### FOR CLASSIFICATION
def plot_roc_auc(fpr, tpr, auc):
    plt.title('RoC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def loss_and_accuracy(tr_loss, val_loss, tr_acc, val_acc):
    epochs = list(range(1, len(tr_loss) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(epochs, tr_loss, label='Train Loss', color='blue')
    axes[0].plot(epochs, val_loss, label='Validation Loss', color='red')
    axes[0].set_title('Loss Over Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(epochs, tr_acc, label='Train Accuracy', color='blue')
    axes[1].plot(epochs, val_acc, label='Validation Accuracy', color='red')
    axes[1].set_title('Accuracy Over epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()
    
def print_confusion_matrix(cm):
    print("Confusion Matrix:")
    print(cm)

def print_f1_scores(f1_scores):
    print("Class-wise F1 Scores:")
    for i, score in enumerate(f1_scores):
        print(f"Class {i}: {score}")

def plot_roc_curves(roc_curves):
    plt.figure(figsize=(12, 6))
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    for curve in roc_curves:
        class_pair = curve['class_pair']
        fpr = curve['fpr']
        tpr = curve['tpr']
        plt.plot(fpr, tpr, label=f'Class {class_pair[0]} vs Class {class_pair[1]}')
    plt.legend(loc='lower right', bbox_to_anchor=(1.02, 0), fontsize='small')
    plt.grid(True)
    plt.show()