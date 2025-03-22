import numpy as np
import matplotlib.pyplot as plt

def plot_loss_acc(epochs_range, train_losses, validation_losses, accuracies):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs_range, train_losses, label='Train Loss')
    ax1.plot(epochs_range, validation_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    ax2.plot(epochs_range, accuracies, label='Accuracy', color='r')
    ax2.set_ylabel('Accuracy')

    plt.title('Loss and Accuracy')
    fig.legend()
    plt.show()

def plot_train_val_loss_acc_separately(epochs_range, train_losses, validation_losses, accuracies):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(211)
    ax1.plot(epochs_range, train_losses, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')

    ax2 = ax1.twinx()
    ax2.plot(epochs_range, accuracies, label='Accuracy', color='r')
    ax2.set_ylabel('Accuracy')

    ax3 = fig.add_subplot(212)
    ax3.plot(epochs_range, validation_losses, label='Validation Loss', color='g')
    ax4 = ax3.twinx()
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation Loss')
    ax4.plot(epochs_range, accuracies, label='Accuracy', color='r')
    ax4.set_ylabel('Accuracy')

    fig.suptitle('Loss and Accuracy')
    fig.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def plot_loss_acc_difference(diff_val_loss, diff_accuracy, scoring_method, p):
    num_epochs = len(diff_val_loss)
    x_vals = list(range(1, num_epochs+1))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlim(min(x_vals), max(x_vals))

    ax1.plot(diff_val_loss, label='diff val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')

    ax2 = ax1.twinx()
    ax2.plot(diff_accuracy, label='diff_accuracy', color='r')
    ax2.set_ylabel('Accuracy')
    plt.title(f'{scoring_method}: Validation of influence with top {np.round(p, 2)*100}% influential data points removed')
    fig.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.show()


def plot_var_acc_legends(x_vals, ckpt_labels, accuracies, data,
                         title, y_label, legend_labels):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlim(min(x_vals), max(x_vals))

    for i in range(len(data)):
        ax1.plot(ckpt_labels, data[i], 'o-', label=legend_labels[i])

    plt.title(title)
    ax1.set_xticks(ckpt_labels)
    ax1.set_xlabel("Checkpoints")
    ax1.set_ylabel(y_label)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ax2.plot(accuracies, label="Accuracy", color='k')

    fig.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.show()

def plot_similar_indices(pw_similarities_lowp, pw_similarities_highp, ckpt_labels, p):
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(211)
    im1 = ax1.imshow(pw_similarities_lowp, cmap='viridis', interpolation='nearest')
    fig.colorbar(im1, ax=ax1, label='Similarity')
    ax1.set_xticks(range(len(ckpt_labels)), ckpt_labels)
    ax1.set_yticks(range(len(ckpt_labels)), [f'Checkpoint {i}' for i in ckpt_labels])
    ax1.set_title(f'Checkpoint Pairwise Top {p*100}% Influential Point Similarity Heatmap')

    ax2 = fig.add_subplot(212)
    im2 = ax2.imshow(pw_similarities_highp, cmap='viridis', interpolation='nearest')
    fig.colorbar(im2, ax=ax2, label='Similarity')
    ax2.set_xticks(range(len(ckpt_labels)), ckpt_labels)
    ax2.set_yticks(range(len(ckpt_labels)), [f'Checkpoint {i}' for i in ckpt_labels])
    ax2.set_title(f'Checkpoint Pairwise Top {0.1*100}% Influential Point Similarity Heatmap')
    plt.tight_layout()

    plt.show()