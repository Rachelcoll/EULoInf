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
    color1 = 'tab:blue'
    color2 = 'tab:red'
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlim(min(x_vals), max(x_vals))

    ax1.plot(diff_val_loss, label='diff val_loss', color=color1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss', color=color1)

    ax2 = ax1.twinx()
    ax2.plot(diff_accuracy, label='diff_accuracy', color=color2)
    ax2.set_ylabel('Accuracy', color=color2)
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax1.spines['left'].set_color(color1)
    ax1.spines['left'].set_linewidth(3)
    ax2.spines['right'].set_color(color2)
    ax2.spines['right'].set_linewidth(2)
    
    plt.title(f'{scoring_method}: Validation of influence with top {np.round(p, 2)*100}% influential data points removed')
    fig.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.show()


def plot_var_acc_legends(x_vals, ckpt_labels, accuracies, data,
                         title, y_label, legend_labels, highlight_regions=None):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlim(min(x_vals), max(x_vals))

    for i in range(len(data)):
        if i == 1 and highlight_regions:
            ax1.plot(ckpt_labels, data[i], 'o-', label=legend_labels[i], alpha=0.3)
        else:
            ax1.plot(ckpt_labels, data[i], 'o-', label=legend_labels[i])

    colors = ['#CCFF00', '#39FF14']
    if highlight_regions:
        for i, (start, end) in enumerate(highlight_regions):
            ax1.axvspan(start, end, facecolor=colors[i], alpha=0.2)

    plt.title(title)
    ax1.set_xticks(ckpt_labels)
    ax1.set_xlabel("Checkpoints")
    ax1.set_ylabel(y_label)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ax2.plot(accuracies, label="Accuracy", color='k', alpha=0.3)

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
    
def plot_cosine_similarity(pw_similarities_lowp, ckpt_labels, p):
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot()
    im1 = ax1.imshow(pw_similarities_lowp, cmap='viridis', interpolation='nearest')
    fig.colorbar(im1, ax=ax1, label='Similarity')
    ax1.set_xticks(range(len(ckpt_labels)), ckpt_labels)
    ax1.set_yticks(range(len(ckpt_labels)), [f'Checkpoint {i}' for i in ckpt_labels])
    ax1.set_title(f'Checkpoint Pairwise Top {p*100}% Influential Point Similarity Heatmap')
    plt.show()
    