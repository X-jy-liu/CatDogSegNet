import matplotlib.pyplot as plt

def training_plot(history, save_path=None):
    """
    Plots the training and validation loss curves from the history dict.

    Args:
        history (dict): Contains 'train_loss' and 'val_loss' lists.
        save_path (str, optional): If provided, saves the plot to this path.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Train Loss', marker='o', ms=4)
    plt.plot(history['val_loss'], label='Validation Loss', marker='o', ms=4)
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
