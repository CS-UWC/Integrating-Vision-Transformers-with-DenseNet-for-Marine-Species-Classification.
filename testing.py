import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_logs(log_dir, output_pdf_path):
    # Load TensorBoard logs
    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()

    # Extract scalar data
    train_loss_data = event_accumulator.Scalars('train/loss')
    lr_data = event_accumulator.Scalars('train/lr')

    # Check if validation loss exists
    try:
        val_loss_data = event_accumulator.Scalars('val/loss')
        val_loss_exists = True
    except KeyError:
        val_loss_exists = False

    # Prepare the plot
    plt.figure(figsize=(12, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot([s.step for s in train_loss_data], [s.value for s in train_loss_data], label='Training Loss')

    if val_loss_exists:
        plt.plot([s.step for s in val_loss_data], [s.value for s in val_loss_data], label='Validation Loss')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss' + (' and Validation Loss' if val_loss_exists else ''))
    plt.legend()

    # Plot learning rate
    plt.subplot(1, 2, 2)
    plt.plot([s.step for s in lr_data], [s.value for s in lr_data], label='Learning Rate')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_pdf_path)
    plt.show()

# Example usage
log_dir = 'logs/Shark_den50'
output_pdf_path = 'wildfish__dense.pdf'
plot_tensorboard_logs(log_dir, output_pdf_path)

