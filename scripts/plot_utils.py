import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, num_epochs):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9))
    ax1.plot(list(range(1, num_epochs+1)), train_loss, label="Train Loss")
    ax1.plot(list(range(1, num_epochs+1)), val_loss, label="Validation Loss")
    ax1.legend()

    ax2.plot(list(range(1, num_epochs+1)), train_acc, label="Train Accuracy")
    ax2.plot(list(range(1, num_epochs+1)), val_acc, label="Validation Accuracy")
    ax2.legend();


def plotValAccuracy(val_acc, num_epochs, activation, k):
    plt.ylim(0.45, 1)
    plt.title("Validation Accuracy for K = {}".format(k))
    plt.plot(range(1, num_epochs+1), val_acc, label=activation)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.grid(True)
    plt.legend();
