import matplotlib.pyplot as plt

def plot_loss(loss_history):

    plt.plot(loss_history)

    plt.xlabel("Iteration")

    plt.ylabel("Loss")

    plt.tile("Training Loss")

    plt.show()
