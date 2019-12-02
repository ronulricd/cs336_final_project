import numpy as np
import matplotlib.pyplot as plt

def displayLoss():
    loss = np.load("learning/imitation/pytorch/dagger_windowloss.npy")
    epochs = np.linspace(0, len(loss), num=len(loss))
    plt.title('Training Loss')
    plt.xlabel("Epochs")
    plt.ylabel('Mean Squared Loss')
    plt.plot(epochs, loss)
    plt.show()

if __name__=="__main__":
    displayLoss()
