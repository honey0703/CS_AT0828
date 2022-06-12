from M093781_HW5 import *

if __name__ == '__main__':
    # train and validate
    hist_train = []
    hist_val = []
    for epoch in range(start_epoch, start_epoch+200):
        hist_train = train(epoch)
        hist_val = valid(epoch)
        scheduler.step()

        #Plot loss chart
        plt.figure(figsize=(10,5))
        plt.title("Training and Validation Loss")
        plt.plot(hist_val,label="val")
        plt.plot(hist_train,label="train")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("Loss Chart.png")

    # testinig
    test()