import pandas as pd
import matplotlib.pylab as plt

loss_list = [44.20351195335388, 21.00797325372696, 16.863190472126007,
             72.7160376906395, 23.207890331745148, 18.51715177297592]
EMA_loss = [44.20351195335388, 41.88395808339119, 39.381881322264675,
            42.71529695910216, 40.76455629636646, 38.5398158440274]
plt.plot(loss_list, label="loss")
plt.plot(EMA_loss, label="EMA loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title('Train'+"_loss")
plt.savefig("test.png")
plt.show()
plt.close()
