import numpy as np
from .train import *

def tuneLearningRate_Torch(lr_array : np.array, optim: str, k:int, loss_type):
    listofValAcc = []
    for learning_rate in lr_array:
        print(f"Learning rate: {learn_rate}")
        model = MLP(k, "BP", loss_type)
        if optim == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

        trainLostListLoc, trainAccListLoc, valLossListLoc, valAccListLoc  = train_model(model, k, trainset, testset, loss_type, loss_fn, optimizer, num_epochs, 
                                                                                        batch_size, validate_model = True, performance=accuracy, device=device, 
                                                                                        lr_scheduler=None, updateWManually=False)

        last5 = valAccListLoc[-10:]
        meanOfLast5 = sum(last5) / len(last5)
        listofValAcc.append(meanOfLast5)
        listofValAccnp = np.array(listofValAcc)
        idx = np.argsort(listofValAccnp)
        best_lr = lr_array[idx][-1]
    
    print("Best learning rate is: ", best_lr)
    return best_lr

def tuneLearningRate_Manual(lr_array : np.array, training_method: str, init_B: str, optim: str, k:int, loss_type):
    listofValAcc = []
    for learning_rate in lr_array:
        print(f"Learning rate: {learning_rate}")
        model = MLPManual(k, learning_rate, loss_type, training_method, init_B, optim, device, measure_alignment, False)
        trainLostListLoc, trainAccListLoc, valLossListLoc, valAccListLoc ,_,_,_ = train_model_manually(model, k, trainset, testset, loss_type, loss_fn, 50,
                                                                                                batch_size, momentum, nesterov_momentum, weight_decay, False, 
                                                                                                validate_model = True, device=device)
        last5 = valAccListLoc[-10:]
        meanOfLast5 = sum(last5) / len(last5)
        listofValAcc.append(meanOfLast5)
        listofValAccnp = np.array(listofValAcc)
        idx = np.argsort(listofValAccnp)
        best_lr = lr_array[idx][-1]
    
    print("Best learning rate is: ", best_lr)
    return best_lr

# get the corresponding weights for single images (consider the case k=3)
def getW1ForImage(k, w):
    tensorList = []
    if k == 0:
        idx = 0
    elif k == 1:
        idx = 28
    else:
        idx = 56

    for i in range(1,29):
        #print(idx ,":", idx+28)
        tensorList.append(w[idx:idx+28,:])
        idx = idx+84
    return torch.vstack(tensorList)


def getMeanStd(model, k, trainset, testset, loss_type, loss_fn, num_epochs, batch_size, 
                momentum, nesterov_momentum, weight_decay, measure_alignment, validate_model, device):
    results = {}
    for i in range(1,4):
        trainLostListLoc, trainAccListLoc, valLossListLoc, valAccListLoc, _, _, _ = train_model_manually(model, k, trainset, testset, loss_type, loss_fn, num_epochs,
                                                                                                batch_size, momentum, nesterov_momentum, weight_decay, measure_alignment,
                                                                                                validate_model = True, device=device)
        results[i] = valAccListLoc
        
    liste = []
    for i in results:
        liste.append(results[i][-1])

    return liste