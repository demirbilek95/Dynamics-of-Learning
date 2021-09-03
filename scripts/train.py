import torch
import numpy as np
from .train_utils import AverageMeter, accuracy
from .mnistParity import MNISTParity

def train_epoch(model, trainLoader, loss_fn, optimizer, loss_meter, performance_meter, performance, device, lr_scheduler):
    
    for X, y in trainLoader:
        X = X.to(device)
        y = y.to(device)
        # 1. reset the gradients previously accumulated by the optimizer
        #    this will avoid re-using gradients from previous loops
        optimizer.zero_grad() 
        # 2. get the predictions from the current state of the model
        #    this is the forward pass
        y_hat = model(X)
        # 3. calculate the loss on the current mini-batch
        loss = loss_fn(y_hat, y)
        # 4. execute the backward pass given the current loss
        loss.backward()
        # 5. update the value of the params
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        # 6. calculate the accuracy for this mini-batch
        acc = performance(y_hat, y)
        # 7. update the loss and accuracy AverageMeter
        loss_meter.update(val=loss.item(), n=X.shape[0])
        performance_meter.update(val=acc, n=X.shape[0])
        
        
def train_model(model, k, trainset, testset, loss_fn, optimizer, num_epochs, batch_size, validate_model = False, performance=accuracy, device=None, lr_scheduler=None, 
                lr_scheduler_step_on_epoch=True):

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    
    model = model.to(device)
    model.train()
    
    trainLostList = []
    trainAccList = []
    valLossList = []
    valAccList = [] 

    # epoch loop
    for epoch in range(num_epochs):
        trainData = MNISTParity(trainset, k, batch_size)
        loss_meter = AverageMeter()
        performance_meter = AverageMeter()

        if lr_scheduler != None: print(f"Epoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.5f}")
        lr_scheduler_batch = lr_scheduler if not lr_scheduler_step_on_epoch else None

        train_epoch(model, trainData.loader, loss_fn, optimizer, loss_meter, performance_meter, performance, device, lr_scheduler_batch)

        print(f"Epoch {epoch+1} completed. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.4f}; Performance: {performance_meter.avg:.4f}")
        trainLostList.append(loss_meter.sum)
        trainAccList.append(performance_meter.avg)

        if validate_model == True:
            testData = MNISTParity(testset, k, batch_size)
            val_loss, val_perf = test_model(model, k,testset, batch_size, performance=accuracy, loss_fn = loss_fn, device = "cuda:0")
            valLossList.append(val_loss)
            valAccList.append(val_perf)

        if lr_scheduler is not None and lr_scheduler_step_on_epoch:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(loss_meter.avg)
            else:
                lr_scheduler.step()

    return trainLostList, trainAccList, valLossList, valAccList

def test_model(model, k, testset, batch_size, performance=accuracy, loss_fn=None, device=None):
    
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    if loss_fn is not None:
        loss_meter = AverageMeter()
        
    performance_meter = AverageMeter()

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        testData = MNISTParity(testset, k, batch_size)
        for X, y in testData.loader:
            X = X.to(device)
            y = y.to(device)
            
            y_hat = model(X)
            loss = loss_fn(y_hat, y) if loss_fn is not None else None
            acc = performance(y_hat, y)
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
            performance_meter.update(acc, X.shape[0])
    # get final performances
    fin_loss = loss_meter.sum if loss_fn is not None else None
    fin_perf = performance_meter.avg
    print(f"TESTING - loss {fin_loss if fin_loss is not None else '--'} - performance {fin_perf:.4f}")
    return fin_loss, fin_perf
