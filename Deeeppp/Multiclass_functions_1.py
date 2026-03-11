import torch 
from  torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DEVICE = "mps" if torch.mps.is_available() else "cpu"

def Train(model, train_DL, criterion, optimizer, EPOCH):
    
    loss_history = []
    acc_history = []
    NoT = len(train_DL.dataset)
    #to train mode
    model.train()
    for ep in range(EPOCH):
        rloss = 0
        rcorrect = 0
        for x_batch, y_batch in train_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            #inference
            y_hat = model(x_batch)
            #loss
            loss = criterion(y_hat, y_batch)
            #update
            optimizer.zero_grad() #gradient 누적방지
            loss.backward() #backprop
            optimizer.step() #weight update
            #loss accumulation
            loss_b = loss.item() * x_batch.shape[0]
            rloss += loss_b
            
            #tracking test accuary
            pred = y_hat.argmax(dim = 1)
            correct_b = torch.sum(pred == y_batch).item()
            rcorrect += correct_b
            
        #print loss & Accuary
        loss_e = rloss/NoT
        loss_history += [loss_e]
        accuracy_e = rcorrect/len(train_DL.dataset)*100
        acc_history += [accuracy_e]
        print(f"Epoch: {ep+1}, train loss: {(round(loss_e, 3))}")
        print(f"Train Accuaracy: {rcorrect}/{len(train_DL.dataset)} ({round(accuracy_e, 1)}%)")
        print("-"*20)        
    return loss_history, acc_history


def Test(model, test_DL):
    model.eval()
    with torch.no_grad():
        rcorrect = 0
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            #inference
            y_hat= model(x_batch)
            
            pred = y_hat.argmax(dim=1)
            correct_b = torch.sum(pred == y_batch).item()
            rcorrect += correct_b
        accuracy_e = rcorrect/len(test_DL.dataset)*100
    print(f"Test Accuaracy: {rcorrect}/{len(test_DL.dataset)} ({round(accuracy_e, 1)}%)")
    
def Test_plot(model, test_DL):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))
        x_batch = x_batch.to(DEVICE)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)

    x_batch = x_batch.to("cpu")

    plt.figure(figsize=(8,4))
    for idx in range(6):
        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(x_batch[idx].permute(1,2,0).squeeze(), cmap="gray")
        pred_class = test_DL.dataset.classes[pred[idx]]
        true_class = test_DL.dataset.classes[y_batch[idx]]
        plt.title(f"{pred_class} ({true_class})", color = "g" if pred_class==true_class else "r")     

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num
            