import torch
from  torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import wandb

DEVICE = "mps" if torch.mps.is_available() else "cpu"


'''
이제 train함수 안에서 train와 vali 둘 다 돌리기 위한
그걸 일단 함수화 해서 train 함수 안에서 돌리게끔..
'''

def loss_epoch(model, DL, criterion, optimizer = None):
    
    N = len(DL.dataset)
    rloss = 0; rcorrect = 0
    
    for x_batch, y_batch in tqdm(DL, leave = None):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        # inference
        y_hat = model(x_batch)
        # loss
        loss = criterion(y_hat, y_batch)
        # update
        if optimizer is not None: # train일때
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #loss accumulation
        loss_b = loss.item() * x_batch.shape[0]
        rloss += loss_b
        
        #accuaracy accumulation
        pred = y_hat.argmax(dim = 1)
        correct_b = torch.sum(pred == y_batch).item()
        rcorrect += correct_b
        
    loss_e = rloss/ N 
    accuracy_e = rcorrect/N*100

    return loss_e, accuracy_e, rcorrect

 

def Train(model, train_DL, val_DL, criterion, optimizer, 
          EPOCH,BATCH_SIZE, TRAIN_RATIO,
          save_model_path, save_history_path, **kwargs):
    
    if "LR_STEP" in kwargs:
        scheduler = StepLR(optimizer, step_size=kwargs["LR_STEP"], gamma=kwargs["LR_GAMMA"])
    else:
        scheduler = None
        
    
    loss_history = {"train" : [], "val" : []}
    acc_history = {"train" : [], "val" : []}
    best_loss = 9999
    
    for ep in range(EPOCH):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {ep+1}, current_LR = {current_lr}")
        
        model.train()
        train_loss, train_acc, _ = loss_epoch(model, train_DL, criterion, optimizer)
        loss_history["train"]+= [train_loss]
        acc_history["train"] += [train_acc]
        
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, _ = loss_epoch(model, val_DL, criterion)
            loss_history["val"]+= [val_loss]
            acc_history["val"] += [val_acc]

            if val_loss < best_loss: # early stopping
                best_loss = val_loss
                torch.save({"model" : model,
                            "ep" : ep,
                            "optimizer" : optimizer,
                            "scheduler" : scheduler}, save_model_path)
                
        if "LR_STEP" in kwargs: # epoch 끝날때마다 step -> update
            scheduler.step()
            
        # print loss
        print(f"train loss: {round(train_loss,5)}, "
              f"val loss: {round(val_loss,5)} \n"
              f"train acc: {round(train_acc,1)} %, "
              f"val acc: {round(val_acc,1)} %, time: {round(time.time()-epoch_start)} s")
        print("-"*20)

    torch.save({"loss_history": loss_history,
                "acc_history": acc_history,
                "EPOCH": EPOCH,
                "BATCH_SIZE": BATCH_SIZE,
                "TRAIN_RATIO": TRAIN_RATIO}, save_history_path)
    
    return loss_history, acc_history

def plot_history(loss_history, acc_history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # --- Loss plot ---
    axes[0].plot(loss_history["train"], label="Train Loss", marker='o', markersize=3)
    axes[0].plot(loss_history["val"],   label="Val Loss",   marker='o', markersize=3)
    axes[0].set_title("Loss History")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # best val loss epoch 표시
    best_ep = int(torch.tensor(loss_history["val"]).argmin())
    axes[0].axvline(x=best_ep, color='red', linestyle='--', alpha=0.5, label=f"Best ep: {best_ep+1}")
    axes[0].legend()

    # --- Accuracy plot ---
    axes[1].plot(acc_history["train"], label="Train Acc", marker='o', markersize=3)
    axes[1].plot(acc_history["val"],   label="Val Acc",   marker='o', markersize=3)
    axes[1].set_title("Accuracy History")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].axvline(x=best_ep, color='red', linestyle='--', alpha=0.5, label=f"Best ep: {best_ep+1}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("train_history.png", dpi=150)
    plt.show()
    print(f"Best Val Loss: {min(loss_history['val']):.5f} @ Epoch {best_ep+1}")

def Test(model, test_DL, criterion):
    model.eval()
    with torch.no_grad():
        test_loss, test_acc, rcorrect = loss_epoch(model, test_DL, criterion, optimizer=None)
    print()
    print(f"Test loss: {round(test_loss,5)}")
    print(f"Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({round(test_acc,1)} %)")
    return round(test_acc,1)


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
        
        
        
def im_plot(DL):
    x_batch, y_batch = next(iter(DL))
    plt.figure(figsize=(8,4))

    for idx in range(6):
        im = x_batch[idx]
        im = im-im.min() # for imshow clipping
        im = im/im.max() # for imshow clipping

        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(im.permute(1,2,0).squeeze())
        true_class = DL.dataset.classes[y_batch[idx]]
        plt.title(true_class, color = "g") 
    print(f"x_batch size = {x_batch.shape}")
    
         

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num
            
            
            


    
    
    