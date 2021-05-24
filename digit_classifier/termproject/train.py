import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model import MyCNN

from tensorboardX import SummaryWriter
from tqdm import tqdm
import pandas as pd
import datetime
import cv2
import time


def dataframe_to_numpy(df_data):
    label = np.array(df_data["0"])
    data = np.array(df_data.drop("0", axis=1)).reshape(-1, 3, 28, 28)

    return label, data

def test(loader, model):
    model.eval()
    n_pred = 0
    n_corr = 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            y_hat = model(X)
            y_hat.argmax()

            _, pred = torch.max(y_hat, 1)

            n_pred += len(pred)
            n_corr += (Y == pred).sum()
    
    acc = n_corr / n_pred
    # print("Accuracy : {0}".format(acc))

    return acc

def run(data_path, test_mode=False):
    save_path = "./models/"

    # Hyperparameters
    learning_rate = 0.001
    batch_size = 100
    num_epochs = 200

    if test_mode:
        df_data = pd.read_csv(data_path + "train_data.csv")
        label, data = dataframe_to_numpy(df_data)

        test_set = list(zip(torch.Tensor(data), torch.LongTensor(label)))

        # Data loader
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)
        model = MyCNN()
        acc = test(test_loader, model)
        print(acc)
        
    else: # test_mode is False
        print("======READ DATA======")
        df_data = pd.read_csv(data_path + "train_data.csv")
        label, data = dataframe_to_numpy(df_data)

        train_data = list(zip(torch.Tensor(data).to(device), torch.LongTensor(label).to(device)))

        # Dataset
        split_n = int(0.8 * len(train_data))
        train_set = train_data[:split_n] # 330960
        valid_set = train_data[split_n:] # 84740

        # DataLoader
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

        # Hyperparameters tuning
        for i in range(1):
            best_acc = 0
            d = datetime.datetime.now()
            formatted_d = "{0}_{1}_{2}_{3}_{4}".format(d.month, d.day, d.hour, d.minute, d.second)
            writer = SummaryWriter(logdir='runs/MyCNN_{0}'.format(formatted_d))

            model = MyCNN().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Train
            print("======TRAIN START======")
            model.train()
            for epoch in tqdm(range(num_epochs)):
                cost = 0
                n_batches = 0

                for X, Y in train_loader:
                    X, Y = X.to(device), Y.to(device)
                    optimizer.zero_grad()
                    y_hat = model(X)
                    loss = criterion(y_hat, Y)
                    loss.backward()
                    optimizer.step()

                    cost += loss.item()
                    n_batches += 1
                
                cost /= batch_size
                if epoch % 50 == 0:
                    print("[Epoch ({0})] : cost = {1}".format(epoch+1, cost))

                # Write Summary
                writer.add_scalar("Loss/train", loss, epoch)
                acc_valid = test(valid_loader, model)
                acc_train = test(train_loader, model)
                writer.add_scalar("Acc/train", acc_train, epoch)
                writer.add_scalar("Acc/valid", acc_valid, epoch)

                if acc_valid > best_acc:
                    # keep the best model
                    best_acc = acc_valid
                    print("BEST ACC : {0} + Epoch {1}".format(best_acc, epoch))
                    torch.save(model.state_dict(), save_path + 'best_model_{0}_epoch{1}.pt'.format(formatted_d, epoch))  # save state_dict of the best model
        
            # Validate
            print("======VALIDATION START======")
            acc = test(valid_loader, model)
            acc_history.append(acc)
            print("======ACCURACY======")
            print(acc_history)
            writer.flush()
            writer.close()
        

if __name__=="__main__":
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    if is_cuda:
        print("Running with CUDA")
    else:
        print("Running with CPU")

    start = time.time()
    # Dataset path
    data_path = "./dataset/"

    run(data_path, test_mode=False)
    end = time.time()
    print("Total Running Time : {0} sec".format(round(end-start, 4)))