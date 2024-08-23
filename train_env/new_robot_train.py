import os
import time
import csv
from new_supervised_env import RobotEnv
import numpy as np
from simple_regression import CNN
import torch
import torch.optim as optim
# import run_case_studies as cs

batch_size = 5
env = RobotEnv(batch_size=batch_size)
epochs = 1
env.reset()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
total_images = 9*600
batches = int(total_images / batch_size)

# model = torch.load("BayesSwarm/trvl_bestmodel.pth")

# optimizer = optim.Adam(model.parameters(), lr=0.00001)
optimizer = optim.RMSprop(model.parameters(), lr=0.00001, momentum=0.02)
train_loss = 0
loss_per_epoch = []
validate_loss = []
current_loss = 1000000

for epoch in range(epochs):
    start_time = time.time()
    print(epoch, "Epoch")
    model.train()
    validate_model = False

    tr_loss = []

    for batch in range(batches):
        image = env.step()
        image = image.to(device)
        down_sampled = model(image)
        loss = env.loss(down_sampled)
        # tr_loss.append(loss.item())
        # train_loss += loss.item()

        optimizer.zero_grad()

        # loss.backward()

        # optimizer.step()

    # loss_per_epoch.append(train_loss / batches)
    # if (train_loss/batches) < current_loss:
    #     model_path = f'BayesSwarm/trvl_bestmodel.pth'
    #     torch.save(model, model_path)
    #     current_loss = train_loss/batches
    #     with open('best_loss.csv', mode='a', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow([train_loss/batches, epoch])
            
    # train_loss = 0

    # with open('tr_loss.csv', 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     for ls in tr_loss:
    #         writer.writerow([ls])
    #     writer.writerow([f"Epoch: {epoch}"])

    # with open('loss_per_epoch.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for ls in loss_per_epoch:
    #         writer.writerow([ls])
    
    # with open('loss_per.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for ls in loss_per_epoch:
    #         writer.writerow([f"epoch: {epoch}", ls])

    # if epoch % 10 == 0:
    #     for param in model.parameters():
    #         if param.grad is not None:
    #             if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
    #                 print("NaN or Inf gradients detected!")
    #                 break

    # end_time = time.time()

    # print("Time for an epoch: ", end_time - start_time)
