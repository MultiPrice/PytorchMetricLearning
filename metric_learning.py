import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import time
import csv


### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
from torchvision import datasets, transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

from torch.utils.tensorboard import SummaryWriter


### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.norm = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.normalize(x, p=2, dim=1)
        return x


# -----------------------------------------------------------------------------------------------------------------------------------
def PlotDigits():
    if(write_data == False):
        return
    examples = iter(test_loader)
    example_data, _ = next(examples)
    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image(name + '_images', img_grid)
    writer.close()

def AddModelGraph():
    if(write_data == False):
        return
    examples = iter(test_loader)
    example_data, _ = next(examples)
    writer.add_graph(model, example_data.to(device))
    writer.close()

def VisualizeTraining(writer, running_loss, batch_idx, batch_iter, epoch, n_total_steps, visualization_name):
    if(write_data == False):
        return
    if(batch_idx + 1) % batch_iter == 0:
        writer.add_scalar(visualization_name + '_training loss_', running_loss, epoch * n_total_steps + batch_idx)

def VisualizeCurves():
    if(write_data == False):
        return
    preds = torch.cat([torch.stack(batch) for batch in preds])
    labels = torch.cat(labels)

    classes = range(10)
    for i in classes:
        labels_i = labels == i
        preds_i = preds[:, i]
        writer.add_pr_curve(name + '_' + str(i), labels_i, preds_i, global_step=0)
        writer.close()

def TrainModel():
    model.train()

    knn_train_data = []
    knn_train_labels = []

    n_total_steps = len(train_loader)

    loss_array = []
    tensorboard_loss = 0.0

    start_time = time.time()

    for epoch in range(epoch_number):
        for batch_idx, (data, labels) in enumerate(train_loader):
            knn_train_labels.extend(labels.cpu().numpy())

            data = data.to(device)
            labels = labels.to(device)

            # forward pass
            embeddings = model(data)
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)
            
            knn_train_data.extend(embeddings.cpu().detach().numpy())
            
            loss_array.append(loss.item())

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # adding current loss for visualization
            tensorboard_loss += loss.item()
            VisualizeTraining(writer, tensorboard_loss, batch_idx, visualization_mod, epoch, n_total_steps, name)
            tensorboard_loss = 0.0

            # printin current status
            # if(batch_idx + 1) % (50) == 0:
            #     print(f"Epoch [{epoch + 1}/{epoch_number}] Step [{batch_idx+1}/{n_total_steps}]: Loss = {loss.item():.8f}")

        loss_array = np.array(loss_array)
        print(f"    EPOCH {epoch + 1}: MEAN LOSS = {loss_array.mean()} TIME = {(time.time() - start_time):.2f} s")
        loss_array = []
    
    global training_time
    training_time = time.time() - start_time

    # Inicjalizacja i trening klasyfikatora k-NN
    # print(f"KNN TRAINING: {name}")
    knn_train_data = np.array(knn_train_data)
    knn_train_labels = np.array(knn_train_labels)

    knn.fit(knn_train_data, knn_train_labels)

def TestModel():
    #print(f"EVALUATION: {name}")
    model.eval()

    knn_test_data = []
    knn_test_labels = []

    with torch.no_grad():
        loss_array = []

        for batch_idx, (data, labels1) in enumerate(test_loader):
            knn_test_labels.extend(labels1.cpu().numpy())
            data, labels1 = data.to(device), labels1.to(device)
            embeddings = model(data)

            knn_test_data.extend(embeddings.cpu().detach().numpy())

            # loss calculation
            indices_tuple = mining_func(embeddings, labels1)
            loss = loss_func(embeddings, labels1, indices_tuple)
            loss_array.append(loss.item())

        # Klasyfikacja danych testowych knn
        #print(f"KNN TESTING: {name}")
        knn_test_data = np.array(knn_test_data)
        knn_test_labels = np.array(knn_test_labels)

        knn_predicted_labels = knn.predict(knn_test_data)

        # Obliczenie dokładności klasyfikacji
        accuracy = accuracy_score(knn_test_labels, knn_predicted_labels)
        print(f"Classification accuracy for {name}: {(accuracy * 100):.2f} %")

        # final mean and std
        loss_array = np.array(loss_array)
        mean = loss_array.mean()
        std = loss_array.std()
        print(f"Final mean loss for {name} = {mean:.10f}")
        print(f"Final std loss for {name} = {std:.10f}")

        result_data = [name, epoch_number,  f"{mean:.8f}", f"{std:.8f}", f"{(accuracy * 100):.2f} %", f"{(training_time):.2f} s"]
        csv_file_path = "training_results.csv"  # Change the file path as needed
        SaveTestResults(csv_file_path, result_data)

def SaveModel():
    FILE = name + ".pth"
    torch.save(model.state_dict(), FILE)

    #loaded_model = ConvNet()
    #loaded_model.load_state_dict(torch.load(FILE))
    #loaded_model.to(device)
    #loaded_model.eval()

def SaveTestResults(file_path, data):
    with open(file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';')
        csv_writer.writerow(data)

def TrainAndTestModel():
    print(f"MODEL: {name}")
    
    TrainModel()
    
    TestModel()

    SaveModel()

def TrainAndTestModelSeries():
    global name
    global epoch_number

    epoch_number = 1
    name = f"{original_name} ({epoch_number})"
    TrainAndTestModel()
    name = original_name

    epoch_number = 5
    name = f"{original_name} ({epoch_number})"
    TrainAndTestModel()
    name = original_name

    epoch_number = 10
    name = f"{original_name} ({epoch_number})"
    TrainAndTestModel()
    name = original_name

    epoch_number = 20
    name = f"{original_name} ({epoch_number})"
    TrainAndTestModel()
    name = original_name

    epoch_number = 50
    name = f"{original_name} ({epoch_number})"
    TrainAndTestModel()
    name = original_name
# -----------------------------------------------------------------------------------------------------------------------------------

# setting seed
torch.manual_seed(303)

# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define writter
writer = SummaryWriter("runs/mnist")
write_data = True
visualization_mod = 1
training_time = time.time()

# hyper-parameters
epoch_number = 1
batch_size = 64
learning_rate = 0.001

# knn parameters
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# model setup
model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# metric learning setup
distance = distances.DotProductSimilarity()
reducer = reducers.MeanReducer()
loss_func = losses.NPairsLoss(distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")

# dataset preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset1 = datasets.MNIST(".", train=True, download=True, transform=transform)
dataset2 = datasets.MNIST(".", train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)
# -----------------------------------------------------------------------------------------------------------------------------------

original_name = 'ContrastiveLossSeed'
name = original_name
distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
reducer = reducers.AvgNonZeroReducer()
loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=1, distance=distance, reducer=reducer)
TrainAndTestModelSeries()

original_name = 'TripletMarginLossSeed'
name = original_name
distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
reducer = reducers.AvgNonZeroReducer()
loss_func = losses.TripletMarginLoss(margin=0.05, distance=distance, reducer=reducer)
TrainAndTestModelSeries()

original_name = 'NPairsLossSeed'
name = original_name
distance = distances.DotProductSimilarity()
reducer = reducers.MeanReducer()
loss_func = losses.NPairsLoss(distance=distance, reducer=reducer)
TrainAndTestModelSeries()

original_name = 'ProxyAnchorLossSeed'
name = original_name
distance = distances.CosineSimilarity()
reducer = reducers.DivisorReducer()
loss_func = losses.ProxyAnchorLoss(10, 128, margin = 0.1, alpha = 32, distance=distance, reducer=reducer)
TrainAndTestModelSeries()

original_name = 'NormalizedSoftmaxLossSeed'
name = original_name
distance = distances.DotProductSimilarity()
reducer = reducers.MeanReducer()
loss_func = losses.NormalizedSoftmaxLoss(10, 128, temperature=0.05, distance=distance, reducer=reducer)
TrainAndTestModelSeries()

original_name = 'NCALossSeed'
name = original_name
distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
reducer = reducers.MeanReducer()
loss_func = losses.NCALoss(softmax_scale=1, distance=distance, reducer=reducer)
TrainAndTestModelSeries()

original_name = 'MarginLossSeed'
name = original_name
distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
reducer = reducers.DivisorReducer()
loss_func = losses.MarginLoss(margin=0.2, nu=0, beta=1.2,  triplets_per_anchor="all", learn_beta=False, num_classes=None, distance=distance, reducer=reducer)
TrainAndTestModelSeries()

original_name = 'LargeMarginSoftmaxLossSeed'
name = original_name
distance = distances.CosineSimilarity()
reducer = reducers.MeanReducer()
loss_func = losses.LargeMarginSoftmaxLoss(10, 128, margin=4, scale=1, distance=distance, reducer=reducer)
TrainAndTestModelSeries()

original_name = 'AngularLossSeed'
name = original_name
distance = distances.LpDistance(p=2, power=1, normalize_embeddings=True)
reducer = reducers.MeanReducer()
loss_func = losses.AngularLoss(alpha=40, distance=distance, reducer=reducer)
TrainAndTestModelSeries()

original_name = 'CircleLossSeed'
name = original_name
distance = distances.CosineSimilarity()
reducer = reducers.AvgNonZeroReducer()
loss_func = losses.CircleLoss(m=0.4, gamma=80, distance=distance, reducer=reducer)
TrainAndTestModelSeries()

original_name = 'FastAPLossSeed'
name = original_name
distance = distances.LpDistance(normalize_embeddings=True, p=2, power=2)
reducer = reducers.MeanReducer()
loss_func = losses.FastAPLoss(num_bins=10, distance=distance, reducer=reducer)
TrainAndTestModelSeries()

original_name = 'LiftedStructureLossSeed'
name = original_name
distance = distances.LpDistance(normalize_embeddings=True, p=2, power=2)
reducer = reducers.MeanReducer()
loss_func = losses.LiftedStructureLoss(neg_margin=1, pos_margin=0, distance=distance, reducer=reducer)
TrainAndTestModelSeries()
