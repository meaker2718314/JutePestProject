import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import warnings
import seaborn as sbn
import numpy as np

use_gpu = False

torch.manual_seed(25)

if torch.cuda.is_available():
    print("Running with CUDA...")
    use_gpu = True
else:
    print("Running with CPU...")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # OMP error fix
warnings.filterwarnings('ignore')

model = models.efficientnet_v2_m(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Dropout(0.33), nn.Linear(1280, 170),
                                 nn.Dropout(0.33), nn.Linear(170, 17),
                                 nn.LogSoftmax(dim=1))

if use_gpu:
    model.cuda()

optimizer = optim.Adam(model.parameters(), 0.00075)
criterion = nn.CrossEntropyLoss()

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # reverse 50% of images
    transforms.RandomVerticalFlip(),
    transforms.Resize((300, 300)),  # resize shortest side to 224 pixels
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    # Don't rotate or flip the test images
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

dir = './Data/Jute_Pest_Dataset/Jute_Pest_dataset'

# Create dataloaders
train_data = datasets.ImageFolder(os.path.join(dir, 'train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(dir, 'test'), transform=test_transform)
val_data = datasets.ImageFolder(os.path.join(dir, 'val'), transform=test_transform)
train_loader = DataLoader(train_data, batch_size=35, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)
val_loader = DataLoader(val_data, batch_size=100, shuffle=True)

epochs = 3

train_accs = []
val_accs = []
train_losses = []
val_losses = []

training_input = input("Train model (T) or load a trained model state? (L)").strip().upper()

if training_input == 'T':
    print("Training started...")
    training = True
else:
    print("Trained model loaded...")
    training = False
    model.load_state_dict(torch.load('./Saved States/learned_effNet_state.pt'))

for i in range(epochs):

    if not training:
        break

    train_loss = 0
    batch_correct = 0
    for b, (x_train, y_train) in enumerate(train_loader):

        if use_gpu:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        train_loss += loss.item()
        if b % 15 == 0:
            print(f"Epoch {i + 1} Batch {b + 1}  |  Loss = {train_loss / (b + 1)}")
        batch_correct += torch.eq(torch.max(y_pred, 1)[1], y_train).sum().item()

        #  main training block
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_losses.append(train_loss / (b + 1))
    train_accs.append(batch_correct / len(train_data))

    with torch.no_grad():
        test_loss = 0
        batch_correct = 0
        for b, (x_val, y_val) in enumerate(val_loader):
            if use_gpu:
                x_val, y_val = x_val.cuda(), y_val.cuda()
            y_pred = model.forward(x_val)
            loss = criterion(y_pred, y_val)
            test_loss += loss.item()
            batch_correct += torch.eq(torch.max(y_pred, 1)[1], y_val).sum().item()
        val_losses.append(test_loss / (b + 1))
        val_accs.append(batch_correct / (len(val_data)))

test_pred = []
test_true = []

with torch.no_grad():
    for x_test, y_test in test_loader:

        if use_gpu:
            x_test, y_test = x_test.cuda(), y_test.cuda()

        y_pred = model.forward(x_test)  # Feed Network

        output = torch.max(y_pred, 1)[1].cpu().numpy()
        test_pred.extend(output)

        labels = y_test.data.cpu().numpy()
        test_true.extend(labels)  # Save Truth

if training:
    torch.save(model.state_dict(), 'Saved States/learned_effNet_state.pt')

    # Graph validation losses vs training losses over time ...
    fig, ax = plt.subplots(figsize=(12, 10))
    h = range(1, epochs + 1)
    train_percents = list(map(lambda a: 100 * a, train_accs))
    val_percents = list(map(lambda a: 100 * a, val_accs))

    ax.scatter(h, train_percents, c='b', marker='s', label='train correct')
    ax.scatter(h, val_percents, c='g', marker='s', label='validation correct')
    ax.plot(h, train_percents, c='b')
    ax.plot(h, val_percents, c='g')

    ax.set_xlabel('Epoch #', labelpad=12, fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, epochs + 1))
    ax.set_yticks(np.linspace(70, 95, 13))

    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.legend(loc=4)

    plt.savefig('./Results/EpochAccuracies.png')

    # Graph accuracy bar chart for training, validation, and test datasets
    test_acc = torch.eq(torch.IntTensor(test_true), torch.IntTensor(test_pred)).sum().item()

    test_acc /= len(test_data)
    fig, ax = plt.subplots(figsize=(9, 9))

    heights = [train_accs[-1], val_accs[-1], test_acc]
    heights = list(map(lambda a: 100 * a, heights))

    bars = ax.bar(['Train', 'Validation', 'Test'], height=heights,
                  color=['cadetblue', 'mediumpurple', 'salmon'], width=0.5)

    ax.set_xlabel("Dataset", labelpad=12, fontsize=14, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold')
    ax.set_title('Final Accuracy Summary', pad=15, color='#333333',
                 weight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(np.linspace(0, 100, 21))
    plt.axhline(y=80, color='silver', linestyle=':')

    plt.savefig('./Results/FinalAccuracies.png')

# constant for classes
classes = ('beet armyworm', 'black hairy', 'cutworm', 'field cricket', 'jute aphid',
           'jute hairy', 'jute red mite', 'jute semilooper', 'jute stem girdler',
           'jute stem weevil', 'leaf beetle', 'mealybug', 'pod borer', 'scopula emissaria',
           'termite', 'termite odontotermes', 'yellow mite')

# Graph confusion matrix ...
cf_matrix = confusion_matrix(test_true, test_pred)

df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                     columns=[i for i in classes])

plt.figure(figsize=(16, 14))
ax = plt.subplot()
hm = sbn.heatmap(df_cm, annot=True, cmap='crest', linewidths=2, linecolor='grey', ax=ax)
ax.set_xlabel('Predicted Label', labelpad=30)
ax.set_ylabel('True Label')
plt.savefig('./Results/ConfusionMatrix.png')
