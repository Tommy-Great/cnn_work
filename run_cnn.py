import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models.resnet import ResNet18_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def load_data(data_path='/tmp/dataset/', batch_size=32):
    def img2rgb(item):
        return item.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        img2rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    ])

    data_all = datasets.Caltech101(root=data_path, download=False, transform=transform)
    num_types = len(data_all.categories)
    train_size = int(0.8 * len(data_all))
    test_size = len(data_all) - train_size
    train_dataset, test_dataset = random_split(data_all, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, num_types


def get_model_normal():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 101)
    return model


def get_model_pretrained():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 101)
    return model


def sub_eval(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def sub_train(model, train_loader, test_loader,
              criterion, optimizer, scheduler, num_epochs, device):
    train_loss_list = []
    test_loss_list, test_acc_list = [], []
    best_acc = 0.0

    count_step = 0
    for epoch in range(num_epochs):
        model.train()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_step = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            count_step += 1
            train_loss_list.append(loss_step)
            # if count_step %100 ==0:
            #     print(f'Step {count_step}, Train Loss: {loss_step:.2f}')
            progress_bar.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
            progress_bar.set_postfix(loss=loss.item())

        test_loss, test_acc = sub_eval(model, test_loader, criterion, device)
        scheduler.step(test_loss)

        test_loss_list.append((count_step, test_loss))
        test_acc_list.append((count_step, test_acc))

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.2f}')

    result_train = {'train_loss': train_loss_list, 'test_loss': test_loss_list,
                    'test_acc': test_acc_list, 'best_acc': best_acc
                    }

    return result_train


def plot_history(history, hyperparams, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    fig_size = (12, 6)
    loss_legend_loc = "upper right"
    acc_legend_loc = "lower right"
    train_color = "deeppink"
    dev_color = 'lightpink'
    font_size = 'large'
    train_linestyle = '-'
    dev_linestyle = '--'

    plt.figure(figsize=fig_size)

    plt.subplot(1, 2, 1)

    train_loss_list = history['train_loss']
    dev_loss_list = history['test_loss']
    # train_steps = list(range(len(train_loss_list), sample_step))
    # train_losses = [float(train_loss_list[step_]) for step_ in train_steps]
    # plt.plot(train_steps, train_losses, color=train_color, linestyle=train_linestyle, label='Train loss')
    plt.plot(train_loss_list, color=train_color, linestyle=train_linestyle, label='Train loss')
    test_steps = [item[0] for item in dev_loss_list]
    test_losses = [float(item[1]) for item in dev_loss_list]
    plt.plot(test_steps, test_losses, color=dev_color, linestyle=dev_linestyle, label='dev loss')
    plt.ylabel('loss', fontsize=font_size)
    plt.xlabel('step', fontsize=font_size)
    plt.legend(loc=loss_legend_loc, fontsize='x-large')

    plt.subplot(1, 2, 2)
    dev_steps = [item[0] for item in history['test_loss']]
    dev_scores = [float(item[1]) for item in history['test_acc']]
    plt.plot(dev_steps, dev_scores, color=dev_color, linestyle=dev_linestyle, label='test accuracy')
    plt.ylabel('accuracy', fontsize=font_size)
    plt.xlabel('step', fontsize=font_size)
    plt.legend(loc=acc_legend_loc, fontsize='x-large')
    plt.grid(True)

    plt.tight_layout()
    path_fig = f'{save_path}/bs_{hyperparams["batch_size"]},lr_{hyperparams["lr"]}.png'
    plt.savefig(path_fig)
    print(path_fig)
    plt.close()


def run_model(run_type = 'pretrain', epochs = 10, 
              bs_list = (32, 64, 128), 
              lr_list = (5e-4, 1e-4, 5e-5)):
    # bs_list = [128]
    # lr_list = [1e-3]
    # epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use {device}")
    for batch_size in bs_list:
        for lr in lr_list:
            print(f"Run model, bs:{batch_size}, lr:{lr}, ep:{epochs}")
            train_loader, test_loader, num_classes = load_data(batch_size=batch_size)
            if run_type == 'pretrain' or run_type == 'pretrained':
                print('use model pretrained')
                model = get_model_pretrained().to(device)
                save_path = './history_pretrained'
            else:
                print('use model normal')
                model = get_model_normal().to(device)
                save_path = './history_normal'

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
            history = sub_train(model=model, train_loader=train_loader, test_loader=test_loader,
                                criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                                num_epochs=epochs, device=device)

            test_acc = history['best_acc']
            print('test accuracy: ', test_acc)
            plot_history(history, {'batch_size': batch_size, 'lr': lr, 'epochs': epochs},
                             save_path=save_path)

            name_model = f"bs_{batch_size}_lr_{lr}_ep_{epochs}_test_{test_acc:.4f}.pth"
            torch.save(model.state_dict(), name_model)
