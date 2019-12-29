import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# from torch.utils.tensorboard import SummaryWriter 

torch.set_printoptions(linewidth = 120)


torch.set_grad_enabled(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = torchvision.datasets.EMNIST('./data', 
                            train = True,
                            split='balanced', 
                            download = False, 
                            transform = transforms.Compose([
                                transforms.ToTensor()]))

test_set = torchvision.datasets.EMNIST('./data', 
                            train = False,
                            split='balanced', 
                            download = False, 
                            transform = transforms.Compose([
                                transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 100)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 100)

def get_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels.to(device)).sum().item()

def linear(in_ft, out_ft):
    return nn.Linear(in_features = in_ft, out_features = out_ft)

def Conv(in_ch, out_ch, ks):
    return nn.Conv2d(in_channels = in_ch ,out_channels = out_ch ,kernel_size = ks)

class Network(nn.Module):
    def __init__(self, in_c, n_classes):
        super(Network, self).__init__()
        
        self.conv1 = nn.Sequential(
                        Conv(in_c,6,5),
                        nn.ReLU(),
                        )
        
        self.conv2 = nn.Sequential(
                        Conv(6,12,5),
                        nn.ReLU()
                        )
    
        self.fc = nn.Sequential(
                        linear(12*4*4,192),
                        linear(192,120),
                        linear(120,n_classes)
                        )
        
    def forward(self, t):

        t = self.conv1(t)

        t = F.max_pool2d(t,2,2) #kernel size, stride
 
        t = self.conv2(t)
        t = F.max_pool2d(t,2,2)

        t = t.reshape(-1,12*4*4)

        t = self.fc(t)


        return t
        
        


network = Network(1,47)
network.cuda()
optimizer = optim.Adam(network.parameters(), lr = 0.01)

# images, labels = next(iter(train_loader))
# grid = torchvision.utils.make_grid(images)

# tb = SummaryWriter()
# tb.add_image('images',grid)
# tb.add_graph(network,images)

for epoch in range(30):

    
    total_loss = 0
    total_correct = 0
    
    for batch in train_loader:
       images, labels = batch
       images = images.to(device)
       labels = labels.to(device)
       preds = network(images)
       loss = F.cross_entropy(preds, labels)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       
       total_loss += loss.item()
       total_correct += get_correct(preds, labels)
    
    # tb.add_scalar('Loss',total_loss,epoch)
    # tb.add_scalar('Number Correct', total_correct,epoch)
    # tb.add_scalar('Accuracy',total_correct/len(train_set),epoch)
    
    # tb.add_histogram('conv1.bias',network.conv1.bias,epoch)
    # tb.add_histogram('conv1.weight',network.conv1.weight,epoch)
    # tb.add_histogram('conv1.weight.grad',network.conv1.weight.grad,epoch)
         
    print("epoch:",epoch,"total_correct:",total_correct, "loss:", total_loss)

    print(total_correct/len(train_set))
  
#confusion matrix 
# tb.close()

@torch.no_grad()
def get_all(model, loader):
    all_preds = torch.tensor([]).to(device)
    for batch in loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        
        all_preds = torch.cat((all_preds, preds), dim = 0)
    return all_preds


prediction_loader = torch.utils.data.DataLoader(train_set, batch_size = 5000)
train_preds = get_all(network, train_loader)
preds_correct = get_correct(train_preds, train_set.targets)

paired_preds = torch.stack((train_set.targets.to(device),train_preds.argmax(dim=1)),dim = 1) #true label, predicted label


cmt = torch.zeros((47,47),dtype = torch.int64)

for pair in paired_preds:
    true, predict = pair.tolist()
    cmt[true, predict] += 1
    
    

## TEST
    
prediction_loader = torch.utils.data.DataLoader(test_set, batch_size = 1000)
test_preds = get_all(network, test_loader)
preds_correct = get_correct(test_preds, test_set.targets)

paired_preds = torch.stack((test_set.targets.to(device),test_preds.argmax(dim=1)),dim = 1) #true label, predicted label


cmt = torch.zeros((47,47),dtype = torch.int64)

for pair in paired_preds:
    true, predict = pair.tolist()
    cmt[true, predict] += 1
    
    
    
    

