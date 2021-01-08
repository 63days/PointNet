import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TNet(nn.Module): # Spatial Transformation Network

    def __init__(self, d=3):
        super(TNet, self).__init__()
        self.d = d

        self.fc1 = nn.Linear(d, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.weights = torch.zeros([256, d*d], requires_grad=True, dtype=torch.float32, device=device)
        self.bias = torch.eye(d, requires_grad=True, dtype=torch.float32, device=device).flatten()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x).transpose(1, 2)), inplace=True)
        x = x.transpose(1, 2)
        x = F.relu(self.bn2(self.fc2(x).transpose(1, 2)), inplace=True)
        x = x.transpose(1, 2)
        x = F.relu(self.bn3(self.fc3(x)).transpose(1, 2), inplace=True) #[B, features, num_points]

        x = torch.max(x, dim=2)[0]  # [B, features]

        x = F.relu(self.bn4(self.fc4(x)), inplace=True)
        x = F.relu(self.bn5(self.fc5(x)), inplace=True)

        x = torch.matmul(x, self.weights)
        x = x + self.bias
        x = x.reshape(-1, self.d, self.d)

        return x

class PointNet(nn.Module):

    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.tnet3d = TNet()
        self.mlp1 = MLP(num_features=[3, 64, 64])
        self.tnet64d = TNet(d=64)
        self.mlp2 = MLP(num_features=[64, 64, 128, 1024])
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        input_transform = self.tnet3d(x)
        x = torch.matmul(x, input_transform)
        x = self.mlp1(x)

        feature_transform = self.tnet64d(x)
        x = torch.matmul(x, feature_transform)
        x = self.mlp2(x)

        x = torch.max(x, 1)[0]
        #global_feature = x

        x = self.mlp3(x)

        return x, feature_transform

    def train_batch(self, x, y):
        self.optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred, feature_transform = self.forward(x)
        loss = self.criterion(pred, y)

        I_hat = torch.bmm(feature_transform, feature_transform.transpose(1, 2))
        I = torch.diag(torch.ones(64, device=device))
        reg_loss = ((I-I_hat)**2).sum()
        loss += 1e-3 * reg_loss
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def valid_batch(self, x, y):
        self.eval()
        x, y = x.to(device), y.to(device)
        pred, _ = self.forward(x)
        loss = self.criterion(pred, y)

        pred = torch.argmax(pred, dim=-1)
        acc_batch = (pred==y).float().mean().item()

        return acc_batch, loss.item()






class MLP(nn.Module):

    def __init__(self, num_features):
        super(MLP, self).__init__()

        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()

        for i in range(1, len(num_features)):
            self.linear_list.append(nn.Linear(num_features[i-1], num_features[i]))
            self.bn_list.append(nn.BatchNorm1d(num_features[i]))

    def forward(self, x):

        for i, linear in enumerate(self.linear_list):
            x = linear(x).transpose(1, 2)
            x = F.relu(self.bn_list[i](x), inplace=True)
            x = x.transpose(1, 2)

        return x


