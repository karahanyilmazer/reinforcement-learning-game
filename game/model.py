import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms
def add_dim(array):
    array = array[None, ...]
    return array
class ResBlock(nn.Module):
    def __init__(self, Fin, Fout, n_neurons=128):
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout
    
class Res_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = ResBlock(input_size, hidden_size)
        self.linear2 = ResBlock(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class HybridQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # self.resnet18 = models.resnet18(pretrained=True)
        # self.feature_extractor = nn.Sequential(self.resnet18.conv1,
        #                                        self.resnet18.bn1,
        #                                        self.resnet18.relu,
        #                                        self.resnet18.maxpool,
        #                                        self.resnet18.layer1,
        #                                        self.resnet18.layer2,
        #                                        self.resnet18.layer3,
        #                                        self.resnet18.layer4
        #                                        )
        self.feature_extractor = models.mobilenet_v3_small(pretrained=True).features
        self.feature_extractor.requires_grad_(False)
        self.feature_extractor.eval()

        self.q_net = Linear_QNet(input_size, hidden_size, output_size)
        weights = torch.load('./model/model_nobound_85.pth')
        self.q_net.load_state_dict(weights)
        print(weights.keys())
        self.q_net.requires_grad_(False)
        self.q_net.eval()

        # self.linear_output = nn.Linear(output_size + 8192, output_size)
        self.linear_output = nn.Linear(output_size + 576*4*4, hidden_size)
        self.linear_output2 = nn.Linear(hidden_size, output_size)
        self.linear_output2.weight = nn.Parameter(weights['linear2.weight'],True) # type: ignore
        self.linear_output2.bias = nn.Parameter(weights['linear2.bias'],True) # type: ignore
        self.linear_output.train()
        self.linear_output2.train()
        self.resize = transforms.Resize(100)
    
    def forward(self,img, x):
        # return self.feature_extractor(img)
        img = self.resize(img)
        # print(img.shape)
        cnn_out = self.feature_extractor(img).detach()
        cnn_out = torch.reshape(cnn_out, (-1,576*4*4))
        # cnn_out = torch.reshape(cnn_out, (-1,8192))
        qnet_out = self.q_net(x)
        hyqnet_in = torch.cat([cnn_out,qnet_out],1)
        lin_out = self.linear_output(hyqnet_in)
        return self.linear_output2(lin_out)
    
    def save(self, file_name='hybrid_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class HybridQTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.device = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    def train_step(self, frame, state, action, reward, next_frame, next_state, done):
        frame = torch.tensor(frame, dtype=torch.float, device = self.device)
        state = torch.tensor(state, dtype=torch.float, device = self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device = self.device)
        next_frame = torch.tensor(next_frame, dtype=torch.float, device = self.device)
        action = torch.tensor(action, dtype=torch.long, device = self.device)
        reward = torch.tensor(reward, dtype=torch.float, device = self.device)
        # (n, x)
        # print('state: ',state.shape)
        if state.shape[0] == 1:
            # (1, x)
            # frame = torch.unsqueeze(frame, 0)
            # state = torch.unsqueeze(state, 0)
            # next_state = torch.unsqueeze(next_state, 0)
            # next_frame = torch.unsqueeze(next_frame, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(frame, state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # print('next_frame[idx],next_state[idx]', next_frame[idx].shape, next_state[idx].shape)
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(add_dim(next_frame[idx]),add_dim(next_state[idx]))
                )

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

if __name__ == '__main__':
    model = HybridQNet(11, 256, 3)
    print(model.feature_extractor)
    img = torch.rand(10,3,640,640)
    resize = transforms.Resize(100)
    img = resize(img)
    x = torch.rand(10,11)
    inference = model(img,x)
    # print(inference.shape)
    inference.sum().backward()
    print(model.linear_output2.weight.grad)
    print(model.q_net.linear1.bias.grad)