import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

class DuelingDQN(nn.Module):
    def __init__(self, n_action, input_shape=(4, 84, 84)):
        super(DuelingDQN, self).__init__()
        self.n_action = n_action
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        r = int((int(input_shape[1] / 4) - 1) / 2) - 3
        c = int((int(input_shape[2] / 4) - 1) / 2) - 3
        self.adv1 = nn.Linear(r * c * 64, 512)
        self.adv2 = nn.Linear(512, self.n_action)
        self.val1 = nn.Linear(r * c * 64, 512)
        self.val2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)
        val = F.relu(self.val1(x))
        val = self.val2(val)
        return val + adv - adv.mean(1, keepdim=True)

    def calc_priorities(self, target_net, transitions, alpha=0.6, gamma=0.999,
                        detach=False,
                        device=torch.device("cpu")):
        batch = utils.Transition(*zip(*transitions))

        next_state_batch = torch.stack(batch.next_state).to(device)
        state_batch = torch.stack(batch.state).to(device)
        action_batch = torch.stack(batch.action).to(device)
        reward_batch = torch.stack(batch.reward).to(device)
        done_batch = torch.stack(batch.done).to(device)

        state_action_values = self.forward(state_batch).gather(1, action_batch)
        next_action = self.forward(next_state_batch).argmax(dim=1).unsqueeze(1)
        next_state_values = target_net(next_state_batch).gather(1, next_action).detach()
        expected_state_action_values = (next_state_values * gamma * (1.0 - done_batch)) + reward_batch
        delta = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduce=False)
        if detach:
            delta = delta.detach()
        prios = (delta.abs() + 1e-5).pow(alpha)
        return delta, prios.detach()

class DuelingLSTMDQN(nn.Module):
    def __init__(self, n_action, batch_size, input_shape=(4, 84, 84)):
        super(DuelingLSTMDQN, self).__init__()
        self.n_action = n_action
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        r = int((int(input_shape[1] / 4) - 1) / 2) - 3
        c = int((int(input_shape[2] / 4) - 1) / 2) - 3
        self.lstm = nn.LSTMCell(r * c * 64, 512)
        self.adv1 = nn.Linear(512, 512)
        self.adv2 = nn.Linear(512, self.n_action)
        self.val1 = nn.Linear(512, 512)
        self.val2 = nn.Linear(512, 1)
        self.cx = torch.zeros(batch_size, 512)
        self.hx = torch.zeros(batch_size, 512)

    def reset(self, done=False):
        self.cx.detach_()
        self.hx.detach_()
        if done:
            self.cx.zero_()
            self.hx.zero_()

    def get_state(self):
        return self.hx.detach().cpu(), self.cx.detach().cpu()

    def set_state(self, state, device):
        hx, cx = state
        self.hx = hx.to(device)
        self.cx = cx.to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        adv = F.relu(self.adv1(hx))
        adv = self.adv2(adv)
        val = F.relu(self.val1(hx))
        val = self.val2(val)
        return val + adv - adv.mean(1, keepdim=True)

    def calc_priorities(self, target_net, transitions, n_burn_in=40,
                        eta=0.9, gamma=0.997,
                        detach=False,
                        device=torch.device("cpu")):
        batch = utils.Transition(*zip(*transitions))

        next_state_batch = torch.stack(batch.next_state).to(device)
        state_batch = torch.stack(batch.state).to(device)
        action_batch = torch.stack(batch.action).to(device)
        reward_batch = torch.stack(batch.reward).to(device)
        done_batch = torch.stack(batch.done).to(device)

        state_action_values = self.forward(state_batch).gather(1, action_batch)
        next_action = self.forward(next_state_batch).argmax(dim=1).unsqueeze(1)
        next_state_values = target_net(next_state_batch).gather(1, next_action).detach()
        expected_state_action_values = utils.rescale((utils.inv_rescale(next_state_values) * gamma * (1.0 - done_batch)) + reward_batch)
        delta = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduce=False)
        if detach:
            delta = delta.detach()
        prios = eta * delta.max() + (1.0 - eta) * delta.mean()
        return delta, prios.detach()
