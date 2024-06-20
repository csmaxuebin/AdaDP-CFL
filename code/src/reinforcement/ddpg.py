import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from random import uniform
import torch.nn.functional as F

class DDPG(object):
    def __init__(self, state_shape, action_shape):
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.memory_size = 500
        self.mem_cntr = 1  # 追踪存储位置
        self.memory = np.zeros((self.memory_size, self.state_shape * 2 + self.action_shape + 1), dtype=np.float32)
        self.noise = OUNoise(action_shape)

        # Actor Network and Target Network
        self.actor = ActorNetwork(state_shape, action_shape)
        self.actor_target = ActorNetwork(state_shape, action_shape)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic Network and Target Network
        self.critic = CriticNetwork(state_shape, action_shape)
        self.critic_target = CriticNetwork(state_shape, action_shape)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        self.actor.eval()  # 设置为评估模式
        with torch.no_grad():
            action = self.actor(state).squeeze(0)
        self.actor.train()  # 设置回训练模式
        noise = self.noise.sample()
        # print("Action shape:", action.shape)
        # print("Noise shape:", noise.shape)
        action = action + noise
        return action

    def soft_update(self, target, source, beta):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - beta) * target_param.data + beta * source_param.data)

    def learn(self,trans):

        index = self.mem_cntr % self.memory_size
        trans = np.array(trans[0], dtype=np.float32)
        self.memory[index] = trans
        self.mem_cntr += 1  # 更新计数器
        # Sample a batch

        filename = f'data/ddpg_memory.npz'
        np.savez(filename, memory=self.memory)

        max_mem = min(self.mem_cntr, self.memory_size)
        indices = np.random.choice(max_mem, size=self.batch_size)
        batch_transition = np.array(self.memory)[indices, :]

        # divide batch into [states, actions, rewards, next_states]
        batch_states = torch.from_numpy(batch_transition[:, :self.state_shape])
        batch_actions = torch.from_numpy(batch_transition[:, self.state_shape: self.state_shape + self.action_shape])
        batch_rewards = torch.from_numpy(batch_transition[:, self.state_shape + self.action_shape: self.state_shape + self.action_shape + 1])
        batch_next_states = torch.from_numpy(batch_transition[:, -self.state_shape:])

        batch_states = batch_states.float()  # 转换为浮点类型
        batch_actions = batch_actions.float()  # 转换为浮点类型
        batch_rewards = batch_rewards.float()
        batch_next_states = batch_next_states.float()

        # Train critic and get loss value
        critic_loss = self.train_critic(batch_states, batch_actions, batch_next_states, batch_rewards)

        # Train actor and get loss value
        actor_loss = self.train_actor(batch_states)

        # Update three target networks
        self.soft_update(self.actor_target, self.actor, self.beta)
        self.soft_update(self.critic_target, self.critic, self.beta)

        return actor_loss, critic_loss

    def train_critic(self, state, action, next_state, reward):
        # 计算目标 Q 值
        # 假设 self.critic_target 是目标批评家网络
        q_target_next = self.critic_target(next_state, action)
        q_target = reward + self.gamma * q_target_next
        # 获取当前预测 Q 值
        q_predicted = self.critic(state, action)
        # 计算 TD 误差
        critic_loss = F.mse_loss(q_predicted, q_target.detach())
        # 优化批评家网络
        self.critic_optimizer.zero_grad()  # 清除之前的梯度
        critic_loss.backward()  # 反向传播
        self.critic_optimizer.step()  # 更新参数
        return critic_loss.item()  # 返回损失值

    def train_actor(self, states):
        # 使用演员网络生成动作
        actions = self.actor(states)
        # 使用批评家网络评估这些动作，计算 Q 值
        q_values = self.critic(states, actions)
        # 计算演员网络的损失
        actor_loss = -torch.mean(q_values)
        # 优化演员网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()

    def save_model(self, model_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
