import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import yaml
from play_and_evaluate import play
import racecar_gym.envs.gym_api
import matplotlib.pyplot as plt
import numpy as np
import torch.distributions as dists
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建环境
scenario = './scenarios/austria.yml'
env = gymnasium.make(
    id='SingleAgentRaceEnv-v0', 
    scenario=scenario,
    render_mode='rgb_array_birds_eye',
    #metadata={'render_fps': 30},
)
env.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(1087,), dtype=np.float32)

# DQN参数
state_size = (1080 + 6 + 1,)# lidar (1080,) + pose (6,) + time (1,)
action_size = 2  # speed and steering
batch_size = 1024
n_episodes = 155
gamma = 0.93  # 折扣因子

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        # define your network architecture
        self.fc1 = nn.Linear(state_size[0], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = self.fc4(x)
        log_std = self.log_std.expand_as(mean)  # 扩展 log_std 的维度，使其与均值的维度匹配
        std = torch.exp(log_std)
        return mean, std  # 返回均值和标准差

    def choose_action(self, state, deterministic=False):  # 添加了 deterministic 参数
        # 检查状态是否已经是一个张量
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.FloatTensor(state).to(device)
        else:
            # 如果已经是张量，直接使用
            state_tensor = state.to(device)
        
        #dist = self.forward(state_tensor)
        #action = dist.sample()
        return action.cpu().numpy()
    
class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        # 定义你的网络结构
        self.fc1 = nn.Linear(state_size[0], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.tanh(x)  # outputs a tensor

# 实例化模型和优化器
actor = Actor(state_size, action_size).to(device)
critic = Critic(state_size).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=0.0005)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.0005)

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_update(actor, critic, actor_optimizer, critic_optimizer, states, actions, log_probs, returns, values, entropy, clip_param=0.2, ppo_epochs=10, mini_batch_size=64):
    returns = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values = torch.cat(values).detach()
    states = torch.cat(states)
    actions = torch.cat(actions)
    
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    for _ in range(ppo_epochs):
        perm = torch.randperm(states.size(0))
        for i in range(0, states.size(0), mini_batch_size):
            idxs = perm[i:i+mini_batch_size]
            state, action, old_log_prob, return_, advantage = states[idxs], actions[idxs], log_probs[idxs], returns[idxs], advantages[idxs]
            
            dist, value = actor(state), critic(state)
            entropy = dist.entropy().mean()
            new_log_prob = dist.log_prob(action)

            ratio = (new_log_prob - old_log_prob).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            critic_loss = (return_ - value).pow(2).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

def extract_data_from_observation(observation):
    # 提取各部分数据
    lidar_data = observation['lidar']
    pose_data = observation['pose']
    time_data = np.array([observation['time']])  # 将时间转换为数组
    
    # 合并所有数据到一个列表
    state = np.concatenate((lidar_data, pose_data, time_data), axis=0)
    return state

# PPO 训练循环
for e in range(n_episodes):
    model_save_dir = './ppo/'
    best_reward = -float('inf')  
    output_path = './ppo/racecar_trained.mp4'

    total_reward = 0
    log_probs = []
    values = []
    states = []
    actions = []
    rewards = []
    masks = []
    entropy = 0
    last_progress=0

    # 环境重置时
    observation, state_info = env.reset()
    state = extract_data_from_observation(observation)
    state_tensor = torch.FloatTensor(state).to(device)  # 将提取的数据转换为 PyTorch 张量

    done = False  # 这里初始化 done

    while not done:
        
        ## Get the distribution from the actor
        mean, std = actor(state_tensor)
        dist = dists.Normal(mean, std)  # 创建正态分布对象
        action = dist.sample()  # 从分布中抽样动作
        # Sample an action from the distribution
        #action = dist.sample()
        # Get the value prediction from the critic
        value = critic(state_tensor)
        
        action_dict = {
            'speed': action.cpu().numpy()[0],
            'steering': action.cpu().numpy()[1]
        }

        # After each step
        next_observation, _, done, _, next_state_info = env.step(action_dict)
        next_state = extract_data_from_observation(next_observation)
        next_state_tensor = torch.FloatTensor(next_state).to(device)
            
        # ... 奖励计算 ...
        progress_reward = state_info['progress'] * 10

        state_progress = state_info["progress"]
        progress_reward1 = (state_progress - last_progress) * 1000000
        last_progress = state_progress

        speed_reward = observation['velocity'][0] * 5
        lateral_speed_penalty = -abs(observation['velocity'][1]) * 2
        angular_velocity_penalty = -np.sum(np.abs(observation['velocity'][3:])) * 1
        velocity_reward = speed_reward + lateral_speed_penalty + angular_velocity_penalty
        wrong_way_penalty = -25 if state_info['wrong_way'] else 5
        collision_penalty = -50 if state_info['wall_collision'] or state_info['opponent_collisions'] else 5
        time_penalty = -0.05 * state_info['time']
        checkpoint_reward = 100 * (state_info['checkpoint'])
        if observation['lidar'].min()<0.1:
            lidar_penalty=-100
        else:
            lidar_penalty=0
        current_reward = progress_reward + wrong_way_penalty + collision_penalty + time_penalty + checkpoint_reward + \
            + velocity_reward+progress_reward1+lidar_penalty
        # 更新
        total_reward += current_reward

        print(f'episode: {e+1}/{n_episodes}, progress: {progress_reward}, speed:{velocity_reward}, collision: {collision_penalty}, wrong_way: {wrong_way_penalty}, time:{time_penalty}, checkpoint:{checkpoint_reward}, total_reward: {total_reward}')
        
        log_prob = dist.log_prob(action).unsqueeze(0)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([current_reward], dtype=torch.float32).to(device))  # 用 current_reward 替换了未定义的 reward
        masks.append(torch.tensor([1-done], dtype=torch.float32).to(device))
        
        states.append(state_tensor)  # 应存储 state_tensor
        actions.append(action)
        
        state_tensor = next_state_tensor  # 更新 state_tensor 为下一个状态的 PyTorch 张量
        observation = next_observation
        state_info = next_state_info
        
    print(f"Episode: {e+1}/{n_episodes}, Total Reward: {total_reward}")
    if (e + 1) % 10 == 0:
        output_path = f'./ppo/racecar_trained_episode_{e+1}.mp4'
        episode_index=f'{e+1}'
        evaluated_reward=play(env=env, policy=actor, best_reward=best_reward, model_save_dir=model_save_dir, output_path=output_path, device=device,episode_index=episode_index)
        evaluated_reward
        # Update best reward if the evaluated reward is better
        if evaluated_reward is not None and evaluated_reward > best_reward:
            best_reward = evaluated_reward
            print(f"New best reward: {best_reward} achieved at episode {e+1}")
        #plot_results(episode_rewards, actor_losses, critic_losses, f'episode_{e+1}')


play(env=env, policy=actor, best_reward=best_reward, model_save_dir=model_save_dir, output_path=output_path, device=device,episode_index='last')
#plot_results(episode_rewards, actor_losses, critic_losses, 'last')
env.close()
