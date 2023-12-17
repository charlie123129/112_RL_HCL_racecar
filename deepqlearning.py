import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from play_and_evaluate import play
import racecar_gym.envs.gym_api
from collections import OrderedDict
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 創建環境
scenario = './scenarios/austria.yml'
env = gymnasium.make(
    id='SingleAgentRaceEnv-v0', 
    scenario=scenario,
    render_mode='rgb_array_birds_eye', # optional: 'rgb_array_birds_eye'
)

# DQN參數
state_size = (1080 + 6 + 1,)  # lidar (1080,) + pose (6,) + time (1,)
action_size = 2  # speed and steering
batch_size = 64
n_episodes = 1
gamma = 0.93  # 折扣因子
epsilon = 1.0  # 探索率
epsilon_min = 0.01
epsilon_decay = 0.995
memory_capacity = 1000  # 增加经验回放容量


# 構建DQN模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return self.fc3(x)
    
    def choose_action(self, state):
        with torch.no_grad():
            action_values = self(state)
            action_speed = action_values[0][0].item()
            action_steering = action_values[0][1].item()
            action = [action_speed, action_steering]
            return action

        
model = DQN(state_size, action_size).to(device)

optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

# 經驗回放
memory = deque(maxlen=memory_capacity)

total_rewards = []
losses = []

# DQN訓練過程
for e in range(n_episodes):
    collision_status = False
    wrong_way_status = False
    observation, state_info = env.reset()
    

    state = np.concatenate([observation['lidar'], observation['pose'], [observation['time']]])
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    #state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    done = False
    total_reward = 0  # 初始化總reward
    
    while not done:
        # ε-greedy策略選擇動作
        if np.random.rand() <= epsilon:
            action_dict = env.action_space.sample()
            action_speed = action_dict['speed'][0]  # Extract speed value
            action_steering = action_dict['steering'][0]  # Extract steering value
            action = [action_speed, action_steering]
        else:
            #with torch.no_grad():
            #    action_values = model(state)
            #    action_speed = action_values[0][0].item()
            #    action_steering = action_values[0][1].item()
            #    action = [action_speed, action_steering]
            action_values = model.choose_action(state)
            action_dict = {'speed': [action_values[0]], 'steering': [action_values[1]]}

        
        #result = env.step(action)
        observation, reward, done, _, state_info = env.step({'speed': action_speed, 'steering': action_steering})

        #observation, reward, done, _ ,state_info= env.step(action_dict)

        #自定義reward機制
        progress_reward = state_info['progress'] * 10

        # 如果没有碰到墙壁，给予正奖励；如果碰到墙壁，当回合reward归零
        collision_penalty = -10 if state_info['wall_collision'] else 0
        if state_info['wall_collision']:
            collision_status = True  # 更新碰撞状态
        #elif collision_status:
        #    # 如果之前是碰撞状态，现在没有碰撞，给予正面奖励
        #    collision_recovery_reward = 20
        #    collision_status = False  # 重置碰撞状态
        else:
            collision_recovery_reward = 1


        wrong_way_penalty = -100 if state_info['wrong_way'] else 0
        if state_info['wrong_way']:
            wrong_way_status = True  # 更新碰撞状态
        #elif wrong_way_status:
            # 如果之前是碰撞状态，现在没有碰撞，给予正面奖励
        #    wrong_way_recovery_reward = 20
        #    wrong_way_status = False  # 重置碰撞状态
        else:
             wrong_way_recovery_reward = 1


        
        total_reward += progress_reward + collision_penalty + collision_recovery_reward+wrong_way_penalty+wrong_way_recovery_reward
        if total_reward <= -100:
            print(f"Episode: {e+1}/{n_episodes}, Total Reward: {total_reward} (Terminated early)")
            break  # 提前結束當前回合

        print(f"episode: {e+1}/{n_episodes},progress_reward: {progress_reward}, collision_penalty: {collision_penalty}, collision_recovery_reward: {collision_recovery_reward}, wrong_way_penalty: {wrong_way_penalty}, wrong_way_recovery_reward: {wrong_way_recovery_reward},total_reward: {total_reward}")

        
        # 總reward
        #total_reward = progress_reward + collision_penalty + speed_reward + wrong_way_penalty #+ acceleration_penalty
        

        # 執行動作並獲得新的觀察值和狀態
        next_state = np.concatenate([observation['lidar'], observation['pose'], [observation['time']]])
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        #next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        
        # 存儲經驗
        memory.append((state, action, total_reward, next_state, done))
        
        # 移動到新狀態
        state = next_state
        
        # 經驗回放
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                state = state.to(device)
                next_state = next_state.to(device)
                #state = state
                #next_state = next_state＼


                target = reward + (gamma * torch.max(model(next_state)).item() * (not done))
                
                
                predicted_q_values = model(state)
                
                target_q_values = predicted_q_values.clone()
                target_q_values[0][0] = target if action[0] == action_speed else predicted_q_values[0][0]
                target_q_values[0][1] = target if action[1] == action_steering else predicted_q_values[0][1]
                loss = loss_fn(target_q_values, predicted_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    # 減少ε
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    # 每個episode結束後打印信息
    if total_reward > -100:
        print(f"Episode: {e+1}/{n_episodes}, Total Reward: {total_reward}")
    # 每個episode結束後打印信息
    #print(f"Episode: {e+1}/{n_episodes}")
    total_rewards.append(total_reward)
    losses.append(loss.item())
    print(f"Episode: {e+1}/{n_episodes}, Total Reward: {total_reward}")

# Plotting the rewards
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title('Total Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(total_rewards)
plt.legend()
plt.subplot(1, 2, 2)
plt.title('Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.plot(losses)
plt.legend()
plt.tight_layout()


model_save_dir = './dqn/'
best_reward = -float('inf')  # 初始最佳奖励设置为负无穷大
output_path = './dqn/racecar_trained.mp4'
plt.savefig(model_save_dir+'traindqn.png')
plt.close
play(env=env, policy=model, best_reward=None, model_save_dir=model_save_dir, output_path=output_path, device=device)

# 關閉環境
env.close()