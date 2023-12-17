import os
import cv2
import torch
import numpy as np
from itertools import chain
from torch import distributions as dists

def play(env = None, policy = None, best_reward: float | None=None, model_save_dir = None, output_path=None, episode_index:int | None=None,device=torch.device("cuda" if torch.cuda.is_available() else "cpu") ):
    ''' Play one round and record video.

    For evaluation during training, pass in `episode_index` instead of `output_path` for video name.
    '''
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        render_env = env
        reset_options = dict(mode='grid')
        observation, info = render_env.reset(options=reset_options)

        if output_path is None:
            output_path = f'./videos/racecar-episode-{episode_index}.mp4'
        
        if not os.path.exists(os.path.dirname(output_path)):
            os.mkdir(os.path.dirname(output_path))

        output_video_writer = cv2.VideoWriter(filename=output_path,
                                              fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                              fps=10,
                                              frameSize=(640,480))
        
        total_rewards = 0
        length = 0
        print(f'[Evaluation] Running evaluation and recording video for episode-{episode_index}')
        while True:
            # preprocess
            observation.pop('time') # pop out redundant observation
            observation = list(chain.from_iterable(observation.values())) # concat all observations
            obs_array = np.expand_dims(observation, axis=0) # state shape: original(s_dim) -> expanded(1,s_dim)
            
            obs_array = obs_array[:,:1087]
            # convert to tensor
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)
            # predict action
            if policy is not None:
                mean, std = policy(obs_tensor) # Get mean and std from policy
                dist = dists.Normal(mean, std)  # Create a normal distribution
                action = dist.sample()  # Sample an action from the distribution

                # Check if action has the correct shape and reshape if necessary
                if action.shape != (2,):
                    action = action.view(2)

                action_dict = {
                    'speed': action.cpu().numpy()[0],
                    'steering': action.cpu().numpy()[1]
                }
            else:
                action = render_env.action_space.sample() # random sample action
                action_dict = action # Assuming 'action' is already a dictionary



            observation, rewards, done, _, states = render_env.step(action_dict)

            if length % 10 == 0: # save 1 frame per 10 step
                image = render_env.render().astype(np.uint8)
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # cv2.imshow('img',frame)
                # key = cv2.waitKey(1)
                output_video_writer.write(frame)
            
            total_rewards += rewards
            length += 1
            
            if done:
                print("[Evaluation] total reward = {:.6f}, length = {:d}".format(total_rewards, length), flush=True)
                # save the best model with best reward
                if policy is not None and model_save_dir is not None:
                    if (total_rewards > best_reward) and (length > 50):
                        best_reward = total_rewards
                        model_filename = f"best_agent_episode_{episode_index}.pt"
                        print(f"\nSaving the best model as {model_filename} ... ", end="")
                        torch.save({
                            "policy": policy.state_dict(),
                        }, os.path.join(model_save_dir, model_filename))
                        print("Done.")
                break
        
        # cv2.destroyAllWindows()
        output_video_writer.release()
        render_env.close()
        
        return best_reward