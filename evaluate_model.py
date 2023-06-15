from dqn import DQN
from numpy import mean, std
import argparse
import torch
from itertools import count
from bit_flip_env import BitFlipEnv


def eval_model(model, env, episodes=10, device='cpu'):
    # turn the model into a evaluation mode
    model.eval()
    steps_done = []
    ep_reward_array = []
    successes = 0.0
    
    # tests some episodes
    for i in range(episodes):
        # Initialize the environment and state
        state = torch.tensor(env.reset(), device=device, dtype=torch.float).unsqueeze(0)
        goal = torch.tensor(env.target, device=device, dtype=torch.float).unsqueeze(0)

        episode_reward = 0

        for t in count():                                                   # run one episode
            # Select and perform an action
            state = torch.cat((state, goal), 1)
            action = model(state).max(1)[1].view(1, 1)
            # interact with the envionment
            next_state, reward, done = env.step(action.item())

            # In the dynamic goal case, the goal might change every step
            # update the target
            goal = torch.tensor(env.target, device=device, dtype=torch.float).unsqueeze(0)

            episode_reward += reward
            # update the state
            state = torch.tensor(next_state, device=device, dtype=torch.float).unsqueeze(0)
            
            # conclude
            if done:
                steps_done.append(t + 1)
                if reward == 0:
                    successes += 1
                break
        
        # get the episode reward
        ep_reward_array.append(episode_reward)
    
    # get the average success rate
    success_rate = successes / episodes

    print('Evaluation done, successful episodes rate = %.2f average steps done = %.1f, average accumulated reward = %.1f, std = %.1f'
          % (success_rate, float(mean(steps_done)),  float(mean(ep_reward_array)), float(std(ep_reward_array))))
    
    # tune the model back to training mode for the next trainings
    model.train()
    return ep_reward_array, success_rate


def run_test(args):
    # get the device and the env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = BitFlipEnv(size=args.state_size, shaped_reward=args.shaped_reward, dynamic=args.dynamic)
    actions_num = env.size
    states_dim = env.size

    # Create Model
    # Load the model from the .pkl file
    model = DQN(states_dim*2, args.hidden_dim, actions_num).to(device)
    model.load_state_dict(torch.load('bit_flip_model.pkl'))
    print("Starting test on Bit-Flip environment, with DQN method.")
    # run an test episode to look at the performance of the current model
    eval_model(model, env, episodes=args.episodes, device=device)
    print("Test done.")


if __name__ == '__main__':
    # this is the test for the multi-flip game
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run')
    parser.add_argument('--hidden-dim', type=int, default=500, help='Hidden layer dimension')
    parser.add_argument('--state-size', type=int, default=4, help='Size of the environments states')
    parser.add_argument('--shaped-reward', action="store_true", help='Use shaped reward instead of the original reward')

    # Dynamic environment
    parser.add_argument('--dynamic', action="store_true", help="Use the dynamic mode for the bit flip environment")

    args = parser.parse_args()
    
    # run the test
    run_test(args)
