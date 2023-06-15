from bit_flip_env import BitFlipEnv
from utils import *
from dqn import DQN
import time
import torch
import argparse
from itertools import count
import random
from evaluate_model import eval_model
import numpy as np


def select_action(args, state, goal, actions_num, policy_net, steps_done, device):
    # use this function to select the action from the Q function evaluation of DQN algorithm
    # get a sample between 0 and 1, for next decision of using random or use policy
    sample = random.random()
    # calculate the current eps
    eps_threshold = max(args.eps_end,
                        args.eps_start * (1 - steps_done / args.eps_decay) +
                        args.eps_end * (steps_done / args.eps_decay))
    # get the actions
    if sample > eps_threshold:
        # use with to ensure proper aqusition and release the resource
        # make sure we do not record the gradients when we calculate the actions from Q function
        with torch.no_grad():
            # the input the the (s, g), output is maximum of the Q value for all possible actions
            # the max return a tuple (), of the maximum value and the position
            # look at the max in second dim, since the first dim is batch
            return policy_net(torch.cat((state, goal), 1)).max(1)[1].view(1, 1)
    else:
        # just return a random action for the one batch, with proper data type, the int / long type
        return torch.tensor([[random.randrange(actions_num)]], device=device, dtype=torch.long)


def optimize_model(args, policy_net, target_net, optimizer, memory, device):
    # use the transition from the replay buffer to update the Q-function
    # First check if there is an available batch
    if len(memory) < args.batch_size:
        # skip if the replay is too small
        return 0

    # Sample batch transitions from the replay buffer to train the networks
    transitions = memory.sample(args.batch_size)

    # Transpose the batch
    # make a dictionary from the raw data of transition (I just doubt why use dict, why not just list)
    # use dict may let the data more readable
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # get the intermediate trans, and make them to tensor, make a mask first
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    # then get and concanate all the s, a, r, s'
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state, dim=0)
    action_batch = torch.cat(batch.action, dim=0)
    reward_batch = torch.cat(batch.reward, dim=0)

    # Compute Q(s_t, a) - the model computes Q(s_t),
    # Then, using gather, we select the columns of actions taken
    # Compute the Q(s, a) for all a (a list of prob), 
    # use torch.gather() to take the output Q-value of the current a， -> Q(s_t, a_t)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # get the calculated Q from the max[Q(s_t+1)]
    next_state_values = torch.zeros((args.batch_size, 1), device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].unsqueeze(1).detach()

    # Compute the expected Q values
    # updated the Q-function by using the bootstraping: Q(s,a) = r + gamma * max(Q(s_t+1, a_t+1) 
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch.float()

    # Compute Huber loss
    # train the Q-func to the target value, use smooth L1 loss and also add the regularization
    loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
    if args.reg_param > 0:
        regularization = 0.0
        for param in policy_net.parameters():
            regularization += torch.sum(torch.abs(param))
        loss += args.reg_param * regularization

    # Optimize the model
    # Optimize the model base on the loss (get the grad and step one step)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    # finally return the loss for record data
    return loss.item()


def train(args):
    # the main function for the training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reset Bit-Flip environment
    # create the env
    env = BitFlipEnv(size=args.state_size, shaped_reward=args.shaped_reward, dynamic=args.dynamic)
    # get the state dim and the range of the action (the action_dim = 1)
    actions_num = env.size
    states_dim = env.size

    # Statistics
    # record the training process data
    steps_done = 0                          # get the total step
    reset_target_cnt = 0                    # get the target reset data
    episode_durations = []                  # get the episode length for the performance of the policy
    eval_reward_array = []                  # get the reward evaluation
    success_ratio_array = []                # get the success ratio data

    # Experience Replay: memory of available transitions
    # create the replay buffer
    memory = ReplayMemory(100000)           # set a very larger capacity

    # Create DQN models
    # Build the Q networks for the DQN, and also build the target network for the trick
    # just a simple two-layer network with relu
    # also put the model to the devices
    # the action can estimate the Q-function of all actions what followed by the state: S -> Q(S,A)
    # notice here, the input is s, and we just assume all the a are option and output Q(s, a) for all action
    # the input will include the target/goal: (s, g), so we need input_dim = 2*state_dim
    # the output will be the Q func for all action: Q(s, ai) for all ai, thus output_dim = action_choices
    policy_net = DQN(states_dim * 2, args.hidden_dim, actions_num, dropout_rate=args.dropout).to(device)
    target_net = DQN(states_dim * 2, args.hidden_dim, actions_num, dropout_rate=args.dropout).to(device)

    # Create optimizer
    # get the optimizer for all the trainable parameters, with the learning rate
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.alpha)

    # Failed episodes buffer for DHER
    # use the buffer for the failed episodes
    failed_episodes = []
    failed_episodes_size = 10
    failed_episodes_index = 0

    # Main loop
    # run enough episodes
    for i_episode in range(args.episodes):
        # Initialize the environment and state
        # get the init state and target and reset the envs
        state = torch.tensor(env.reset(), device=device, dtype=torch.float).unsqueeze(0)
        goal = torch.tensor(env.target, device=device, dtype=torch.float).unsqueeze(0)
        
        # prepare for the rewards and common buffer
        episode_reward = 0
        episode_memory = []
        # the count here is import from the itertools, the itertools.count() create infinite sequence, (count(start=a, step=b))
        # it can also return the count start from the start, and then step
        # similiar to while(True):
        for t in count():
            # Select and perform an action
            # get the action base on the Q function evaluation, this is for 1-batch state
            action = select_action(args, state, goal, actions_num, policy_net, steps_done, device)
            # get the value of the action, and send it into env to step
            next_state, reward, done = env.step(action.item())
            
            # get the old goal in advance
            old_goal = goal.clone()

            # In the dynamic goal case, the goal might change every step
            # this is the changing target case, get the new goal / target
            goal = torch.tensor(env.target, device=device, dtype=torch.float).unsqueeze(0)

            # Next state to tensor
            # convert the next state
            next_state = torch.tensor(next_state, device=device, dtype=torch.float).unsqueeze(0)
            
            # get the statistic updated
            steps_done += 1
            reset_target_cnt += 1
            episode_reward += reward
            
            # put them into tensor
            reward = torch.tensor([reward], device=device).unsqueeze(0)

            # Store the transition in memory
            # put the transition into standard buffer
            if not (done and reward < 0):
                episode_memory.append((state, action, next_state, reward, old_goal, goal))

            # Move to the next state
            # update the old state to the current state
            state = next_state

            # Update the target network
            # update the target net if need
            if reset_target_cnt > args.target_update:                   # freq of target update, set to be 500
                reset_target_cnt = 0
                # update / copy the model
                target_net.load_state_dict(policy_net.state_dict())

            # Episode duration statistics
            # check if one episode is already down
            if done:
                episode_durations.append(t + 1)
                # see the final step success or not, if fail, then put the whole buffer in a "fail" buffer
                if reward < 0:
                    # This is a failed trajectory, see if enough space, use a pointer to update
                    if len(failed_episodes) < failed_episodes_size:
                        failed_episodes.append(episode_memory)
                    else:
                        failed_episodes[failed_episodes_index] = episode_memory
                        failed_episodes_index = (failed_episodes_index + 1) % failed_episodes_size
                # break the episode        
                break
        
        # after one episode, we can go to deal with the replay and set the additional goals
        # Experience Replay
        for t in range(len(episode_memory)):
            # get one transition from the memory, here the t is the current time step
            state, action, next_state, reward, old_goal, goal = episode_memory[t]
            
            # get the (s,g)
            state_memory = torch.cat((state, goal), 1)
            # get the next (s,g)
            if torch.all(next_state == goal):
                next_state_memory = None
            else:
                next_state_memory = torch.cat((next_state, goal), 1)
            
            # put the transition which can be used for training into the replay buffer
            memory.push(state_memory, action, next_state_memory, reward)
            
            """
            deal with the replay buffer for the HER, for each timestep we look after to create new goals
            here the implementations is:
            for each timestep, sample some new goal, and see include the current timestep in the new goal trajectory
            in this way, the new goal don't have complete trajectory, but some sampled, discrete ones
            this seems the original method for HER
            this is a easier method which can fuse into the common replay buffer, but seems not very balance
            it have higher change to reach the final states of the episode and thus take longer time
            and it will garrenty to get a success state at the final end of the epoch
            """
            # I would suggest to also sample some accessable / reachable state, and update the whole trajectory before timestep
            # and then put them in the replay buffer, maybe not that efficient
            #HER
            if args.HER:
                # set limited new goals for the HER
                for g in range(args.goals):
                    # sample a timestep after or include the t (both include, please change to (t, len(episode_memory)-1) if error
                    future_goal = np.random.randint(t, len(episode_memory))
                    # get the s' of the old transition, which is an accessable state
                    _, _, new_goal, _, _, _ = episode_memory[future_goal]
                    # then, generate the new transition by using the changed goal, use the s in t step and s' in the future step
                    state_memory = torch.cat((state, new_goal), 1)
                  
                    if torch.all(next_state == new_goal):  # Done           # if the s' in t is same as the new goal, then it finishes
                        next_state_memory = None
                        # change the reward (here noticing it just change one step near the goal position)
                        reward = torch.zeros(1, 1)                          
                    else:
                        next_state_memory = torch.cat((next_state, new_goal), 1)
                        # also change the reward, here it cannot reach the new goal in one step and get a invalid transition
                        reward = torch.zeros(1, 1) - 1.0
                    memory.push(state_memory, action, next_state_memory, reward)
                    
        """
        this is the DHER for changing targets, this is outside the common replay buffer update
        
        """
        # DHER
        if args.DHER:
            finish = False                                                      # set a finish flag
            for i_ep, failed_ep_i in enumerate(failed_episodes):                # deal with the failed episodes and try to stitch other
                for j_ep, failed_ep_j in enumerate(failed_episodes):
                    # get every two trajectory combination
                    if i_ep == j_ep:
                        continue
                    # then go to get two time-steps from the two diff trajectory
                    for i_i, t_i in enumerate(failed_ep_i):
                        for j_j, t_j in enumerate(failed_ep_j):
                            # Checks if the ith episode's next state is the same as the jth episode's next goal
                            # Check if one of the experience state of (i) is actually the goal of other traj (j)
                            if torch.all(t_i[2] == t_j[5]):
                                # the trajectory is: "state, action, next_state, reward, old_goal, goal = episode_memory[t]"
                                # if the goal of (i) is found in (j), then try to rewrite the (i) goals,
                                # to make it want merge into the (j)
                                m = min(i_i, j_j)                                       # get the earlier timestep
                                # rewrite all the goal and the reward before this timestep
                                for t in range(m, -1, -1):
                                    new_current_goal = failed_ep_j[j_j - t][4]          # get the current goal of (j)
                                    new_next_goal = failed_ep_j[j_j - t][5]             # get the next goal of (j)
                                    next_state = failed_ep_i[i_i - t][2]                # get the next state of origianl (i)
                                    if torch.all(next_state == new_next_goal): 
                                        # if get to the target, then None for the next state
                                        next_state_memory = None
                                        reward = torch.zeros(1, 1)
                                    else:
                                        # otherwise get a proper next state and reward
                                        next_state_memory = torch.cat((next_state, new_next_goal), 1)
                                        reward = torch.zeros(1, 1) - 1.0
                                    state_memory = torch.cat((failed_ep_i[i_i - t][0], new_current_goal), 1)
                                    action = failed_ep_i[i_i - t][1]
                                    # put the new trajectory of the (i) with new goal and the reward into the replay buffer
                                    memory.push(state_memory, action, next_state_memory, reward)
                                finish = True
                            if finish:
                                break
                        if finish:
                            break
                    # we just need to find one possible reachable goal in (j)s for one (i)
                    if finish:
                        break
        
        # after one episode
        # Perform one step of the optimization (on the target network)
        # Then, Perform some step of the network update (on the Q function), by using the replay buffer
        optimization_steps = 5
        loss = 0.0
        for _ in range(optimization_steps):
            # go to update the Q-func, go to update the relevent network for RL
            loss += optimize_model(args, policy_net, target_net, optimizer, memory, device)
        # get the averaged Q-func approx loss
        loss /= optimization_steps

        # Episodes statistics
        # evaluate the model every 10 qpisodes
        if i_episode % 10 == 0 and i_episode != 0:
            print("Evaluation:")
            eval_reward, success_ratio = eval_model(model=policy_net, env=env, episodes=10, device=device)
            eval_mean_reward = np.mean(eval_reward)
            eval_reward_array.append(eval_mean_reward)
            success_ratio_array.append(success_ratio)
        
        # print out the date every 10 episodes
        print("Episode %d complete, episode duration = %d, loss = %.3f, reward = %d" %
              (i_episode, episode_durations[-1], loss, episode_reward))
        
        # save the model every 10 episodes
        torch.save(policy_net.state_dict(), 'bit_flip_model.pkl')
    
    # save the success_ratio after all the episode
    np.save("success_ratio.npy", success_ratio_array)
    print('Complete')


if __name__ == '__main__':
    # the main file for the training, primarily use DQN for training
    
    # use the parser
    parser = argparse.ArgumentParser(description=None)

    # Run parameters, training param
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size to train on')           # batch size of exper clips
    parser.add_argument('--episodes', type=int, default=5000, help='Amount of train episodes to run')   # eposide limit
    parser.add_argument('--state-size', type=int, default=4, help='Size of the environments states')    # size of states/game
    parser.add_argument('--shaped-reward', action="store_true")                                         # shaped reward                 

    # Model arguments
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')                        # network dropout for overfit
    parser.add_argument('--hidden-dim', type=int, default=500, help='Dimension of the hidden layer')    # network shape/dim for middle

    # Optimizer arguments
    parser.add_argument('--alpha', type=float, default=0.001, help='Alpha - Learning rate')             # learning rate
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma - discount factor')            # discount
    parser.add_argument('--target-update', type=int, default=500, help='Number of steps until updating target network')
                                                                                                        # target update frequency
    parser.add_argument('--reg-param', type=float, default=0, help='L1 regulatization parameter')       # network regulation for overfit
   

    # Epsilon - Greedy arguments
    # the epsilon-greedy desicion how we choose the random action or an action from the policy
    parser.add_argument('--eps-start', type=float, default=1.0, help='Starting epsilon - in epsilon greedy method')
    parser.add_argument('--eps-end', type=float, default=0.1,
                        help='Final epsilon - in epsilon greedy method. When epsilon reaches this value it will stay')
    # this is a little bit large, since the max_step in the env is only 200
    parser.add_argument('--eps-decay', type=int, default=500000,
                        help='Epsilon decay - how many steps until decaying to the final epsilon')

    # Hindsight Experience Replay (HER) arguments
    # the para and flags for the HER, the action=stor_ture set the arg to T when this arg show up, and F when the arg is missing
    parser.add_argument('--HER', action="store_true", help="Use the HER algorithm")
    parser.add_argument('--goals', type=int, default=4, help="Number of goals for the HER algorithm")

    # Dynamic Hindsight Experience Replay (DHER) arguments
    # the para and flag for DHER
    parser.add_argument('--DHER', action="store_true", help="Use the HER algorithm")
    parser.add_argument('--dynamic', action="store_true", help="Use the dynamic mode for the bit flip environment")

    args = parser.parse_args()

    start_time = time.time()
    # go to the training pipeline
    train(args)
    # get the whole training time
    print('Run finished successfully in %s seconds' % round(time.time() - start_time))
