import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0

        for state in range(env.nS):
            v = V[state]

            for action, action_p in enumerate(policy[state]):
                temp = 0
                
                for  prob, next_state, reward, _ in env.P[state][action]:
                    temp += prob * (reward + discount_factor * V[next_state])
                    
                v += action_p * temp
                
            V[state] = v        
            delta = max(delta, np.abs(v - V[state]))
            
        if delta < theta:
            break

    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    # YOUR CODE HERE
    while True:
        # Step 2
        V = policy_eval_v(policy, env, discount_factor)
        # Step 3
        policy_stable = True
        for state in range(env.nS):
            old_action = np.argmax(policy[state])

            action_probs = np.zeros(env.nA)
            for action in range(env.nA):
                for prob, next_state, reward, _ in env.P[state][action]:
                    action_probs[action] += prob * (reward + discount_factor * V[next_state])
            
            action = np.argmax(action_probs)
            policy[state] = np.identity(env.nA)[action]
            
            if old_action != action:
                policy_stable = False
            
        if policy_stable:
            break

    return policy, V
