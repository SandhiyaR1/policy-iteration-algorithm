# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.
## PROBLEM STATEMENT
The slippery walk problem in reinforcement learning involves an agent navigating a 7-state environment to reach a target state. Due to the environment’s unpredictable nature, the agent might end up moving in the opposite direction of its intended move, adding an extra layer of complexity to the task
## POLICY ITERATION ALGORITHM
Initialize Policy:

Start with an arbitrary policy. In the provided code, the initial policy pi is randomly generated, where each state is assigned a random action. Policy Evaluation:

Evaluate the current policy by calculating the value function 𝑉 V for each state. This involves determining how good it is to be in each state under the current policy. This is done using the policy_evaluation function, which computes the value function 𝑉 V by solving a system of linear equations or using iterative methods until convergence (based on a threshold theta). Policy Improvement:

Improve the policy based on the value function 𝑉 V obtained from the evaluation step. This involves calculating the action-value function 𝑄 Q for each state-action pair and updating the policy to choose the action that maximizes the 𝑄 Q value. This is handled by the policy_improvement function, which returns a new policy pi where each state is assigned the action that has the highest value according to 𝑄 Q. Check for Convergence:

Compare the newly improved policy with the old policy. If the policy does not change (i.e., it is stable), then the algorithm has converged, and the process can stop. If the policy has changed, repeat the policy evaluation and policy improvement steps. Return Results:

Once the policy has converged, return the final value function 𝑉 V and the optimal policy pi.</br>
</br>

## POLICY IMPROVEMENT FUNCTION
### Name: Sandhiya R
### Register Number:212222230129
```python
def policy_improvement(V, P, gamma=1.0):
  Q=np.zeros((len(P),len(P[0])))
  for s in range(len(P)):
      for a in range(len(P[s])):
          for prob, next_state, reward, done in P[s][a]:
              Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
  new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
  return new_pi

```
## POLICY ITERATION FUNCTION
### Name:Sandhiya R
### Register Number:212222230129
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
    pi = lambda s: np.random.choice(list(P[s].keys()))  # Initialize with a random policy
    while True:
        V = policy_evaluation(pi, P, gamma, theta)  # Evaluate policy
        new_pi = policy_improvement(V, P, gamma)   # Improve policy
        if all(pi(s) == new_pi(s) for s in range(len(P))):  # Check convergence
            break
        pi = new_pi  # Update policy
    return V, pi

```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
![image](https://github.com/user-attachments/assets/69a5f6bc-67af-4b20-8a74-da551193fbf6)


### 2. Policy, Value function and success rate for the Improved Policy
![image](https://github.com/user-attachments/assets/82aea5d7-1236-4d60-a67d-79f49124870e)


### 3. Policy, Value function and success rate after policy iteration
![image](https://github.com/user-attachments/assets/4d0ba6f5-b81f-427b-aefd-d70a583b5e0e)


## RESULT:

Thus, a Python program is developed to find the optimal policy for the given MDP using the policy iteration algorithm.
