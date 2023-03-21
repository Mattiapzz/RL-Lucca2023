'''
Assignment 1

Authors
-------
 - Mattia Piazza,
 - Sebastiano Taddei.

'''

import numpy as np


def state_transition_matrix_P1():
    '''
    
    State transition matrix for policy 1.
    
    '''
    
    P1 = np.zeros((20,20))
    for i in range(19):
        if i%5 == 4:
            P1[i,0] = 1
        else:
            P1[i,i+1] = 1

    P1[19,0] = 1

    return P1

def state_transition_matrix_P2():
    '''
    
    State transition matrix for policy 2.
    
    '''
    
    P2 = np.eye(20, k=1)
    P2[19,0] = 1

    return P2

def pi1(s):
    '''

    Policy 1.

    '''

    if s == 5 or s == 10 or s == 15 or s == 20:
        return 3
    else:
        return 2
    
def pi2(s):
    '''

    Policy 2.

    '''

    if s == 20:
        return 3
    else:
        return 2
    
def reward(s, a):
    '''

    Reward function.

    '''

    if a == 3:
        if s == 5:
            return 5
        elif s == 10:
            return 20
        elif s == 15:
            return 45
        elif s == 20:
            return 80
    else:
        return 0
    
def evolve_state(s, policy):
    '''

    Evolve state according to policy.

    '''

    a = policy(s)
    if a == 2:
        return s + 1
    elif a == 3:
        return 1
    elif a == 1:
        return s

    
def V_pi(state_0, gamma, policy):
    '''

    Compute the value function of a policy.

    '''

    sum   = 0
    state = state_0
    for k in range(10000):
        sum   = sum + gamma**k * reward(state, policy(state))
        state = evolve_state(state, policy)

    return sum

def main():
    gammas = (0.5, 0.85, 0.9, 0.99, 1-1e-5) #, 1 -> omitted for singularity

    for gamma in gammas:
        print("-"*20)
        print(f"gamma = {gamma}")
        print("-"*20)
        print(f"V_pi1 = {V_pi(1, gamma, pi1)}")
        print(f"V_pi2 = {V_pi(1, gamma, pi2)}")
        print("-"*20)

        reward_P1 = [reward(i, pi1(i)) for i in range(1, 21)]
        reward_P2 = [reward(i, pi2(i)) for i in range(1, 21)]

        P1 = state_transition_matrix_P1()
        P2 = state_transition_matrix_P2()

        V_bell_P1 = np.linalg.inv( np.eye(20)-gamma*P1 ) @ reward_P1
        V_bell_P2 = np.linalg.inv( np.eye(20)-gamma*P2 ) @ reward_P2
        print(f"V_pi1 (Bellman) = {V_bell_P1[0]}")
        print(f"V_pi2 (Bellman) = {V_bell_P2[0]}")
        print("-"*20)

if __name__ == "__main__":
    main()
