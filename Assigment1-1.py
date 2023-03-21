'''
Assignment 1

Authors
-------
 - Mattia Piazza,
 - Sebastiano Taddei.

'''

import numpy as np
import matplotlib.pyplot as plt

MAX_ITER = 10000 # maximum number of iterations (best results with 1000000)
NUM_DASHES = 20


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
    for k in range(MAX_ITER):
        sum   = sum + gamma**k * reward(state, policy(state))
        state = evolve_state(state, policy)
    return sum

def Q_pi(state_0, action_0, gamma, policy):
    '''
    
    Compute the Q function of a policy.

    '''
    sum    = 0
    state  = state_0
    action = action_0
    for k in range(MAX_ITER):
        sum    = sum + gamma**k * reward(state, action)
        state  = evolve_state(state, policy)
        action = policy(state)
    return sum
    
def PE(gamma,policy,reward,value_function,epsilon):
    '''

    Policy evaluation.

    '''
    V_plus = np.zeros(20)
    V = np.array([value_function(i+1, gamma, policy) for i in range(20)])
    while np.linalg.norm(V_plus - V) > epsilon:
        V = V_plus
        for i in range(20):
            V_plus[i] = reward(i+1,policy(i+1)) + gamma * value_function(evolve_state(i+1,policy), gamma, policy)
    V = V_plus
    return V

def PE_hyst(gamma,policy,reward,value_function,epsilon, max_iter):
    '''

    Policy evaluation.

    '''
    V_hyst = []
    V = np.zeros(20)
    V_plus = np.zeros(20)
    for _ in range(max_iter):
        V[:] = V_plus[:]
        for i in range(20):
            V_plus[i] = reward(i+1,policy(i+1)) + gamma * V[evolve_state(i+1,policy)-1] #value_function(evolve_state(i+1,policy), gamma, policy)
        V_hyst.append(V_plus)

        if np.linalg.norm(V_plus - V) < epsilon:
            break
        
    V = V_plus
    
    return V, V_hyst


def exercise1(gammas):
    '''
    
    Exercise 1.

    '''
    print("-"*NUM_DASHES)
    print("Exercise 1")
    for gamma in gammas:
        print(f"gamma = {gamma}")

        reward_P1 = [reward(i, pi1(i)) for i in range(1, 21)]
        reward_P2 = [reward(i, pi2(i)) for i in range(1, 21)]

        P1 = state_transition_matrix_P1()
        P2 = state_transition_matrix_P2()

        V_bell_P1 = np.linalg.inv( np.eye(20)-gamma*P1 ) @ reward_P1
        V_bell_P2 = np.linalg.inv( np.eye(20)-gamma*P2 ) @ reward_P2

        # print a formatted table with value of V_pi1, V_pi2, V_pi1 (Bellman), V_pi2 (Bellman)
        print(f"{'State':^6} {'V_pi1':^15} {'V_pi1 (Bellman)':^15} {'V_pi2':^15} {'V_pi2 (Bellman)':^15}")
        for i in range(20):
            print(f"{i+1:>6} {V_pi(i+1, gamma, pi1):>15.4e} {V_bell_P1[i]:>15.4e} {V_pi(i+1, gamma, pi2):>15.4e} {V_bell_P2[i]:>15.4e}")
        
        print("-"*NUM_DASHES)


def exercise2(action, gammas):
    '''
    
    Exercise 2.

    '''
    print("-"*NUM_DASHES)
    print("Exercise 2")
    for gamma in gammas:
        print(f"gamma = {gamma}")

        reward_P1 = [reward(i, pi1(i)) for i in range(1, 21)]
        reward_P2 = [reward(i, pi2(i)) for i in range(1, 21)]

        P1 = state_transition_matrix_P1()
        P2 = state_transition_matrix_P2()

        V_bell_P1 = np.linalg.inv( np.eye(20)-gamma*P1 ) @ reward_P1
        V_bell_P2 = np.linalg.inv( np.eye(20)-gamma*P2 ) @ reward_P2

        # print a formatted table with value of V_pi1, V_pi2, V_pi1 (Bellman), V_pi2 (Bellman)
        print(f"{'State':^6} {'Q_pi1':^15} {'Q_pi2':^15} {'Err. Q_pi1':^15} {'Err. Q_pi2':^15} {'Err. Q_pi1 (Bellman)':^15} {'Err. Q_pi2 (Bellman)':^15}")
        for i in range(20):
            Q_pi1_tmp = Q_pi(i+1, action, gamma, pi1)
            Q_pi2_tmp = Q_pi(i+1, action, gamma, pi2)
            diff_Q_pi1 = Q_pi(i+1, pi1(i+1), gamma, pi1) - V_pi(i+1, gamma, pi1)
            diff_Q_pi2 = Q_pi(i+1, pi2(i+1), gamma, pi2) - V_pi(i+1, gamma, pi2)
            diff_Q_pi1_bell = Q_pi(i+1, pi1(i+1), gamma, pi1) - V_bell_P1[i]
            diff_Q_pi2_bell = Q_pi(i+1, pi2(i+1), gamma, pi2) - V_bell_P2[i]

            print(f"{i+1:>6} {Q_pi1_tmp:>15.4e} {Q_pi2_tmp:>15.4e} {diff_Q_pi1:>15.4e} {diff_Q_pi2:>15.4e} {diff_Q_pi1_bell:>15.4e} {diff_Q_pi2_bell:>15.4e}")

        
        print("-"*NUM_DASHES)

def exercise3(gammas):
    '''
    
    Exercise 3.

    '''
    print("-"*NUM_DASHES)
    print("Exercise 3")
    for gamma in gammas:
        print(f"gamma = {gamma}")
        reward_P1 = [reward(i, pi1(i)) for i in range(1, 21)]
        reward_P2 = [reward(i, pi2(i)) for i in range(1, 21)]

        P1 = state_transition_matrix_P1()
        P2 = state_transition_matrix_P2()

        V_bell_P1 = np.linalg.inv( np.eye(20)-gamma*P1 ) @ reward_P1
        V_bell_P2 = np.linalg.inv( np.eye(20)-gamma*P2 ) @ reward_P2

        V, V_hyst = PE_hyst(gamma, pi1, reward, V_pi, 1e-8, 20)

        err_hyst = [np.linalg.norm(V_tmp - V_bell_P1) for V_tmp in V_hyst]

        plt.figure()
        plt.plot(err_hyst)
        plt.show()
            







def main():
    gammas = (0.5, 0.85, 0.9, 0.99, 1-1e-5) #, 1 -> omitted for singularity
    # exercise1(gammas)
    # exercise2(2,gammas)
    exercise3(gammas)

    
if __name__ == "__main__":
    main()
