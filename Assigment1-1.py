'''
Assignment 1-1

Authors
-------
 - Mattia Piazza,
 - Sebastiano Taddei.

'''
import math
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

def action_set(state):
    '''

    Action set.

    '''
    actions = [1]
    if state < 20:
        actions.append(2)
    if state == 5 or state == 10 or state == 15 or state == 20:
        actions.append(3)

    return actions

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
    
    

def evolve_state(s, a):
    '''

    Evolve state according to policy.

    '''

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
        state = evolve_state(state, policy(state))
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
        state  = evolve_state(state, policy(state))
        action = policy(state)
    return sum


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
            V_plus[i] = reward(i+1,policy(i+1)) + gamma * V[evolve_state(i+1,policy(i+1))-1] 
        V_hyst.append(np.copy(V_plus))

        if np.linalg.norm(V_plus - V) < epsilon:
            break
        
    V = V_plus
    
    return V, V_hyst

def VI_hyst(gamma,reward,epsilon, max_iter):
    '''

    Value iteration.

    '''
    V_hyst = []
    V = np.zeros(20)
    V_plus = np.zeros(20)
    for _ in range(max_iter):
    
        V[:] = V_plus[:]
        for i in range(20):
            V_plus[i] = np.max([ reward(i+1,action) + gamma * V[evolve_state(i+1,action)-1] for action in action_set(i+1) ])
        V_hyst.append(np.copy(V_plus))

        if np.linalg.norm(V_plus - V) < epsilon:
            break
        
    V = V_plus
    
    return V, V_hyst

def PI_hyst(gamma,reward,epsilon, max_iter):
    '''

    Policy iteration.

    '''
    V_hyst = []
    V = np.zeros(20)
    V_plus = np.zeros(20)
    pi_s = np.zeros(20)
    for _ in range(max_iter):
    
        V[:] = V_plus[:]
        for i in range(20):
            index = np.argmax([ reward(i+1,action) + gamma * V[evolve_state(i+1,action)-1] for action in action_set(i+1) ])
            pi_s[i] = action_set(i+1)[index]
        for i in range(20):
            V_plus[i] = reward(i+1,pi_s[i]) + gamma * V[evolve_state(i+1,pi_s[i])-1] 
        V_hyst.append(np.copy(V_plus))

        if np.linalg.norm(V_plus - V) < epsilon:
            break
        
    V = V_plus
    
    return V, V_hyst


def exercise_a(gammas):
    '''
    
    exercise a.

    '''
    print("-"*NUM_DASHES)
    print("exercise a")
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


def exercise_b(action, gammas):
    '''
    
    exercise b.

    '''
    print("-"*NUM_DASHES)
    print("exercise b")
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

def exercise_c(gammas):
    '''
    
    exercise c.

    '''
    print("-"*NUM_DASHES)
    print("exercise c")
    for gamma in gammas:
        print(f"gamma = {gamma}")
        reward_P1 = [reward(i, pi1(i)) for i in range(1, 21)]
        reward_P2 = [reward(i, pi2(i)) for i in range(1, 21)]

        P1 = state_transition_matrix_P1()
        P2 = state_transition_matrix_P2()

        V_bell_P1 = np.linalg.inv( np.eye(20)-gamma*P1 ) @ reward_P1
        V_bell_P2 = np.linalg.inv( np.eye(20)-gamma*P2 ) @ reward_P2

        V, V_hyst = PE_hyst(gamma, pi1, reward, V_pi, 1e-8, 50)

        err_hyst = [np.linalg.norm(V_tmp - V_bell_P1) for V_tmp in V_hyst]

        plt.figure()
        plt.title("Error PE policy $\pi_1$")
        plt.plot(err_hyst)
        plt.show()

        plt.figure()
        data = np.array(V_hyst).T
        plt.imshow(data, interpolation="None")
        plt.title("PE policy $\pi_1$")
        plt.xlabel("iteration")
        plt.ylabel("state")
        plt.xlim(0, len(V_hyst))
        plt.ylim(0, 20)
        plt.colorbar()
        plt.show()

        V, V_hyst = PE_hyst(gamma, pi2, reward, V_pi, 1e-8, 50)

        err_hyst = [np.linalg.norm(V_tmp - V_bell_P2) for V_tmp in V_hyst]

        plt.figure()
        plt.title("Error PE policy $\pi_2$")
        plt.plot(err_hyst)
        plt.show()

        plt.figure()
        data = np.array(V_hyst).T
        plt.imshow(data, interpolation="None")
        plt.title("PE policy $\pi_2$")
        plt.xlabel("iteration")
        plt.ylabel("state")
        plt.xlim(0, len(V_hyst))
        plt.ylim(0, 20)
        plt.colorbar()
        plt.show()



def exercise_d(gammas):
    '''
    
    exercise d.

    '''
    print("-"*NUM_DASHES)
    print("exercise d")
    for gamma in gammas:
        print(f"gamma = {gamma}")

        V, V_hyst = VI_hyst(gamma, reward, 1e-8, 50)
        plt.figure()
        data = np.array(V_hyst).T
        plt.imshow(data, interpolation="None")
        plt.title(f"Value Iteration $\gamma = {gamma}$ ")
        plt.xlabel("iteration")
        plt.ylabel("state")
        plt.xlim(0, len(V_hyst))
        plt.ylim(0, 20)
        plt.colorbar()
        plt.show()

    
    print("-"*NUM_DASHES)

def exercise_e(gammas):
    '''
    
    exercise e.

    '''
    print("-"*NUM_DASHES)
    print("exercise e")
    for gamma in gammas:
        print(f"gamma = {gamma}")

        V, V_hyst = PI_hyst(gamma, reward, 1e-8, 50)
        plt.figure()
        data = np.array(V_hyst).T
        plt.imshow(data, interpolation="None")
        plt.title(f"Policy Iteration $\gamma = {gamma}$ ")
        plt.xlabel("iteration")
        plt.ylabel("state")
        plt.xlim(0, len(V_hyst))
        plt.ylim(0, 20)
        plt.colorbar()
        plt.show()

    
    print("-"*NUM_DASHES)


            







def main():
    gammas = (0.5, 0.85, 0.9, 0.99, 1-1e-5) #, 1 -> omitted for singularity
    # exercise_a(gammas)
    # exercise_b(2,gammas)
    # exercise_c(gammas)
    exercise_d(gammas)
    exercise_e(gammas)

    
if __name__ == "__main__":
    main()
