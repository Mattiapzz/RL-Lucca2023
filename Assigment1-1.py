import numpy as np


GAMMA = (0.5,0.85,0.9,0.99,1-1e-5,1)


P1 = np.zeros((20,20))
for i in range(0,19):
    if i < 20:
        P1[i,i+1] = 1
    if i == 20:
        P1[i,1] = 1

P2 = np.zeros((20,20))
for i in range(0,19):
    if i < 20:
        P2[i,i+1] = 1
    if i == 5 or i == 10 or i == 15:
        P2[i,1] = 1

Vs = np.zeros((20,1))







def pi1(s):
    if s == 5 or s == 10 or s == 15 or s == 20:
        return 3
    else:
        return 2
    
def pi2(s):
    if s == 20:
        return 3
    else:
        return 2
    
def reward(s,a):
    if a == 3:
        if s == 5:
            return 5
        elif s == 10:
            return 20
        elif s == 15:
            return 45
        elif s == 20:
            return 80
        # else:
        #     return 0
    else:
        return 0
    
def evolve_state(s,policy):
    a = policy(s)
    if a == 2:
        return s + 1
    elif a == 3:
        return 1
    elif a == 1:
        return s

    
def V_pi(s0,gamma,policy):
    sum = 0
    s = s0
    for k in range(1,1000):
        sum = sum + gamma**k * reward(s,policy(s))
        s = evolve_state(s,policy)
    return sum

def main():
    for gamma in GAMMA:
        print("gamma = ",gamma)
        print("V_pi1 = ",V_pi(1,gamma,pi1))
        print("V_pi2 = ",V_pi(1,gamma,pi2))
        print("")
        for i in range(1,20):
            Vs[i-1] = reward(i,pi1(i))
        print("V_pi1 (Bellman) = ", np.matmul( np.linalg.inv( np.eye(20)-gamma*P1 ) , Vs ) )
    # print(
    # 
    # _pi2 = ",V_pi(1,0.5,pi2))

if __name__ == "__main__":
    main()
