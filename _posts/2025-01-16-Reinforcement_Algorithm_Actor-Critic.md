---
layout: single

title:  "강화학습 알고리즘: Actor Critic"

categories: RL_Algorithm

tag: [Reinforcement Learning, Policy Gradient, Actor Critic]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
published: true
---



**글에 들어가기 앞서...**

이 포스팅은 '**강화학습 알고리즘**'에 대한 내용을 담고 있습니다.



자료 출처: 단단한 강화학습, Reinforcement Learning An Introduction , 2nd edition. 리처드 서튼, 앤드류 바트로, 김성우(옮긴이)>, <https://github.com/seungeunrho/minimalRL>









# Actor-Critic

몬테카를로 정책 경사(**REINFORCE**) 알고리즘에서 이득의 분산을 줄이기 위한 여러 기술적인 요소들이 있었음에도, 여전히 꽤나 높은 분산을 가집니다. 몬테카를로 방식을 사용하는 알고리즘들은 다 분산이 큽니다. 분산이 작은 알고리즘으로는 TD 방식이 있었습니다. 그리고 이제 정책 경사 알고리즘에 TD 방식을 사용합니다.







## Baseline Policy Gradient

$$
\nabla J(\theta) \propto E_\pi[ \sum_a  (q_\pi (S_t, a) - b(s))\nabla \log \pi(a|S_t, \theta)]
$$



위는 정책 경사 에서 기준선을 사용하는 목적함수의 경사도 식입니다. $b(s)$는 $a$에 영향을 받지 않는 어떤 함수도 가능합니다. 다만 갱신 기댓값이 갖는 분산을 낮춰주기 위해서는 최대한 $q_\pi (S_t, a)$에 가까운 값을 찾아주는게 좋은데요, 이런 요구에 가장 잘 맞는 함수가 바로 상태 함수 $v(S_t)$입니다.


$$
\nabla J(\theta) \propto E_\pi[ \sum_a  (q_\pi (S_t, a) - V(S_t))\nabla \log \pi(a|S_t, \theta)]
$$


때문에 기준선을 사용하는 정책 경사 알고리즘은 위의 식을 사용합니다. REINFORCE에서는 $q_\pi (S_t, a)$ 대신에 $G_t$를 사용했는데요, **행동자 비평가 알고리즘에서는 TD 목표(TD-Target)를 사용합니다.** 더불어 상태를 평가할 때 근사 함수($V \rightarrow \hat v$)를 사용합니다.


$$
\nabla J(\theta) \propto E_\pi[ \sum_a  (R_{t+1} + \gamma \hat v(S_{t+1}, w) - \hat v(S_t, w))\nabla \log \pi(a|S_t, \theta)]
$$

$$
\theta_{t+1} \doteq \theta_{t} + \alpha \delta_t\nabla \log \pi(a|S_t, \theta)
$$

- 위 수식은 행동 비평가 알고리즘 중에서도 **A2C**에 해당하는 알고리즘입니다. 가장 기본형의 Actor Critic 알고리즘에서는 $(q_\pi (S_t, a) - V(S_t))$가 아닌 $q_\pi (S_t, a)$를 사용합니다. baseline이 없기 때문에 갱신 기댓값이 갖는 분산이 크다는 단점이 있습니다. 







## 알고리즘

<img src="/images/2025-01-16-Reinforcement_Algorithm_Actor-Critic/image-20250116172001734.png" alt="image-20250116172001734" style="zoom:40%;" />

$w$ 파라미터 식이 조금 뜬금없는데요, $w$는 비평가의 파라미터인데 TD-error를 통한 업데이트가 수행된다고 생각하면 됩니다. 대부분의 경우에 MSE Error로 구현됩니다.







## 파이썬 코드

아래는 파이썬에서 구현한 Actor-Critic 알고리즘입니다. 

```python
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import time  # 렌더링 시 잠깐씩 멈출 때 사용

writer = SummaryWriter(log_dir="logs/actor_critic")

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98
n_rollout     = 10

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
  
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()         
      
def main():  
    env = gym.make('CartPole-v1')
    model = ActorCritic()    
    print_interval = 20
    score = 0.0

    for n_epi in range(3000):
        done = False
        s, _ = env.reset()
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)
                model.put_data((s,a,r,s_prime,done))
                
                s = s_prime
                score += r
                
                if done:
                    break                     
            
            model.train_net()
            
        if n_epi%print_interval==0 and n_epi!=0:
            writer.add_scalar('score', score/print_interval, n_epi)
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    writer.close()
    env.close()

    # ---------- 학습이 끝난 뒤, 렌더링 테스트 ----------
    print("Training completed. Now testing (rendering) the final policy...")
    env = gym.make('CartPole-v1', render_mode='human')  # 다시 생성
    for test_ep in range(5):  # 5 에피소드 정도 시각화
        s, _ = env.reset()
        done = False
        ep_score = 0.0
        while not done:
            env.render()
            time.sleep(0.01)
            prob = model.pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()
            s, r, done, truncated, info = env.step(a)
            ep_score += r
        print("Test Episode #{} Score: {}".format(test_ep, ep_score))
    env.close()

if __name__ == '__main__':
    main()
```

역시나 핵심 부분은 `train_net`입니다. 자세히 살펴보겠습니다.



```python
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()    
```

`loss`가 우리가 눈여겨 봐야 할 부분입니다. $\pi$를 업데이트 하기 위한 `-torch.log(pi_a) * delta.detach()`부분과, $w$를 업데이트 하기 위한 `F.smooth_l1_loss(self.v(s), td_target.detach())` 를 확인할 수 있습니다. 

- `F.smooth_l1_loss(self.v(s), td_target.detach())` 코드가 조금 헷갈리는데요, 단순히 `self.v(s)`와 `td_target.detach()`의 차이를 `F.smooth_l1_loss`로 측정했다고 생각하면 됩니다.



Pytorch에서 자동으로 기울기를 계산해주기 때문에 코드에는 업데이트 식이 없구요, REINFORCE 알고리즘과 마찬가지로(여타 다른 모든 강화학습 알고리즘과 마찬가지로) 경사 상승을 해야하기 때문에 우리가 위에서 구했던 수식에 음수가 붙어있는 것을 확인할 수 있습니다.



```python
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
```

행동자와 비평가가 동일한 `self.fc1(x)`레이어를 공유하는 것을 확인할 수 있는데요, 사용하는 경우도 있고 사용하지 않는 경우도 쓰입니다. 

- 네트워크를 공유하는 경우 파라미터 수가 줄어들기 때문에 학습 효율성이 늘고 계산량이 감소합니다.
- 네트워크를 공유하지 않는 경우 학습 간섭이 없기 때문에 학습 불안정성이 전파되지 않고 하이퍼파라미터를 조정하기 용이합니다. 





### 학습 결과

![image-20250116173516734](/images/2025-01-16-Reinforcement_Algorithm_Actor-Critic/image-20250116173516734.png)

REINFORCE와 비교할 때 꽤나 불안정한 모습을 보이는데요, 학습 시간이 부족한 탓일 수도 있고, 환경 자체의 단순함 때문일 수도 있겠습니다. 아래는 테스트 렌더링 결과입니다.



![AC_cartpole](/images/2025-01-16-Reinforcement_Algorithm_Actor-Critic/AC_cartpole.gif)





# SAC(Soft Actor Critic)

SAC는 Soft Q-learning을 사용하는 Actor Critic 알고리즘을 의미합니다. 







## Soft Q-Learning

Soft Q-Learning에서는 max함수가 아닌 Smooth Maximum Function을 사용합니다.


$$
LSE(z) := \beta \ln(\exp(z_1/\beta) + \exp(z_2/\beta)+\cdots+\exp(z_n/\beta))
$$

위 식은 max함수와 비교했을 때, 좀 더 부드러운 최댓값을 반환합니다. $\beta$가 무한대로 갈때 LSE는 max함수와 동일합니다. 따라서 $\beta$를 통해 부드러운 정도를 조절할 수 있습니다.







## 파이썬 코드

```python
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random

#Hyperparameters
lr_pi           = 0.0005
lr_q            = 0.001
init_alpha      = 0.01
gamma           = 0.98
batch_size      = 32
buffer_limit    = 50000
tau             = 0.01 # for target network soft update
target_entropy  = -1.0 # for automated alpha update
lr_alpha        = 0.001  # for automated alpha update

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class PolicyNet(nn.Module):
    def __init__(self, learning_rate):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc_mu = nn.Linear(128,1)
        self.fc_std  = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

class QNet(nn.Module):
    def __init__(self, learning_rate):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_cat = nn.Linear(128,32)
        self.fc_out = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def calc_target(pi, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob= pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target
    
def main():
    env = gym.make('Pendulum-v1')
    memory = ReplayBuffer()
    q1, q2, q1_target, q2_target = QNet(lr_q), QNet(lr_q), QNet(lr_q), QNet(lr_q)
    pi = PolicyNet(lr_pi)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        count = 0

        while count < 200 and not done:
            a, log_prob= pi(torch.from_numpy(s).float())
            s_prime, r, done, truncated, info = env.step([2.0*a.item()])
            memory.put((s, a.item(), r/10.0, s_prime, done))
            score +=r
            s = s_prime
            count += 1
                
        if memory.size()>1000:
            for i in range(20):
                mini_batch = memory.sample(batch_size)
                td_target = calc_target(pi, q1_target, q2_target, mini_batch)
                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)
                entropy = pi.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)
                
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(n_epi, score/print_interval, pi.log_alpha.exp()))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()
```











