---
layout: single

title:  "강화학습 알고리즘: REINFORCE"

categories: RL_Algorithm

tag: [Reinforcement Learning, Policy Gradient, REINFORCE]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
---



**글에 들어가기 앞서...**

이 포스팅은 '**강화학습 알고리즘**'에 대한 내용을 담고 있습니다.



자료 출처: 단단한 강화학습, Reinforcement Learning An Introduction , 2nd edition. 리처드 서튼, 앤드류 바트로, 김성우(옮긴이), <https://velog.io/@koyeongmin/REINFORCE-알고리즘>









# REINFORCE

REINFORCE 알고리즘은 정책 기반 강화학습 알고리즘입니다. 오랜 시간동안 강화학습에서는 가치 함수를 기반으로 정책을 선택하는 방식을 채택했습니다. 그러나 이 방식으로는 연속되는 상태공간을 가지는 문제를 풀기가 굉장히 어려웠습니다. 이 문제점의 해결책으로 가치함수를 파라미터화하는 아이디어가 제시되었는데요, REINFORCE 알고리즘에서는 가치함수를 파라미터화하는 것이 아니라 정책을 바로 내뱉는 함수를 파라미터로 표현합니다. 


$$
\pi_{\theta}(s) = a
$$


이렇게 가치 함수를 기반으로 정책을 결정하는 것이 아니라, 정책 함수 자체를 업데이트하는 방식을 **정책 기반 강화학습**이라고 부릅니다. 그리고 **정책 기반 강화학습 중에서도 몬테카를로 업데이트를 수행하는 알고리즘을 REINFORCE 알고리즘이라고 합니다.** 수식들이 꽤나 복잡한데, 자세히 살펴보겠습니다.







## 행동 선택

REINFORCE 에서 행동의 선택은 파라미터화된 함수의 결과입니다. 그리고 궤적은 그러한 행동과 상태들의 나열이므로, 궤적은 파라미터화된 행동 선택 함수에 따르게 됩니다.


$$
\pi_{\theta}(s) = a
$$

$$
\tau \sim \pi_\theta
$$

 





## 목적 함수

당연히 파라미터는 처음에 의미 없는 값으로 설정되기 때문에, 행동 선택 함수가 선택하는 행동은 정말 별로인 행동들일 것입니다. 이 파라미터를 적절한 값으로 업데이트하기 위해서는 우리가 원하는 목적을 수식으로 분명하게 명세해야 합니다.


$$
J(\pi_\theta) = E_{\tau \sim \pi_\theta}[R(\tau)] = E_{\tau \sim \pi_\theta}[\sum^T_{t=0}\gamma^tr_t]
$$


위 수식에서 $J(\pi_\theta)$가 행동 선택 함수의 목적 함수입니다. 목적 함수는 $E_{\tau \sim \pi_\theta}[R(\tau)]$로 정의되었는데요, 말로 풀어서 설명하면 행동 선택 함수를 통해 만들어진 궤적들이 가지는 각각의 보상의 합, 이득들의 평균값입니다. 그리고 이 목적 함수를 사용해서 알고리즘의 목표를 표현할 수 있습니다.


$$
\max_\theta J(\pi_\theta) = E_{\tau \sim \pi_\theta}[R(\tau)]
$$


이로써 알고리즘의 목표, 파라미터가 향해야 할 목표가 표현되었습니다. $\max_\theta J(\pi_\theta)$에 가까워지도록 파라미터가 바뀌어야 하는데, 이는 미분을 사용해서 쉽게 업데이트할 수 있습니다.


$$
\theta \leftarrow \theta + \alpha\nabla_\theta J(\pi_\theta)
$$

$$
\theta \leftarrow \theta + \alpha\nabla_\theta E_{\tau \sim \pi_\theta}[R(\tau)]
$$


그런데 위 수식에는 문제점이 있는데요, 그것은 $\nabla_\theta E_{\tau \sim \pi_\theta}[R(\tau)]$을 구하는게 어렵다는 점입니다. $\theta$가 분포 안에 숨어있는 형태로 있기 때문에 미분을 하기 위해서는 분포를 표면으로 끌어내주어야 합니다.


$$
\nabla_\theta E_{\tau \sim \pi_\theta}[R(\tau)] = \nabla_\theta\int dx \space R(\tau)p(\tau|\theta)
$$


기댓값은 적분으로 정의됩니다. 기댓값을 풀어서 조건부에 파라미터가 드러나게 합니다.


$$
= \int dx \space R(\tau) \nabla_\theta p(\tau|\theta)
$$

 

$dx$와 $R(\tau)$는 $\theta$와 관련이 없기 때문에, 이 둘을 넘어서 $\nabla_\theta$를 위치시킵니다.

 

$$
= \int dx \space R(\tau) \nabla_\theta p(\tau|\theta) \times \frac{p(x|\theta)}{p(x|\theta)}
$$

$$
= \int dx \space R(\tau)  p(\tau|\theta) \times \frac{\nabla_\theta p(\tau|\theta)}{p(x|\theta)}
$$

$$
= \int dx \space R(\tau)  p(\tau|\theta) \times \nabla_\theta \log p(\tau|\theta)
$$

$$
= E_{\tau \sim \pi_\theta}[R(\tau)\nabla_\theta \log p(\tau|\theta)]
$$



위와 같이 로그 미분의 형태로 바꿔줬는데요, 궤적의 확률값은 마르코프 상황에서 아래와 같이 쉽게 구해집니다.



$$
p(\tau|\theta) = \prod_{t\geq0}p(s_{t+1}|s_t, a_t)\pi_\theta(a_t|s_t)
$$

$$
\log p(\tau|\theta) = \log\prod_{t\geq0}p(s_{t+1}|s_t, a_t)\pi_\theta(a_t|s_t)
$$

$$
\sum_{t\geq0} [\log p(s_{t+1}|s_t, a_t) + \log \pi_\theta(a_t|s_t)]
$$

 

위 식에 $\nabla_\theta$을 취하게 되면, 오른쪽 로그 식은 $\theta$와 관련이 없기 때문에 사라지게 됩니다.



$$
\nabla_\theta \log p(\tau|\theta) = \nabla_\theta\sum_{t\geq0} \log \pi_\theta(a_t|s_t)
$$



이제 전체적으로 다시 쓰면 아래와 같습니다.


$$
\nabla_\theta J(\pi_\theta) = E_{\tau \sim \pi_\theta}[\sum_{t\geq0}^T R(\tau) \nabla_\theta \log \pi_\theta(a_t|s_t)]
$$



그런데요 위 식을 그대로 쓰게되면 분산이 굉장히 큽니다. 왜냐하면 전체 궤적의 이득 값을 사용하기 때문인데요, $t$ 시점 이전의 보상들은 결과값에 영향을 미치지 못하기 때문에 아래와 같이 $t$ 시점 이후의 보상들의 합으로 바꿔써도 괜찮습니다.



$$
= E_{\tau \sim \pi_\theta}[\sum_{t\geq0}^T R_t(\tau) \nabla_\theta \log \pi_\theta(a_t|s_t)]
$$


한마디로 $R(\tau)$를 $R_t(\tau)$로 바꿔적는건데요, 이게 괜찮은 이유는 아래에서 설명합니다.


$$
\sum_0^{t-1} \gamma^{t'} r_{t'} + R_t(\tau)
$$


$R(\tau)$와 $R_t(\tau)$의 관계는 위와 같습니다. $R_t(\tau)$는 $t$ 시점의 행동 선택에 따라 달라지는 반면, $\sum_0^{t-1} \gamma^{t'} r_{t'}$의 값은 영향을 받지 않습니다. 


$$
\nabla_\theta J(\pi_\theta) = E_{\tau \sim \pi_\theta}[\sum_{t\geq0}^T R(\tau) \nabla_\theta \log \pi_\theta(a_t|s_t)]
$$

$$
= E_{\tau \sim \pi_\theta}[\sum_{t\geq0}^T (R_t(\tau) +  \sum_0^{t-1} \gamma^{t'} r_{t'}) \nabla_\theta \log \pi_\theta(a_t|s_t)]
$$



두 식이 동일하기 위해서는 아래 식의 두 번째 항의 값이 0이 되어야 합니다.


$$
= E_{\tau \sim \pi_\theta}[\sum_0^{t-1} \gamma^{t'} r_{t'} \nabla_\theta \log \pi_\theta(a_t|s_t)]
$$

그런데 앞쪽 궤적의 보상의 합 부분은 상수이기 때문에, 뒤쪽 식만 0인지 확인하면 됩니다.



$$
E_{\tau \sim \pi_\theta}[ \nabla_\theta \log \pi_\theta(a_t|s_t)]
$$

$$
= \int \pi_\theta(a_t|s_t) \nabla_\theta \log \pi_\theta(a_t|s_t) d\tau
$$

$$
= \int \nabla_\theta \pi_\theta(a_t|s_t) d\tau
$$

$$
= \nabla_\theta \int  \pi_\theta(a_t|s_t) d\tau
$$

$$
= \nabla_\theta[1]
$$

$$
= 0
$$


때문에 $t$ 시점 이전의 값을 빼어주어도 결과 값에 아무런 값의 차이가 없습니다. 동일한 맥락으로 $t$시점의 행동에 영향을 받지 않는 값을 빼어주어도 괜찮습니다.


$$
\nabla_\theta J(\pi_\theta) = E_{\tau \sim \pi_\theta}[\sum_{t\geq0}^T (R_t(\tau) -b(s_t))\nabla_\theta \log \pi_\theta(a_t|s_t)]
$$


보통 위와 같이 현재 시점의 상태와 관련된 함수값을 빼주는데요, 왜냐하면 지금 내가 행하는 행동이 있는데 행동을 하는 상태까지는 이미 결정되어 있기 때문입니다. 







## 알고리즘

<img src="/images/2025-01-15-Reinforcement_Algorithm_REINFORCE/image-20250115164312507.png" alt="image-20250115164312507" style="zoom:50%;" />

구현된 알고리즘을 보면 위에서 계속 전개해 나갔던 수식이랑 모습이 사뭇 다른데요, 한 번에 업데이트할 지, 조금씩 업데이트할 지의 차이일 뿐 동일합니다.







## 파이썬 코드

아래는 파이썬에서 구현한 REINFORCE 알고리즘입니다. 

```python
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import time  # 렌더링 시 잠깐씩 멈출 때 사용

writer = SummaryWriter(log_dir="logs/REINFORCE")

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20
    
    
    for n_epi in range(3000):
        s, _ = env.reset()
        done = False
        
        while not done: # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, truncated, info = env.step(a.item())
            pi.put_data((r,prob[a]))
            s = s_prime
            score += r
            
        pi.train_net()
        
        if n_epi%print_interval==0 and n_epi!=0:
            writer.add_scalar('score', score/print_interval, n_epi)
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0
    # ---------- 학습이 끝난 뒤, 렌더링 테스트 ----------
    print("Training completed. Now testing (rendering) the final policy...")
    # CartPole 환경 재생성을 권장 (render_mode='human' 가능, 하지만 Gym 버전에 따라 다름)

    writer.close()  # 텐서보드 끄기
    env.close()  # 기존 env 닫고
    env = gym.make('CartPole-v1', render_mode='human')  # 다시 생성
    for test_ep in range(5):  # 5 에피소드 정도 시각화
        s, _ = env.reset()
        done = False
        ep_score = 0.0
        while not done:
            env.render()
            time.sleep(0.01)  # 속도를 너무 빠르게 하고 싶지 않다면 잠깐 딜레이
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()
            s, r, done, truncated, info = env.step(a)
            ep_score += r
        print(f"Test Episode {test_ep+1} Score: {ep_score:.1f}")
    env.close()

if __name__ == '__main__':
    main()
```

코드가 꽤나 긴데, REINFORCE 알고리즘의 핵심이 되는`train`부분을 집중적으로 살펴보겠습니다.



```python
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

```

코드를 보면, 하나의 에피소드가 진행되는 동안에는 학습이 수행되지 않고, 에이전트가 받은 보상과 에이전트가 선택한 행동의 정책 확률을 저장합니다. 그리고 REINFORCE 알고리즘의 아래 수식에 해당하는 값을 반복 계산합니다.


$$
R(\tau) \nabla_\theta \log \pi_\theta(a_t|s_t)
$$


모든 궤적에 대해서 기울기를 구했으면 마지막에 한 번에 업데이트를 수행합니다. 위에서 설명했던 수식과 다르게 코드에서는 `loss = -torch.log(prob) * R` 로 음수 기호가 붙는데요, 이는 경사하강이 아닌 경사상승을 해야하기 때문입니다. Pytorch를 포함하는 대부분의 라이브러리는 기본적으로 경사하강을 하는 상황을 가정하기 때문에 음수 기호를 붙혀줘 보상을 최대화하도록 업데이트합니다.



<img src="/images/2025-01-15-Reinforcement_Algorithm_REINFORCE/image-20250116102640302.png" alt="image-20250116102640302" style="zoom:40%;" />

실제로 학습을 시켜보면 꾸준히 성능이 오르는 것을 확인할 수 있습니다. 



![cartpole_reinforce 2](/images/2025-01-15-Reinforcement_Algorithm_REINFORCE/cartpole_reinforce 2.gif)

학습된 모델로 환경에서 렌더링한 결과인데요, 적당히 잘 중심을 맞추는 것도 확인할 수 있습니다.
