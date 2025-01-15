---
layout: single

title:  "텐서보드"

categories: Tools

tag: [log]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
---



**글에 들어가기 앞서...**

이 포스팅은 '**텐서보드**'에 대한 내용을 담고 있습니다.



자료 출처: <https://tutorials.pytorch.kr/recipes/recipes/tensorboard_with_pytorch.html>, <https://github.com/seungeunrho/minimalRL>









# 텐서보드

텐서보드는 다양한 머신러닝 실험 결과들의 시각화를 제공해주는 편리한 툴입니다. 터미널에서 라이브러리 형태로 다운로드 받을 수 있고, 사용하는 방법도 굉장히 간단합니다.







## 설치

텐서보드는 원래 `TensorFlow`에서 머신러닝을 할 때, 시각화를 편하게 할 수 있도록 제공해주는 라이브러리로 `TensorFlow`를 설치하면 자동으로 함께 설치됩니다. 그러나 요즘 머신러닝을 하는 대부분의 사람들은 `Pytorch`를 사용하고 저 역시 그렇습니다. 이 포스팅에서는 `Pytorch`에서 텐서보드를 사용하는 방법에 대해서 소개합니다. 



`Pytorch` 라이브러리에서는 텐서보드를 함께 제공하지 않기 때문에, 별도로 설치를 해줍니다. `pip` 패키지 관리자를 통해 간단하게 설치가 가능합니다.

```
pip install tensorboard
```







## 코드 사용법

텐서보드도 여느 라이브러리와 마찬가지로 파이썬 파일 내에서 메서드의 호출로 사용됩니다. 아래는 REINFORCE 알고리즘을 사용하는 간단한 강화학습 코드 예제입니다.



```python
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='logs/REINFORCE')

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
    random_input = torch.rand(4)
    writer.add_graph(pi, random_input)
    
    score = 0.0
    print_interval = 20
    
    
    for n_epi in range(10000):
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
        
        if n_epi % print_interval==0 and n_epi!=0:
            writer.add_scalar('reward', score/print_interval, global_step=n_epi)
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0

    writer.close()
    env.close()
    
if __name__ == '__main__':
    main()
```



가장 먼저 `SummaryWriter`를 import 해옵니다.

``` python
from torch.utils.tensorboard import SummaryWriter
```



그리고 다음으로 `SummaryWriter`의 객체를 생성합니다.

```python
writer = SummaryWriter(log_dir='logs/REINFORCE')
```



`log_dir` 인자는 단어 그대로 로그를 저장할 장소를 지정하는 역할을 합니다. 디렉터리가 존재하지 않는 경우에는 디렉터리가 자동으로 새롭게 생성됩니다. 



<img src="/images/2025-01-15-Tensorboard/image-20250115105033489.png" alt="image-20250115105033489" style="zoom:50%;" />

좌측 상단에 새롭게 디렉터리가 생성되고, 로그가 저장된 것을 확인할 수 있습니다.



저장하고자 하는 변수를 지정해서 `writer`에 저장합니다.

```python
writer.add_scalar('reward', score/print_interval, global_step=n_epi)
```



`'reward'`는 그래프의 이름이 되고, 그 뒤의 인자(`score/print_interval`)가 바로 텐서보드에 기록되는 변수가 됩니다. 마지막 인자인 `global_step`은 현재 기록되는 변수 값의 에폭 값, 그러니까 x축 값을 지정해서 저장하고 싶을 때 사용합니다.



마지막으로 코드 말단에 아래와 같이 `writer`객체를 닫는 명령어를 추가합니다.

```python
writer.close()
```







## 텐서보드 확인

위와 같이 코드를 작성하면 지정한 디렉터리에 로그가 차곡차곡 쌓이게 될 텐데요, 이제 직접 텐서보드를 실행해서 시각화 결과를 확인해보겠습니다.



일단 현재 터미널은 파이썬 파일을 실행중이라 다른 명령어 입력이 어렵기 때문에 새로운 터미널을 켜서 아래의 명령어를 입력합니다(텐서보드가 설치되어 있는 가상환경을 실행했는지 확인할 것).

```
tensorboard --logdir=logs/REINFORCE
```



위와 같이 실행하면 아래와 같이 로컬 호스트 주소를 받을 수 있습니다.

<img src="/images/2025-01-15-Tensorboard/image-20250115110032102.png" alt="image-20250115110032102" style="zoom:50%;" />

`https://localhost:6006/`이라는 주소를 받았는데요, 접속하면 아래와 같습니다.

<img src="/images/2025-01-15-Tensorboard/image-20250115110155960.png" alt="image-20250115110155960" style="zoom:50%;" />

그래프의 이름, 저장되는 변수 값, 그리고 에폭 크기까지 올바르게 저장되는 것을 확인할 수 있습니다. 그런데 텐서보드를 처음 실행하면, 그래프가 자동으로 업데이트가 되지 않습니다. 텐서보드를 자동으로 업데이트하게 할 수 있는데요, 우측 상단의 톱니바퀴 버튼을 누르면 아래와 같은 창을 확인할 수 있습니다.



<img src="/images/2025-01-15-Tensorboard/image-20250115110357012.png" alt="image-20250115110357012" style="zoom:50%;" />

여기에서 업데이트 빈도를 설정할 수 있습니다.









