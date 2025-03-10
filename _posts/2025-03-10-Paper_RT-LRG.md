---
layout: single

title:  "Learning-based Initialization of Trajactory Optimization for Path-following Problems of Redundant Manipulators"

categories: Paper

tag: [Robotics, Trajectory_Optimization]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
published: true

---





이 포스팅은 '**Learning-based Initialization of Trajactory Optimization for Path-following Problems of Redundant Manipulators**'에 대한 내용을 담고 있습니다.



논문 주소: <https://ieeexplore.ieee.org/document/10161426>









# Learning-based Initialization of Trajactory Optimization for Path-following Problems of Redundant Manipulators

이 논문에서는 로봇 매니퓰레이터의 말단부를 제어하는 Path-Following 문제를 다룹니다. 말단부가 특정 경로를 따라가게 하기 위해서는 그에 맞는 관절 궤적이 필요한데, 이 때 아무런 관절 궤적을 선택하는 것이 아니라 로봇의 관절 한계를 넘어서지 않는지, 특이점을 지나는 경로인지, 주변에 존재하는 물체와 충돌하는지 등등 고려해야할 대상이 많습니다. 그리고 그 중에서도 가장 적절한 관절 궤적을 찾아내는 것은 꽤 까다로운 일입니다.

가장 전통적으로는 differential IK를 사용해 국소 해를 찾는 방법이 있습니다. 이 방법은 너무 근시안적인 측면이 있고, 또, 특이점 주변에서 잘 동작하지 않는다는 문제점이 있었습니다. 다른 방법으로는 전역 탐색을 통해 최적해를 찾는 방법도 있는데, 이 방법은 너무 시간이 많이 걸린다는 단점이 있습니다. 최근에는 **TO(Trajectory Optimization)** 방법론이 가장 잘 사용되는데, 이 방법에서는 경로를 추종하도록 하는 목적함수를 설정하고, 최적화를 통해 초기 경로에서 점점 요구되는 경로에 맞아지도록 조금씩 업데이트됩니다. 근데 당연히 관절 궤적을 찾는 문제는 non-convex하기 때문에 초기 경로에 따라 다른 수렴 값을 가지게 됩니다. 그렇기 때문에 TO는 초기 관절 궤적이 굉장히 중요합니다.

관절 궤적 상에서 선형으로 초기 궤적을 설정하는 방법도 있고, 역기구학 해를 greedy하게 선택해 초기 궤적을 설정하는 방법도 있습니다. 이 논문에서는 강화학습을 기반으로 하는 **RT-ITG**를 통해 초기 궤적을 만들어내 이전의 방법론들보다 비교했을 때 훨씬 더 좋은 성능을 보여줍니다.







## Problem Definition and Motivation


$$
X = [x_0, x_1, x_2, ..., x_{N-1}] \in \mathcal X
$$

$$
\xi = [q_0, q_1, ..., q_{N-1}] \in \Xi
$$



말단부의 궤적과, 관절 궤적을 위와 같이 표현할 수 있습니다. 관절 변수의 차원은 사용하는 로봇의 자유도로 설정됩니다. 이 논문에서는 7자유도 Fetch 로봇을 사용해서 7을 설정됩니다.


$$
\mathcal U[\xi] =\mathcal U_{\text{pose}}[\xi] + \lambda_1\,\mathcal  U_{\text{obs}}[\xi] + \lambda_2\, \mathcal U_{\text{smooth}}[\xi]
$$


TO의 목적 함수는 위와 같이, 말단부의 포즈가 요구되는 궤적과 일치하는지, 주변 환경과 충돌하는지, 그리고 마지막으로 로봇의 행동이 부드러운지에 대한 평가치를 모두 합산합니다.







## RL-based Initial Trajectory Generator

**RL-ITG**는 초기 경로를 강화학습 에이전트의 일련의 행동들을 통해 생성합니다. 문제를 어떻게 정의되고 목적함수는 또 어떻게 설정되는지 살펴봅니다.





### Formulation of an MDP and an RL objective function



$$
\mathcal M_X = \langle  \mathcal S, \mathcal A, \mathcal R_\mathcal I, \mathcal T, \mathcal Q_0, \gamma \rangle_{X \sim \mathcal X}
$$

$$
\mathcal R_\mathcal I : = \text{time varying reward function}
$$

$$
\mathcal I = \{ i \in \mathbb N | 0 \leq i \lt N\}: \text{time steps}
$$



MDP 문제 정의는 위와 같이 설정됩니다. 가장 처음에 시작하는 포즈는, 역기구학 해 중에서 샘플링을 통해 결정합니다.



$$
\text{maximize}_\pi \; \mathbb{E}_{X \sim P(X)} \left[
  \mathbb{E}_{(s_i,a_i) \sim \rho_\pi,\; q_0 \sim Q_0}
  \left[ \sum_{i=0}^{N-1} \gamma^i \cdot R_i(s_i, a_i) \right]
\right]
$$



목적함수는 다른 강화학습 문제와 동일한 형식으로 설정됩니다.


$$
s_i = (q_i, p_i^\text{link}, p_i^\text{target}, z_{env})
$$

$$
p_i^\text{link} \in \mathbb R^{(d + 1, 9)}
$$



상태는 위와 같이 현재 관절 각도, 링크별 위치 및 방위, 현재 말단부 위치와 이 다음 K개 목표 말단 위치 사이의 상대적 거리, 마지막으로 3차원 occupancy grid map과 같은 인지 정보의 임베딩 벡터로 구성됩니다. 


$$
a_i = \Delta q_i \in \mathbb R^d
$$

$$
q_{i+1} = q_i + \Delta q_i
$$



 action은 관절 공간의 벡터로 얻게되고, 이 값을 현재 관절각도에 더해 다음 상태로 이동하게 됩니다.



#### Reward Function

이어서 보상함수가 어떻게 설정되는지 살펴봅니다.


$$
e^\text{pos}_{ i} = \| x^\text{pos}_{ i} - \hat{x}^\text{pos}_{ i}(q_i) \|_2
$$

$$
e^\text{pos}_{ i} = 2 \cdot \cos^{-1}\Bigl( \Bigl| \langle x^\text{quat}_{ i},\, \hat{x}^\text{quat}_{ i}(q_i) \rangle \Bigr| \Bigr)
\tag{5}
$$

$$
\langle \cdot \rangle: \text{inner product}
$$



먼저 말단부의 위치와 방위에 대해 각각 거리 오차를 구합니다. 


$$
f(e, w) = w_0 \cdot \exp(-w_1 \cdot e) - w_2 \cdot e^2,\quad f(e,w) \in \mathbb{R}
$$

$$
R_{\text{task}, i} = f(e^\text{pos}_{ i}, w^\text{pos}) + f(e^\text{rot}_{ i}, w^\text{rot}) \cdot \mathbf{1}_{\sqrt{e^\text{pos}_{ i}} \leq 5\,\text{cm}}
$$



위치 방위에 대한 오차를 그냥 합으로 사용하기 보다는, 각각에 대해 가중치를 부여하는 정규화 함수를 시켜 얻은 결과값의 합을 사용합니다. 그런데 방위에 대한 오차는 처음부터 바로 사용하지 않고, 일단 거리 오차를 먼저 줄이게 하기 위해서 거리 오차가 5cm 이하로 줄어든 경우에만 방위 오차를 사용하도록 합니다.


$$
e_{\text{im}, i} = \Bigl\| \Bigl( I - J(q_i)^\dagger J(q_i) \Bigr) \Bigl( \xi_{\text{demo}}[i] - q_i \Bigr) \Bigr\|_2
$$

$$
R_{\text{im}, i} = f(e^\text{im}_{ i}, w^\text{im})
$$



다음으로 에이전트가 전문가 TO 방법론의 **Null-space**상의 행동을 모사하도록 하기 위해서 SDF를 기반으로 한 전문가 TO 방법론으로 얻은 관절 궤적 경로와의 거리 보상에 반영합니다. 그런데 두 관절 궤적의 차이를 그대로 반영하면 강화학습 에이전트가 전문가 TO 방법론의 경로를 따르게 되는데, 이렇게 학습시키면, 에이전트가 전문가 TO 방법론이 만들어내는 경로를 완전히 따라 만들게 될 뿐, 그보다 더 좋은 경로를 생성하지 못합니다. 여기에서 학습시키고자 하는 것은 제약 조건 내에서 부드럽고 일관된 Null-space 모션이므로, 단위행렬에 자코비안과 자코비안의 무어-펜로즈 역행렬을 행렬곱한 값을 빼 얻은 행렬을 곱해 Null-space에 투영합니다. 이렇게 얻은 거리를 마찬가지로 정규화 함수를 통과시켜 얻는 출력 값을 보상으로 사용합니다.


$$
R_{\text{cstr}, i} = R_{C, i} + R_{J, i} + R_{S, i} + R_{E, i}
$$

$$
R_{C, i} &= -10 \cdot \mathbf{1}_{\text{collision}}, \\
R_{J, i} &= -1 \cdot \mathbf{1}_{q_i < q_{\min} \text{ or } q_i > q_{\max}}, \\
R_{S, i} &= -0.1 \cdot \mathbf{1}_{\det\bigl(J(q_i)J(q_i)^T\bigr) < 0.005}, \\
R_{E, i} &= -3 \cdot \mathbf{1}_{\sqrt{e_{\text{pos}, i}} > 20\,\text{cm}}.
$$
















## Experiments

























