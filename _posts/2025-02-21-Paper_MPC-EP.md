---
layout: single

title:  "Online Multi-Contact Feedback Model Predictive Control for Interactivate Robotic Tasks"

categories: Paper

tag: [Robotics, Collision, MPC, Control]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
published: True
---





이 포스팅은 '**Online Multi-Contact Feedback Model Predictive Control for Interactivate Robotic Tasks**'에 대한 내용을 담고 있습니다.



논문 주소: <https://arxiv.org/pdf/2411.10049>









# Online Multi-Contact Feedback Model Predictive Control for Interactivate Robotic Tasks

과거와 달리 인간과 로봇이 동일한 작업 공간을 공유하는 경우가 있고, 그렇게 설계된 로봇들도 많이 나오고 있습니다. 인간과 작업공간을 공유하는 경우, 당연하게도 돌발요소가 더 많이 존재합니다. 로봇 팔은 쇳덩이로 만들어지기 때문에 인간과 부딪칠 경우 정말 치명적인 결과를 초래할 수 있거든요. 따라서 예기치 못한 물리적 상호작용에 신속하고 효과적으로 대처할 수 있는 능력이 협동로봇에게는 정말 중요합니다. 꼭 사고 때문이 아니라고 하더라도, 여러 물리적인 상호작용을 포착하는 능력은 태스크를 수행할 때에 반드시 필요합니다. 예를 들어 테이블 연마를 하는 작업에서는 테이블에 힘을 줌과 동시에 위치를 움직여야 합니다. 이렇게 두 가지 작업을 동시에 수행하도록 하는 것은 꽤나 까다로운 작업인데, 이런 종류의 태스크를 **Hybrid Motion-Force Control**이라고 합니다.

요즘에는 컴퓨팅 속도가 크게 발전해 최적화를 활용한 문제 해결 방법이 많이 사용됩니다. **MPC(Model Predictive Control)**가 그 대표적인 방법입니다. MPC를 사용하기 위해서는 로봇과 환경의 접촉 모델에 대한 명시적인 설정이 필요합니다. 그런데 기존의 방법론들에서는 단일 접촉에 대해서, 그것도 end effector에서 발생한 접촉에 대해서만 다룰 수 있습니다. 당연히 현실에서는 여러 군데에서 접촉이 동시에 발생할 수도 있고, end effector가 아닌 다른 부분에서도 접촉이 발생할 수 있는데, 이런 상황들은 기존 방법론으로 해결할 수 없습니다.

이 논문에서는 **MCP-EP(Multi-Contact Particle Fileter with Exploratiobn Particle)**이라는 새로운 알고리즘을 사용해 다중 접촉과 접촉 힘을 추정합니다. 많은 방법론에서 **전신 동역학(Full-Body Dynamics)**을 사용하지 않고 일부 단순화 가정을 사용해 정확성을 일부 포기하는데, MCP-EP에서는 DDP와효율적인 구현을 통해 전신 동역학을 사용하면서도 실시간 처리가 가능합니다.







## System and Contact Modeling

로봇의 상태에 대한 MPC 시스템 모델은 아래와 같이 정의됩니다.


$$
\dot x(t) = f(x(t), u(t), \lambda(t)) \quad x(0) = \tilde x
$$

$$
\lambda(t) = \{\lambda_1(t), \lambda_2(t), ...,\lambda_k(t) \} \in \mathbb R^{3 \times k}
$$

$$
\lambda_i(t) = g(x(t); \theta_i), \quad \theta_i = h(\tilde r_{c, i}, \tilde\lambda_i), \quad \forall i = \{  1, 2, ..., k\}
$$



시스템 모델의 입력으로 현재 로봇의 상태(관절 각도, 각속도)와 제어 입력(조인트 토크), 그리고 하나 이상의 접촉을 통해 로봇 팔이 받는 외력 집합을 받습니다. 외력을 계산하기 위해서 **Spring Contact Model**을 사용하는데요, 이 모델은 입력으로 현재 로봇의 상태와 접촉 피드백을 받습니다. 접촉 피드백은 추정되는 접촉 위치와 힘을 바탕으로 계산됩니다. 그런데 이 접촉 모델에서는 계산할 때, 현재의 제어 입력이 필요하지 않습니다. 만약 제어 입력이 접촉 모델을 계산할 때 필요하다면, 제어 입력의 변화가 접촉력을 변화시키고, 접촉력의 변화가 다시 제어 입력의 값을 변화시키는 순환 관계로 인해 최적화를 풀 때 훨씬 더 복잡한 문제가 됩니다. 여기에서는 제어 입력이 접촉 모델에 영향을 주지 않기 때문에 보다 안정적으로 최적화를 수행할 수 있습니다.





### Contact-Involved Robot Dynamics 



$$
M(q)\ddot q + C(q, \dot q) \dot q + g(q) = \tau_c + \tau_{ext}
$$

$$
M: \text{inertia matrix}
$$

$$
C: \text{Coriolis/centrifugal matrix}
$$

$$
g: \text{gravity}
$$



접촉이 포함된 로봇의 동역학은 위와 같이 표현됩니다. 각가속도에 관성 행렬을 곱한 값에, 각속도에 코리올리/원심력 행렬을 곱한 값, 그리고 마지막으로 현재 로봇의 각도에 따라 받게되는 중력의 힘의 합이 로봇의 관절에 가해지는 토크와, 외력에 의해 발생하게 되는 토크의 합과 동일합니다. 그리고 이 '외력에 의해 발생하는 토크'는 아래의 식을 통해 구할 수 있습니다.


$$
\tau_{ext} = \sum_{i=1}^k J_i^T(q, r_{c, i})\lambda_i
$$

$$
J_i(q, r_{c, i})  \in \mathbb R ^{3 \times n} \quad \text{(positional Jacobian)}
$$



**위치 자코비안(Positional Jacobian)**은 접촉 위치변화에 따른 로봇 관절의 변화 방향을 의미합니다. 예를 들어 우리가 문 손잡이를 당길 때, 손잡이가 우리에게 가하는 힘의 방향이 달라짐에 따라, 우리가 각 팔의 관절에 가해야하는 힘의 방향도 계속 달라지게 되는데, 이를 표현하는게 위치 자코비안입니다.

































