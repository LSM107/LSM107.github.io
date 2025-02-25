---
layout: single

title:  "Proprioceptive Sensor-Based Simultaneous Multi-Contact Point Localization and Force Identificaion for Robotic Arms"

categories: Paper

tag: [Robotics, Contact, MCP]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
published: false
---





이 포스팅은 '**Proprioceptive Sensor-Based Simultaneous Multi-Contact Point Localization and Force Identificaion for Robotic Arms**'에 대한 내용을 담고 있습니다.



논문 주소: <https://arxiv.org/pdf/2303.03825>









# Proprioceptive Sensor-Based Simultaneous Multi-Contact Point Localization and Force Identificaion for Robotic Arms

최근들어 인간과 작업공간을 공유하는 협동 로봇의 사용이 점차 증가하고 있습니다. 때문에 로봇이 일으키는 물리적 상호작용을 정확하게 감지하고 대처하는 것이 중요한 과제가 되고 있습니다. 이를 해결하기 위해 크게 두 가지 접근 방법이 사용됩니다. 하나는 아예 처음부터 물리적 상호작용을 일으키지 않는 계획을 만드는 것이고, 다른 하나는 물리적 상호작용이 발생했을 때 이를 빠르게 감지하고 대처(passivity-based controller)하는 방법입니다. 물리적 상호작용을 감지한다는 것은 접촉 지점과 힘을 추정한다는 말인데요, 이를 잘 감지하기 위해 촉각 센서(tactile sensor)를 사용하기도 하지만, 애초에 로봇 팔에 설치되어 나오지도 않고, 센서 자체도 개선해야 할 여지가 많이 남아있습니다. 현재로서 가장 실용적인 접근은 로봇 팔에 설치된 조인트 힘, 토크 센서인 **고유감각(proprioceptive)** 센서를 활용하는 방법입니다. 고유감각 센서를 활용해 접촉을 추정하려는 많은 연구들이 있어왔는데, 이 논문에서는 Particle Filter를 활용한 접근 방법을 사용합니다. 기존의 Particle Filter 방식은 크게 두 가지 문제점이 있습니다. 하나는 실행 시간이 오래 걸린다는 점이고, 다른 하나는 특이점 문제가 존재한다는 점입니다.

이 논문에서는 이전에 다른 연구에 소개된 메쉬 전처리 방식을 PF에 도입해 실행 시간이 오래 걸리는 문제점을 해결하고 로봇 베이스 부분에 힘, 토크 센서를 추가로 부착시켜 특이점 문제를 완화시킵니다. 







## QP-Based Contact Force Identification Without Localization

우선 여기에서는 <u>접촉 위치가 주어진 상태</u>에서 얼마만큼의 접촉 힘이 가해져야 하는지 추정하는 방법을 설명합니다. 우리가 이미 접촉 위치를 알고 있을 때, 각 조인트와 베이스에 이 정도의 힘과 토크가 가해진다면 얼마만큼의 힘이 가해져야 하는지를 이차 계획법으로 구하는 과정을 설명합니다.





### Estimating External Joint Torque using JTS


$$
M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) = \tau_j + \tau_{\text{ext}}
$$


위 식은 조인트 토크와 로봇 팔의 상태 사이의 관계를 나타내는 동역학 방정식입니다. 만약에 로봇에 k개의 접촉 있을 때, 아래와 같이 외력에 의해 발생하는 토크와 조인트 토크 사이의 관계식을 표현할 수 있습니다.


$$
\tau_{\text{ext}} = \sum_{i=1}^{k} J_i^T(q, r_{c,i})\, F_{\text{ext},i}
$$



위에서 사용된 $J$는 위치 자코비안으로, 힘이 가해지는 위치에 따라 각 조인트에 전달되는 토크의 방향성을 나타냅니다. 외력에 가해진 힘과 그 위치를 알면 각 조인트에 전달되는 토크의 힘을 구할 수 있습니다.


$$
\hat{\tau}_{\text{ext}}(t) = K_o \left\{ p - \int_{0}^{t} \Bigl( \tau_j + n(q,\dot{q}) + \hat{\tau}_{\text{ext}} \Bigr) \, ds \right\}
$$

$$
n(q, \dot q) = C^T(q, \dot q)\dot q - g(q)
$$



그런데 외력에 의한 토크를 정확하게 측정하기가 어렵기 때문에 **momentum-based observer**에서 속도와 각속도를 사용해 좀 더 정확한 외력에 의한 토크를 추정합니다.





### Estimating Wrench Caused by Contact forces using Base F/T sensor


$$
W_{\text{ext},b} = \sum_{i=1}^{k} 
\underbrace{
\begin{bmatrix} I_{3\times3} \\ \operatorname{skew}(r_{c,i}) \end{bmatrix}}_{ \triangleq X_{c,i}(r_{c, i})}

 F_{\text{ext},i}
$$

$$
\text{skew}(a)b = a \times b
$$



위는 외력에 의해 베이스에 걸리는 힘과 토크를 계산하는 식입니다. 이 논문의 주요 contribution 중 하나가 베이스에서 측정한 힘, 토크를 통해 특이점 문제를 해결하는 건데요, 위 식을 통해 렌치 값으로부터 힘의 크기를 역으로 얻어낼 수 있게 됩니다.





### QP Formulation for Contact Force Identification 

이제 관절 토크 센서와 베이스 센서에서 얻은 정보를 사용해, 주어진 접촉 지점에서 얼마만큼의 접촉힘이 발생하는지 추정하는 문제를 이차계획법을 통해 구하는 과정을 살펴봅니다.


$$
\hat{W} = \begin{bmatrix} \hat{\tau}_{\text{ext}} \\ \hat{W}_{\text{ext},b} \end{bmatrix} \in \mathbb{R}^{n+6}
$$

센서를 통해 얻게되는 정보는 베이스 센서에서 6개, 그리고 n개 개별 조인트에서 1개씩, 총 n + 6의 차원을 가지게 됩니다. 그리고 접촉한 위치만 알 뿐이지, 무슨 방향으로 힘이 가해지고 있는지는 모릅니다. 접촉 위치가 주어지는 상황에서 힘의 방향으로 가능한 구역들이 있는데, 이를 [**마찰 원뿔(Friction Cone)**](https://lsm107.github.io/paper/Paper_CDM/#optimization-based-contact-point-localization)이라고 말합니다.





























































































































