---
layout: single

title:  "강화학습 알고리즘: TRPO(Trust Region Policy Optimization)"

categories: RL_Algorithm

tag: [Reinforcement Learning, Policy Gradient, REINFORCE]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
published: false
---



**글에 들어가기 앞서...**

이 포스팅은 '**강화학습 알고리즘**'에 대한 내용을 담고 있습니다.



자료 출처: 단단한 강화학습, Reinforcement Learning An Introduction , 2nd edition. 리처드 서튼, 앤드류 바트로, 김성우(옮긴이), <https://www.youtube.com/watch?v=Ukloo2xtayQ>









# TRPO(Trust Region Policy Optimization)

DDPG는 굉장히 훌륭한 알고리즘이지만, 성능의 단조 향상이 보장되지 않는다는 큰 문제점이 있습니다. 성능이 단조 향상하지 않는다는 것은 우리가 파라미터를 최적화할 때, 목적함수의 증가가 일어나지 않을 수 있다는 것을 의미합니다. 이러한 불명확함은 알고리즘의 학습에 불안정성을 야기하고, 알고리즘이 Step Size에 굉장히 민감하게 반응하게 만듭니다.

**TRPO(Trust Region Policy Optimization)**은 DDPG의 상기 문제점을 **신뢰 구역(Trust Region)**이라는 아이디어를 도입해 해결합니다. 신뢰 구역은 KL(Kullback-Leibler)-divergence로 정의되는데, 이 구역 내에서 파라미터 업데이트가 일어날 때 성능의 단조 향상이 보장되고, Step Size의 영향으로부터 벗어나게 됩니다.

TRPO 논문에는 단조 향상이 이론적으로 완벽하게 보장되는 알고리즘을 소개하는데요, TRPO의 성능은 매우 훌륭하지만, 이 알고리즘을 실제로 구현하는 것이 꽤나 어렵습니다.. 때문에 이론적인 TRPO에 여러 번 근사하는, 보다 실용적인 알고리즘을 별도로 소개합니다. 그럼에도 구현이 꽤나 복잡하고 방대한 계산량을 필요로 하기 때문에 정책망의 크기가 조금만 커져도(CNN, RNN의 수준을 감당할 수 없음) 계산량을 감당하기 어려워집니다.







## Trust Region

<img src="/images/2025-01-20-Reinforcement_Algorithm_TRPO/image-20250120145843849.png" alt="image-20250120145843849" style="zoom:50%;" />

위의 사진을 통해 'Trust Region이 왜 필요한가?'에 대한 직관적인 이해를 얻을 수 있습니다. DDPG는 경사 상승을 통해 파라미터를 업데이트합니다. 지금 밟고있는 땅에서 올라가는 방향으로 걸음을 내딛는 일과 동일합니다. 여기에서 가장 큰 이슈는 얼마나 큰 발걸음을 할 수 있는가입니다. 알고리즘이 빠르게 수렴하기 위해서는 걸음을 크게 설정해야 하지만, 그랬다가는 낭떨어지로 빠질 위험이 굉장히 높아집니다. 그렇다고 걸음을 느리게 작게 설정하면 알고리즘이 수렴할 때까지 굉장히 오랜 시간을 소요하게 됩니다.

신뢰 구역은 내가 얼마나 발을 내딛을 수 있는지, 그 범위를 의미합니다. 신뢰 구역이 에이전트에게 전해지면 에이전트는 해당 구역 내에서는 안전하다는게 보장되기 때문에 가장 높이 올라가는 방향으로 큰 발걸음을 내딛을 수 있게 됩니다. 때문에 Step Size로부터 자유로워지고, 성능의 단조 향상이 보장되기 때문에 국소 최적점이든 전역 최적점이든 아무튼 최적점에 도달하게 됩니다.







## TRPO 알고리즘의 핵심 아이디어

$$
\eta(\pi) = E_{s_0, a_0, ...}[\sum_{t=0}^{\infty}\gamma^tr(s_t)]
$$



TRPO 알고리즘의 손실함수입니다. 특정 에피소드의 모든 상태에서 받은 보상의 감쇠합으로 정의됩니다. 


$$
\eta(\pi)
$$

$$
= \eta(\pi_{old}) + E_{\tau \sim \pi}[\sum_{t=0}^\infty \gamma^tA_{\pi_{old}}(s_t, a_t)]
$$

$$
= \eta(\pi_{old}) + \sum_s\rho_{\pi}(s)\sum_a\pi(a|s)

A_{\pi_{old}}(s, a)
$$



새로운 정책의 목적 함수 값은 이전 정책의 목적 함수 값과 정책으로 정의할 수 있습니다(자세한 유도는 논문 참조). 위 식에서 단조 증가가 보장되기 위해서는 아래의 식이 항상 0보다 커야 합니다.


$$
\sum_a\pi(a|s)

A_{\pi_{old}}(s, a)
$$


그런데 위의 식을 바로 적용하는 것은 꽤나 어려운 일인데요, 아직 $\pi$를 구하는 중이기 때문에 그 새로운 정책의 상태 방문 빈도($\rho_{\pi}(s)$)를 알아내는 것은 굉장히 많은 샘플링을 필요로 합니다. 때문에 위 수식을 조금 다른 알고리즘으로 근사하게 되는데, 그렇게 하면 또 위 식에서 음수가 나올 가능성이 발생하게 되고, 이 때 단조 향상을 보장하기 위해서 신뢰 구역을 설정하게 됩니다.







## 근사 TRPO 알고리즘


$$
\eta(\pi)= \eta(\pi_{old}) + \sum_s\rho_{\pi}(s)\sum_a\pi(a|s)

A_{\pi_{old}}(s, a)
$$

$$
L_{\pi_{old}}(\pi)= \eta(\pi_{old}) + \sum_s\rho_{\pi_{old}}(s)\sum_a\pi(a|s)

A_{\pi_{old}}(s, a)
$$



위에서 새로운 정책의 상태 방문 빈도를 계산하기 어렵다고 했는데요, 새로운 정책의 상태 방문 빈도를 사용하지 않고 이전 정책의 상태 방문 빈도를 사용하는 방법이 있습니다. 이를 **국소 근사(Local Approximate)**라고 합니다. 그냥 이전 정책의 상태 방문 빈도로 갈아끼워도 괜찮은건가.. 싶은데, 생각보다 정말 괜찮은 근사 방법입니다.



<img src="/images/2025-01-20-Reinforcement_Algorithm_TRPO/image-20250121110759340.png" alt="image-20250121110759340" style="zoom:40%;" />


$$
L_{\pi_{\theta_0}}(\pi_{\theta_0}) = \eta(\pi_{\theta_0})
$$

$$
\nabla_\theta L_{\pi_{\theta_0}} (\pi_{\theta}) |_{\theta = \theta_0} = \nabla_\theta\eta(\pi_{\theta})|_{\theta = \theta_0}
$$



왜냐하면 일단 이전 정책과 동일한 정책일 때 같은 값을 가지고, 그 때의 기울기 값도 동일합니다. 정리하면 해당 지점의 위치와 그 위치에서 나아가는 방향성이 동일하다는 말인데, 때문에 작은 step에 대해서는 비슷한 값을 가지게 됩니다. 하지만 어느정도의 step까지 괜찮은 건지는 아직 알 수 없습니다. 이런 문제를 다루기 위해서 **Conservative Policy Iteration Update**를 사용합니다.


$$
\pi(a|s) = (1-\alpha)\pi_{old}(a|s) + \alpha\pi'(a|s)
$$

$$
where \space \pi' = \arg \max_\pi L_{\pi_{old}}(\pi)
$$



**Conservative Policy Iteration Update**에서는 기존 정책과 새로운 정책의 **짬뽕 정책(Mixture Policy)**, 구체적으로는 EWMA를 사용한 지연 최적화 정책을 사용합니다. 이 때의 새로운 정책은 $L$상에서 가장 높았던 지점을 도달케하는 정책($\pi'$)을 의미합니다. **위의 업데이트 방법을 따랐을 때 우리가 얻을 수 있는 최고의 장점은 하한을 알 수 있다는 점입니다.**


$$
\eta(\pi) \geq L_{\pi_{old}}(\pi) - \frac{2\epsilon \gamma}{(1-\gamma)^2} \alpha^2
$$

$$
where \space \epsilon = \max_s[E_{a \sim \pi'(a|s)}[A_{\pi_{old}}(s, a)]]
$$

위 식에서의 $\epsilon$은 $\pi'$ 정책에 따라서 행동을 할 때, 과거 정책($\pi_{old}$)의 어드밴티지 함수의 평균값을 각 상태별로 구할 수 있을 텐데, 상태별 평균값들 중 가장 큰 값을 $\epsilon$이 됩니다. 하지만, 짬뽕 정책에서만 하한이 적용된다는 점 때문에 실제로는 사용이 굉장히 제한적입니다.





