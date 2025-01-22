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
published: True
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



TRPO 알고리즘의 손실함수는 기대 이득으로 설정됐습니다. 특정 에피소드의 모든 상태에서 받은 보상의 감쇠합으로 정의됩니다. 


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


그런데 새로운 정책의 상태 방문 빈도($\rho_{\pi}(s)$)를 알아내는 것은 굉장히 많은 샘플링을 필요로 합니다. 때문에 전체 수식을 조금 다른 알고리즘으로 근사하게 되는데, 그렇게 하면 또 바로 위 식의 값이 음수가 될 가능성이 발생하게 됩니다. 이런 문제들을 해결하기 위해 신뢰 구역을 설정하게 되는건데요, 아래에서 자세하게 살펴보겠습니다.







## Conservative Policy Iteration Update


$$
\eta(\pi)= \eta(\pi_{old}) + \sum_s\rho_{\pi}(s)\sum_a\pi(a|s)

A_{\pi_{old}}(s, a)
$$

$$
L_{\pi_{old}}(\pi)= \eta(\pi_{old}) + \sum_s\rho_{\pi_{old}}(s)\sum_a\pi(a|s)

A_{\pi_{old}}(s, a)
$$



위에서 새로운 정책의 상태 방문 빈도를 계산하기 어렵다고 했는데요, 새로운 정책의 상태 방문 빈도를 사용하지 않고 이전 정책의 상태 방문 빈도를 사용하는 방법이 있습니다. 이를 **국소 근사(Local Approximate)**라고 합니다. 그냥 이전 정책의 상태 방문 빈도로 갈아끼워도 괜찮은건가.. 싶은데, 생각보다 정말 괜찮은 근사 함수가 됩니다.

- 이전 정책(Old Policy): $\pi_{old}$
- 새로운 정책(New Policy): $\pi_{old}$
- $L$에서 찾은 가장 좋은 정책: $\pi'$



<img src="/images/2025-01-20-Reinforcement_Algorithm_TRPO/image-20250121110759340.png" alt="image-20250121110759340" style="zoom:40%;" />

$$
L_{\pi_{\theta_0}}(\pi_{\theta_0}) = \eta(\pi_{\theta_0})
$$

$$
\nabla_\theta L_{\pi_{\theta_0}} (\pi_{\theta}) |_{\theta = \theta_0} = \nabla_\theta\eta(\pi_{\theta})|_{\theta = \theta_0}
$$



왜냐하면 일단 이전 정책과 동일한 정책일 때 같은 값을 가지고, 그 때의 기울기 값도 동일합니다. 정리하면 해당 지점의 위치와 그 위치에서 나아가는 방향성이 동일하다는 말인데, 때문에 작은 step에 대해서는 비슷한 변화를 가지므로 원래 손실함수의 향상을 가져올 수 있습니다. 하지만 어느정도의 step까지 괜찮은 건지는 아직 알 수 없습니다. 이런 문제를 다루기 위해서 **Conservative Policy Iteration Update**를 사용합니다.


$$
\pi(a|s) = (1-\alpha)\pi_{old}(a|s) + \alpha\pi'(a|s)
$$

$$
where \space \pi' = \arg \max_\pi L_{\pi_{old}}(\pi)
$$



**Conservative Policy Iteration Update**에서는 기존 정책과 새로운 정책의 **짬뽕 정책(Mixture Policy)**, 구체적으로는 EWMA를 사용한 지연 최적화 정책을 사용합니다. 이 때의 새로운 정책은 $L$상에서 가장 높았던 지점을 도달케하는 정책($\pi'$)을 의미합니다. **위의 짬뽕 정책을 사용할 때 우리가 얻을 수 있는 최고의 장점은 하한을 알 수 있다는 점입니다.  이 하한이 지속적으로 상승하도록 보장하는 최적화를 수행하면, 원래의 손실함수 또한 지속적으로 향상시킬 수 있을 것입니다.


$$
\eta(\pi) \geq L_{\pi_{old}}(\pi) - \frac{2\epsilon \gamma}{(1-\gamma)^2} \alpha^2
$$

$$
where \space \epsilon = \max_s[E_{a \sim \pi'(a|s)}[A_{\pi_{old}}(s, a)]]
$$

위 식에서의 $\epsilon$은 $\pi'$ 정책에 따라서 행동을 할 때, 과거 정책($\pi_{old}$)의 어드밴티지 함수의 평균값을 각 상태별로 구할 수 있을 텐데, 상태별 평균값들 중 가장 큰 값을 $\epsilon$이 됩니다. **하지만 짬뽕 정책에서만 하한이 적용된다는 점 때문에 실제로는 사용이 굉장히 제한적입니다.** 사실 $\epsilon$을 실제로 계산하는 것도 어렵지만, 짬뽕 정책 자체를 실제 코드로 적용하는 것도 어렵습니다.







## General Stochastic Policy

하한을 알 수 있다는 큰 장점이 있음에도, EWMA 꼴로 엮여있는 짬뽕 정책에 대해서만 하한을 알 수 있기 때문에 이 하한을 활용할 수 있는 범위가 제한적이었습니다. 짬뽕 정책에 대해서 알려진 하한을 통해 그냥 어떤 새로운 정책에 대한 하한을 알 수 있도록 식을 변형합니다.


$$
\alpha = \max_sD_{TV}(\pi_{old}(\cdot|s)||\pi(\cdot|s))
$$

$$
\epsilon = \max_{s, a}|A_{\pi_{old}}(s, a)|
$$



원래 식에서 $\alpha$는 EWMA의 가중 평균 정도를 결정하는 하이퍼파라미터였습니다. 여기서는 $\alpha$의 값을 Old Policy와 New Policy 분포 사이의 최대 총 분산 거리(Total Variation Distance, $D_{TV}$), 다시 말하면, 분포의 총 분산 거리가 최대가 되게 하는 $s$를 넣었을 때의 그 거리로 설정합니다. 그리고 $\epsilon$도 마찬가지로 위와 같이 모든 상태 행동 쌍에서의 최대 값으로 설정합니다.


$$
D_{TV}(P, Q) = \frac{1}{2}||P - Q||_1
$$

$$
D_{KL}(P||Q) = \int_xp(x)\log\frac{p(x)}{q(x)}dx
$$

$$
D_{KL}^{max}(\pi_{old}, \pi) = \max_s D_{KL}(\pi_{old}(\cdot|s)||\pi(\cdot|s))
$$



DTV와 DKL 사이에는 위와 같은 관계식이 성립합니다. 이 사실을 이용해 정리하면 아래와 같습니다.


$$
\eta(\pi) \geq L_{\pi_{old}}(\pi) - CD^{max}_{KL}(\pi_{old}, \pi)
$$

$$
where \space C = \frac{4\epsilon\gamma}{(1 - \gamma)^2}
$$



이제 짬뽕 정책을 사용하지 않아도 하한을 알 수 있습니다.







## MM(Minorization-Maximization) Algorithm

<img src="/images/2025-01-20-Reinforcement_Algorithm_TRPO/image-20250121133843940.png" alt="image-20250121133843940" style="zoom:50%;" />

위 그림에서 MM 알고리즘의 동작을 한 눈에 확인할 수 있습니다. 위 그림에서 빨간색은 원래(Orginal) 목적 함수, 그리고 파란색이 대리(Surrogate) 목적 함수입니다. 예를 들어 현재 위치가 $\pi_i$일 때, 아래의 조건을 만족하는 대리 목적 함수를 설정합니다.

1. 대리 목적 함수는 현재 지점에서 원래 목적 함수와 접해야 합니다(값이 같고 기울기가 같음).
2. 대리 목적 함수는 항상 원래 목적 함수보다 작은 값을 가져야 합니다.
3. 대리 목적 함수는 원래 목적 함수보다 최적화하기 쉬운 함수를 사용합니다.



<img src="/images/2025-01-20-Reinforcement_Algorithm_TRPO/image-20250121141005499.png" alt="image-20250121141005499" style="zoom:50%;" />

위의 조건들을 만족하는 대리 목적 함수를 구했다면, 최적화를 수행합니다.

1. 대리 목적 함수에서 최적 지점을 찾습니다.
2. 원래 목적 함수에서 위에서 찾은 최적 지점을 찾아갑니다.
3. 새롭게 찾은 최적 지점에서 다시 대리 목적 함수를 설정하고 동일한 작업을 반복합니다.







## MM 알고리즘 적용

<img src="/images/2025-01-20-Reinforcement_Algorithm_TRPO/image-20250121144048816.png" alt="image-20250121144048816" style="zoom:50%;" />

General Stochastic Policy에서 구한 하한은 MM 알고리즘의 대리 목적 함수의 조건에 완벽하게 부합합니다. 따라서 MM 알고리즘을 적용시키면 단조 성능 향상을 보장할 수 있습니다.


$$
\eta(\theta) \geq L_{\theta_{old}}(\theta) - CD^{max}_{KL}(\theta_{old}, \theta)
$$

$$
\max_{\theta}[L_{\theta_{old}}({\theta}) - CD_{KL}^{max}(\theta_{old}, \theta)]
$$



위의 두 식을 비교를 해보겠습니다. $\eta(\theta) \geq L_{\theta_{old}}(\theta) - CD^{max}_{KL}(\theta_{old}, \theta)$에서 $\theta$를 찾으면, 그건 바로 최적 정책이 될 텐데요, 아래의 $\max_{\theta}[L_{\theta_{old}}({\theta}) - CD_{KL}^{max}(\theta_{old}, \theta)]$식에서 구한 $\theta$는 그저 다음 정책에 불과합니다. 이 과정을 계속 반복해야 위의 최적 정책을 얻을 수 있습니다. 여기에서 꽤나 많은 계산량을 필요로합니다. 그리고 $C$값은 $\gamma$값이 커질 때 같이 커지게 되는데요, 그런 경우에 굉장히 작은 보폭으로 최적화가 일어나기 때문에 시간이 많이 소요됩니다.







## Trust Region Constraint











