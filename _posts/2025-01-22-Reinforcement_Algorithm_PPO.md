---
layout: single

title:  "강화학습 알고리즘: PPO(Proximal Policy Optimization)"

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









# PPO(Proximal Policy Optimization)

TRPO는 신뢰 구역이라는 아이디어를 통해 높은 성능을 달성할 수 있었지만, 구현 난이도가 너무 높고 계산량이 너무 크다는 단점이 있었습니다. 


$$
\max_{\theta} L^{TRPO}(\theta) =  E[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}A_{\theta_{old}}(s, a)]
\space\space\space   subject \space to \space \space\space  
E_{s\sim\rho_{\theta_{old}}}[D_{KL}(\pi_{old}(\cdot | s) || \pi(\cdot | s))] \leq \delta
$$


계산량과 난이도의 원인은 TRPO에서 제시했던 신뢰 구역에 있습니다. 신뢰 구역을 의미하는 제약식의 KL Divergence를 구할 때 Hessian 행렬을 사용하기 때문에 큰 계산량을 필요로 합니다. 그런데 신뢰 구역이 알고리즘에게 주는 이득이 무엇인지 생각해보면, 신뢰 구역은 결국 현재 정책과 다음 정책 사이의 괴리가 커지지 않도록 제한하는 역할을 합니다. 만약에 위의 식에서 제약식이 없었다고 한다면, 평균적으로 좋은 행동(Advantage가 큰 행동)을 선택할 확률을 무조건적으로 높이는 방향으로 업데이트가 수행될 것입니다. 이렇게 업데이트를 하면 목적함수의 값은 올라가겠지만, 결과적으로는 좋지 못한 정책에 도달할 가능성이 높기 때문에 이를 방지하기 위해 신뢰 구역을 TRPO에서 도입했습니다.

그런데 신뢰 구역이 현재 정책과 새로운 정책 사이 분포의 차이를 제한하는 역할을 수행하는 거라면, 그걸 꼭 엄청난 계산량을 필요로 하는 KL-Divergence를 사용할 필요는 없어보입니다. 단순히 Clipping을 통해서도 현재 정책과 새로운 정책 사이의 분포 변화를 조절할 수 있을 것입니다. 이렇게 TRPO에서 제약식을 사용하지 않고 좀 더 단순한 방식으로 현재 정책과 새로운 정책 사이의 분포 변화를 조절한 알고리즘이 바로 **PPO(Proximal Policy Optimization)**입니다.







## Clipped Surrogate Objective Function


$$
\max_\theta L^{CLIP}(\theta) = \max_{\theta} E[\min (r(\theta)\space A_{\theta_{old}}(s, a), \space clip(r(\theta),\space 1-\epsilon, \space1 + \epsilon)\space A_{\theta_{old}}(s, a))]
$$

$$
r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}
$$



$r(\theta)$는 현재 정책과 다음 정책 사이의 차이라고 생각할 수 있는데, 이 값의 크기를 제약식이 아니라 $clip$ 함수를 사용해 조절합니다. 그런데 TRPO의 목적함수인 $\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}A_{\theta_{old}}(s, a)$가 그냥 정해진게 아니라 $\eta(\theta)$의 하한이 되도록 정해진 것이었는데, 위의 $L^{CLIP}(\theta)$ 도 하한을 만족하는지 확인할 필요가 있습니다. 















