---
layout: single

title:  "강화학습 07: n단계 부트스트랩"

categories: RL

tag: [Reinforcement Learning, TD]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
published: true
---





이 포스팅은 '**강화학습**'에 대한 내용을 담고 있습니다.



자료 출처: 단단한 강화학습, Reinforcement Learning An Introduction , 2nd edition. 리처드 서튼, 앤드류 바트로, 김성우(옮긴이)









# n단계 부트스트랩

n단계 부트스랩이라고 나와있지만, 사실 n단계 TD 방법에 대해서 살펴봅니다. 앞서 다루었던 TD 방법과 MC를 통합하는 방식이기도 한데요, 이 방식에서는 보다 더 자유롭게 갱신에 사용할 타입 스텝의 개수를 결정할 수 있습니다. 







## n단계 TD 예측

MC와 TD 방법의 차이는 갱신에 사용하는 보상의 개수에 있습니다. MC는 이후 모든 보상을 갱신에 사용하는 반면, TD는 다음 첫 보상과 부트스트랩을 갱신에 사용합니다. 그런데, 부트스트랩을 사용하는 경우에도 보상을 한 개만 사용할 필요는 없습니다. 이후 여러 단계의 실제 보상과 그 다음에 부트스트랩을 사용하는 방법도 가능할 것입니다. 예를 들어,  두 단계 갱신은 처음 두 개의 보상과 두 단계 이후의 상태 가치 추정값을 기반으로 하여 갱신합니다. 비슷하게 세 단계, 네 단계 갱신을 생각할 수 있는데요, 이렇게 시간 단계를 n단계로 확장하는 방법을 **n단계 TD 방법**이라고 부릅니다.


$$
G_t = R_{t+1} + \gamma^{} R_{t+2} + \gamma^{2} R_{t+3} + \cdot\cdot\cdot + \gamma^{T-t-1} R_T
$$

$$
G_{t:t+1} = R_{t+1} + \gamma V_t(S_{t+1})
$$

위 식에서 첫 번째 식은 MC의 갱신 목표이고, 그 아래 식은 TD의 갱신 목표입니다. n단계 TD 방법에서는 이를 **단일 단계 이득(One-Step Return)**이라고 부릅니다.


$$
G_{t:t+2} = R_{t+1} +  \gamma^{} R_{t+2} + \gamma^{2} V_t(S_{t+2})
$$
위 식은, 다음 두 단계의 실제 이득을 사용하고 그 다음에 추정값을 사용하는 두 단계 이득**(Two-Step Return)**입니다. 







## n단계 살사

상태-행동 가치 함수를 사용하는 살사에 대해서도 동일하게 적용하면, n단게 살사가 됩니다. 기댓값 살사 역시 크게 다르지 않게 그대로 적용됩니다. 


$$
Q_{t+n}(S_t, A_t) \doteq Q_{t+n-1}(S_t, A_t) + \alpha[G_{t:t+n} - Q_{t+n-1}(S_t, A_t)]
$$







## n단계 비활성 정책 학습

MC에서 비활성 정책을 사용하는 경우가 있었습니다. 이런 경우에 두 정책 사이의 차이점을 고려해야 하고, 이를 위해 중요도추출비율이 사용되었습니다. n단계 비활성 정책에서도 마찬가지로 가치함수를 갱신할 때 이 중요도추출비율을 사용합니다.


$$
\rho_{t:h} \doteq \prod_{k=t}^{min(h, T-1)} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
$$


$$
V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) +\alpha \rho_{t:t+n-1}[G_{t:t+n} - V_{t+n-1}(S_t)]
$$

$$
Q_{t+n}(S_t, A_t) \doteq Q_{t+n-1}(S_t, A_t) +\alpha \rho_{t+1:t+n}[G_{t:t+n} - Q_{t+n-1}(S_t, A_t)]
$$

상태 가치 함수에 대한 갱신 식과 상태-행동 가치 함수에 대한 갱신 식, 두 개가 있습니다. $\rho$의 시간 단계가 상태-행동 가치 함수에서 한 단계 뒤로 밀려있는 것을 확인할 수 있는데요, 이는 상태-행동 가치 함수에서는 이미 행동이 선택된 상태에서 시작하기 때문입니다.







## 중요도추출법을 사용하지 않는 비활성 정책 학습: n단계 트리 보강 알고리즘

중요도추출법을 사용하지 않고도 비활성 정책 학습이 가능할까요? 사실 TD를 다룰 때 다룬 Q 학습과 기댓값 살사는 단일 단계에 대해서 중요추출법을 사용하지 않는 비활성 정책 학습 방법이었습니다. 이 개념을 확장시키면 다단계 경우에도 비활성 정책을 학습이 가능합니다.


$$
G_{t:t+1} \doteq R_{t+1} + \gamma\sum_a\pi(a|S_{t+1})Q_t(S_{t+1}, a)
$$
위 식은 기댓값 살사이면서 단일 단계 이득입니다. 


$$
G_{t:t+2} \doteq R_{t+1} + \gamma\sum_{a\neq A_{t+1}}\pi(a|S_{t+1})Q_t(S_{t+1}, a) +\\ \gamma \pi(A_{t+1}|S_{t+1})(R_{t+2}+ \gamma \sum_a\pi(a|S_{t+2})Q_{t+1}(S_{t+2}, a))
$$

$$
= R_{t+1} + \gamma\sum_{a\neq A_{t+1}}\pi(a|S_{t+1})Q_t(S_{t+1}, a) +\\ \gamma \pi(A_{t+1}|S_{t+1})G_{t+1:t+2}
$$

기댓값 살사를 두 단계로 확장시킨 것인데요, 실제로 에피소드에서 실행한 행동인 경우 얻은 보상이 존재합니다. 이 행동에 대해서는 가치 함수 추정값을 사용하지 않고 실제로 얻은 보상과 그 이후에 선택할 수 있는 행동들 각각의 추정값의 기댓값을 사용합니다. 마찬가지로 세 단계에 대해서도, 네 단계에 대해서도 확장시킬 수 있겠습니다.













