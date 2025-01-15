---
layout: single

title:  "강화학습 알고리즘: REINFORCE"

categories: RL Algorithm

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
  

$34$



