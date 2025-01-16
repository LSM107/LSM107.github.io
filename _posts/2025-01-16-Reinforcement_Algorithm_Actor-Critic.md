---
layout: single

title:  "강화학습 알고리즘: Actor Critic"

categories: RL_Algorithm

tag: [Reinforcement Learning, Policy Gradient, Actor Critic]

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



자료 출처: 단단한 강화학습, Reinforcement Learning An Introduction , 2nd edition. 리처드 서튼, 앤드류 바트로, 김성우(옮긴이)









# Actor Critic

Monte-Carlo Policy Gradient(REINFORCE) 알고리즘에서 이득의 분산을 줄이기 위한 여러 기술적인 요소들이 있었음에도, 여전히 꽤나 높은 분산을 가집니다.











