---
layout: single

title:  "A Reachability Tree-Based Algorithm for Robot Task and Motion Planning"

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





이 포스팅은 '**A Reachability Tree-Based Algorithm for Robot Task and Motion Planning**'에 대한 내용을 담고 있습니다.



논문 주소: <https://arxiv.org/pdf/2303.03825>









# A Reachability Tree-Based Algorithm for Robot Task and Motion Planning

로봇이 계획부터 실제 수행까지 모든 조작 과정을 자율적으로 수행하게 만들기 위해 다양한 선행 연구들이 있어왔습니다. 로봇이 조작 과정 전체를 자율적으로 수행하기 위해서는 **TP(Task Planning)**과 **MP(Motion Planning)**을 수행할 수 있어야 합니다. 이 두 가지 계획을 모두 해결해야 하는 문제를 **TAMP**문제라고 합니다.**Task Planning**이란 태스크 수준에서의 계획 수립을 의미하고, Motion Planning은 그 각각의 개별 태스크를 수행하기 위해 구체적인 제어 계획을 의미합니다. 태스크 수준에서의 계획은 추상적인 수준 수립되기 때문에, 실제로 기하학적인 관계를 고려하기 어렵습니다. 주방에서 요리를 하는 예시를 들어보면, Task Planning 단계에서 `세척` 이후 `조리` 라는 계획을 수립할 수 있습니다(추상적인 단계에서 수립됨). 그런데 만약 세척을 수행해야는 싱크대와 조리를 수행해야 하는 스토브 사이에 이동을 어렵게 하는 장애물이 있는 경우, Motion Planning이 불가능할 수 있습니다. 중간에 거쳐갈 수 있는 도마가 있다면, `세척`을 하고, `도마로 이동`시키고,  마지막으로  `조리`를 수행하는 경우에만 Motion Planning이 가능한 경우가 있을 수 있다는 거죠. 다시 말하면, Task Planning을 할 때 이러한 기하적인 관계까지를 고려해야 올바른 계획을 수립할 수 있습니다. 이걸 고려하지 않으면 Motion Planning 단계에서 불가능한 계획 판단하기 위해 많은 컴퓨팅 자원을 쓴 후, 다시 Task Planning을 수행하는 과정을 반복해야 하기 때문에 많은 컴퓨팅 자원을 소모하게 됩니다.

이러한 문제점을 해결하기 위해서 **RRT(Rapid Exploring Reachability Tree)**와 비슷한 방법을 사용하면서도, Task Planning까지 수행할 수 있도록 하는 계층적 전략을 취하는 새로운 계획 모델을 제시합니다. 







## Related Work

옮겨야 하는 물체가 존재하는 문제에서는 전형적인 MP로 해결하기가 어렵습니다. 이런 문제들을 해결하기 위해 **모드(*mode*)**라는 개념이 사용됩니다. 모드란 로봇이 충돌 없이 움직일 수 있는 자유 공간을 의미합니다. 물체를 옮길 때마다 로봇팔의 모드는 조금씩 변화하게 되는데요, 모드들 사이의 이동으로 계획을 수립하는 문제를 **MMMP(Multi-Modal Motion Planning)**이라고 합니다. 가장 잘 알려진 접근 방법은 **RRT-like MMMP**으로, 새로운 모드로의 이동을 트리의 간선으로 삼아 문제를 해결합니다. **TAMP**도 마찬가지로 조작 계획 문제를 다루는데, 여기서는 Symbolic Planner를 사용해 이산화된 상태 공간을 탐색합니다. TAMP의 경우 추상적인 상태와 행동에 대해 계획을 수립할 수 있다는 장점이 있습니다. 이 논문에서는 위의 모든 아이디어를 종합해 조작 계획 문제를 해결합니다.







## Manipulation Planning Problem

TAMP 문제를 해결하기 위해 MP, MMMP, TP 방법론들을 모두 모아 다 같이 사용합니다. 각각의 아이디어들이 어떻게 사용되는지 아래에서 자세하게 살펴봅니다.





### Problem Definition


$$
(x_I, \mathcal{X}_G) \quad x_I \in \mathcal X(\text{initial state}) \quad \mathcal X_G: \text{set of goal states}
$$


조작 계획 문제는 위와 같이 튜플로 정의할 수 있습니다. 하나로 결정된 시작 상태와, 목표 상태의 집합을 두 원소로 가지는 튜플로 정의됩니다. 이 문제에서 얻어야 하는 건, 어떻게 목표 상태 집합으로 이동할 수 있는지, 그 행동 나열을 찾아내는 것입니다. 목표 상태 집합은 특정 변수에 대한 수식으로 명세될 수 있습니다. 예를 들어 어떤 블록을 특정 위치에 이동시켜야 하는 태스크라면, 해당 블록이 x축으로 어떤 범위, y축으로 어떤 범위 안에 들어감을 표현하는 수식으로 목표가 표현될 수 있습니다.





### Notation

다음으로 문제에서 사용하는 세부적인 표기법들에 대해서 살펴보겠습니다.


$$
o :\text{object}\in \mathcal O \quad m :\text{movable object}\quad r:\text{robot}
$$

$$
\mathcal M \subset \mathcal O \quad r \in \mathcal O
$$



TP에서 object 집합은 movable object와 robot을 다 포함합니다.


$$
\alpha_m = (m, p, {^m}T_P): \text{attachment}
$$

$$
\text{where} \quad m \in \mathcal M, \space p \in \mathcal O, \space {^m}T_p \in SE(3)
$$



위의 튜플을 **attachment**라고 합니다. attachment는 물체 사이의 접촉 관계를 표현하는데, 위의 표현은 <u>m이 p 위에 붙어있음</u>을 의미합니다. 그리고 가장 마지막 원소는 두 물체 사이의 변환 행렬이 됩니다. 그러니까 attachment는 두 물체 사이의 추상적인 관계(누가 누구한테 붙어있음)와 구체적인 위치 관계(변환 행렬)까지의 모든 정보를 포함합니다.


$$
\sigma:\text{mode} = \{\alpha_m \mid m \in \mathcal M\} \in \Sigma :\text{infinite set of modes}
$$


위에서 언급한 **mode**는 위와 같이 정의됩니다. 움직일 수 있는 모든 물체의 attachments를 원소로 하는 집합이 mode입니다. mode가 결정되면 로봇이 충돌 없이 움직일 수 있는 공간도 따라서 결정되는데 이를 $\mathcal C^\sigma_{free}$라고 표현합니다.























