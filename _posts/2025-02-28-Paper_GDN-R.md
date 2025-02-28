---
layout: single

title:  "Graph-based 3D Collision-distance Extimation Network with Probabilistic Graph Rewiring"

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





이 포스팅은 '**Graph-based 3D Collision-distance Extimation Network with Probabilistic Graph Rewiring**'에 대한 내용을 담고 있습니다.



논문 주소: <https://arxiv.org/pdf/2310.04044>









# **Graph-based 3D Collision-distance Extimation Network with Probabilistic Graph Rewiring**

로봇에게 충돌을 감지할 수 있는 능력은 사고 방지를 위해 매우 중요합니다. 이 논문에서는 두 물체 사이의 최소거리를 추정해 충돌을 방지하는 방법론을 소개합니다. 두 물체 사이의 최소거리를 추정하는 가장 전통적인 iterative한 방법으로는 **GJK(Gilbert-Johnson-Keerthi)**가 있습니다. Convex한 두 물체 사이의 거리를 추정할 때 매우 유용한 도구이긴 하지만, 복잡한 기하학적 물체 사이의 거리를 추정할 때 적용하기 어렵다는 문제점이 있습니다. 다른 방법으로 **Data-driven distance estimation** 방법론이 있습니다. 이 방법론에서는 다양한 학습 및 최적화 방법론들을 사용해 두 물체 사이의 거리를 추정하는데, 가장 전형적으로는 point cloud를 사용합니다. 그런데 단순히 점을 기반으로 표면을 표현하면 convex하지 않은 물체에 대해서 부정확한 거리를 추정할 수 있고, 새로운 환경에 대한 일반화 능력도 떨어진다는 단점이 있습니다. 또 다른 방법으로 **SDFs(Signed Distance Fields)**를 사용하는 접근이 있지만, 역시 일반화 능력이 떨어지고 학습할 때 비용이 굉장히 높다는 단점이 있습니다.

이러한 기존의 방법론들의 문제점들을 해결하기 위해, 이 논문의 저자는 이전에 [GNN 기반의 해결책](https://www.semanticscholar.org/reader/182ba39f1fb1cdcec0b23e0f0ab412b96b350434)(**GDN**)을 제시했습니다. 두 물체마다 각각 표면을 따라 그래프를 만들고 이를 사용해 두 물체 사이의 거리를 어텐션 매커니즘을 통해 추정했는데요, 그래프를 사용해 물체의 거리를 추정하는 방법 자체가 계산량을 많이 필요로 해서, 2D 이상으로는 적용하기 어렵다는 문제점이 있었습니다. 

이 문제를 해결하기 위해서 기존의 방법론을 보완한 **GDN-R(Rewiring)**을 제안합니다. 두 그래프를 입력으로 받은 다음 여러 **MPNN(Message-Passing Neural Network)**를 통과시킨 뒤, 가장 높은 연결성을 갖는 노드들을 선정하는 **Rewiring** 단계를 거칩니다. 마지막으로 Gumble noise로 무작위성을 추가해 다양한 물체에 대한 일반화 능력을 갖출 수 있게 합니다. 







## Background



 





















