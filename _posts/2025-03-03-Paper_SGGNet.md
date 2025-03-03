---
layout: single

title:  "SGGNet$^2$: Speech-Scene Graph Grounding Network
for Speech-guided Navigation"

categories: Paper

tag: [Robotics, Contact, MCP]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
published: true

---





이 포스팅은 '**SGGNet$^2$: Speech-Scene Graph Grounding Network
for Speech-guided Navigation**'에 대한 내용을 담고 있습니다.



논문 주소: <https://arxiv.org/pdf/2402.01183>









# SGGNet$^2$: Speech-Scene Graph Grounding Network for Speech-Guided Navigation
<img src="/images/2025-03-03-Paper_SGGNet/image-20250303193012544.png" alt="image-20250303193012544" style="zoom:50%;" />

로봇이 음성을 명령어로 바로 받아들일 수 있게 하기 위해 다양한 연구들이 있어 왔습니다. 그간의 연구들에서는 음성을 텍스트로 바꾼 다음, 바뀐 텍스트를 해석해 명령을 수행하는 방법론을 사용했습니다. 그런데 이 방법론에서는 전체 정확도가 음성을 텍스트로 바꾸는 **자동 음성 인식(Automatic Speech Recognition)** 결과에 크게 의존합니다. 만약에 ASR에서 잘못된 변환을 하게 된다면 로봇이 실제 명령과 다른 동작을 수행하게 됩니다. 이런 문제점을 해결하기 위해서, 이 논문에서는 음성을 ASR에서 텍스트로 바꿔서 사용하는 것 이외에 추가로 모델 입력으로 넣어줌으로써 텍스트 변환에서 오류가 발생했을 때, 이를 바로잡을 수 있도록 해줍니다.







## Problem Formulation


$$
o_t^* = \text{argmax}_{o_t \in \mathcal{O}} P(o_t \mid \Lambda, \mathcal O)
$$

$$
= \text{argmax}_{o_t \in \mathcal{O}} P(o_t \mid Z, \Lambda, \Upsilon, \mathcal O) \, P(Z \mid \Lambda, \Upsilon, \mathcal O) \, P(\Upsilon \mid \Lambda, \mathcal O)
$$

$$
= \text{argmax}_{o_t \in \mathcal{O}}
\underbrace{P(o_t \mid Z, \Upsilon)}_{\text {NLG}}
\underbrace{P(Z \mid \Lambda) }_{\text{ASR}}
\underbrace{P(\Upsilon \mid \mathcal O)}_{\text{World-Model Generator}}
$$


$$
\Lambda: \text{human's speech command}
$$

$$
\Upsilon: \text{world model}
$$

$$
o_i \in \mathcal O: \text{set of detectable objects}
$$

$$
\varrho: \text{detector}
$$



기존의 language-grounding framework는 위와 같습니다. 로봇의 목적이 물체들과 음성 명령이 주어져 있을 때, 가장 확률이 높은 물체를 고르는 일이라고 한다면, 로봇은 먼저 음성 명령을 텍스트로 변환하고, 바뀐 텍스트를 기반으로 가장 확률이 높은 물체를 고르게 됩니다. 때문에 텍스트 변환이 작업에서의 오류가 곧바로 물체 선택 과정에 전달됩니다.


$$
o^*_t = \text{argmax}_{o_t \in \mathcal O}

\underbrace{P(o_t \mid \Lambda, \Upsilon_{\text{sg}})}_{\text {SGGNet}^2}

\underbrace{P(\Upsilon_\text{sg} \mid \mathcal O)}_{\text{Scene-graph generator}}
$$


따라서 이러한 문제점을 해결하기 위해서 SGGNet$^2$에서 바로 음성 신호와 scene graph를 바로 받아 목표 물체를 예측합니다. 모델 내부적으로 텍스트 변환을 하는 부분이 있는데, 텍스트로 바꾼 결과만 사용하지 않고 음성 신호를 함께 모델 예측에 사용해, 텍스트로 잘못 변환되더라도 올바르게 목표 물체를 예측할 수 있도록 해줍니다.







## Speech-Scene Graph Grounding Network

<img src="/images/2025-03-03-Paper_SGGNet/image-20250303202731408.png" alt="image-20250303202731408" style="zoom:50%;" />

SSGNet$^2$의 전체 구조는 위와 같습니다. ASR에서는 음성 명령을 받아 음성 임베딩과 텍스트 임베딩을 반환합니다. ASR은 인코더-디코더 구조로 되어 있는데, 인코더만을 통과해 얻게되는 벡터가 음성 임베딩 벡터가 되고, 디코더까지 통과해 얻게되는 각 단어 토큰들의 특징 벡터가 텍스트 임베딩 벡터가 됩니다.











