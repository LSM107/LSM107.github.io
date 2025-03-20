---
layout: single

title:  "HumanMimic: Learning Natural Locomotion and Transitions for Humanoid Robot via Wasserstein Adversarial Imitation"

categories: Paper

tag: [Robotics, Humanoid]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
published: true
---





이 포스팅은 '**HumanMimic: Learning Natural Locomotion and Transitions for Humanoid Robot via Wasserstein Adversarial Imitation**'에 대한 내용을 담고 있습니다.



논문 주소: <https://arxiv.org/abs/2309.14225>









# HumanMimic: Learning Natural Locomotion and Transitions for Humanoid Robot via Wasserstein Adversarial Imitation

<img src="/images/2025-03-19-Paper_HumanMimic/image-20250320161002832.png" alt="image-20250320161002832" style="zoom:40%;" />

사람은 굉장히 다양한 이동 행동을 수행하고, 속도 변화, 외부 방해에도 적절히 대응하면서 걸음걸이 패턴을 전환할 수 있는 능력이 있습니다. 사람이 보여주는 이러한 자연스러운 걸음걸이 패턴을 로봇에 구현하는 것은 휴머노이드 로봇의 본질적인 복잡성으로 인해 굉장히 어려운 문제입니다. 단순한 모델이나 최적 제어를 통해 이를 구현해보려는 시도가 있었지만, 애초에 액추에이터가 부족한데 그마저도 복잡해서 자연스러운 움직임을 보여주지는 못했습니다. 그 대안으로 심층 강화학습을 통해 모델링 과정을 반자동화해 해결하려는 방향이 최근에 많이 시도되고 있고 꽤나 성공적으로 보여집니다. 그럼에도 복잡한 휴머노이드 로봇에서는 바람직하지 못한 전신 행동, 팔 스윙, 부자연스러운 걸음걸이 등등이 나타납니다. 이런 문제점을 해결하기 위해 휴리스틱을 설정하거나 정교한 보상 설계를 통해 자연스러운 걸음걸이를 학습하도록하는 시도들이 있었지만, 사람의 자연스러운 걸음걸이 패턴을 표현하는 것 부터가 어려운 일입니다.

위의 문제점들을 해결하기 위해 특별한 보상 함수를 설계하지 않고, 인간 시연자의 걸음걸이 데이터를 제공해서 이를 따라하도록 학습시키는 방법론이 사용됩니다. 구체적으로는 **판별기(Discriminator)**를 생성해서 현재 로봇의 행동과 인간 시연자의 걸음걸이와 비교 구분시키고, 이를 보상으로 제공하는 **AMP(Adversarial Motion Prior)** 방법론이 있습니다. 그런데 AMP에서 사용하는 판별기는 BCE Loss 혹은 LS Loss로 학습되는데요, 고차원 공간에서 확률 분포가 겹치지 않는 경우에는 위 손실함수를 사용했을 때 학습이 제대로 수행되지 못하고 붕괴되는 경우가 더러 발생합니다. 특히 현재 논문에서 다루는 휴머노이드 로봇은 굉장히 많은 관절으로 구성되어 있기 때문에, 더더욱 확률 분포가 겹치지 않는 경우가 많습니다. 이 문제를 해결하기 위해서 논문에서는 **Wasserstein Distance**를 도입해 두 분포 사이의 거리를 보다 적절하게 측정할 수 있도록 해줍니다.







## Motion Retargeting

<img src="/images/2025-03-19-Paper_HumanMimic/image-20250320161512328.png" alt="image-20250320161512328" style="zoom:40%;" />

모델 안에 선언된 판별기는 인간의 걸음걸이 데이터와 이 논문에서 사용한 JAXON 로봇의 걸음걸이를 비교해서, 인간의 걸음걸이인지, 로봇의 걸음걸이인지를 분류합니다. 그런데 일단 비교를 하려면 동일한 형태로 데이터로 맞춰줘야 합니다. 따라서 인간 걸음걸이 데이터와 JAXON의 형태를 Primitive Skeleton으로 추상화시킨 다음 두 데이터를 동일한 형태로 정렬시켜줍니다.


$$
L' = \{l'_{ij}\}
$$

$$
M'_s = \{m'_t \mid t \in \{1, ..., T\} \}
$$

$$
m'_t = ({^w}P'_r, {^w}R'_r, {^0}R'_1,..., {^{j-1} }R'_j)
$$



MoCap 데이터는 링크들의 길이, 시간에 따른 움직임을 나타내는 변환 행렬으로 구성되어 있습니다. 루트 조인트 기준으로 해서 계층적으로 퍼져나가면서 구할 수 있도록 주어지는데, 위의 정보들을 사용해서 모든 키 조인트들의 월드 좌표계에 대한 동차 변환 행렬을 구할 수 있습니다.


$$
S = \{s_k \mid k \in \{1, ..., n\}\}
$$

$$
\overrightarrow r'_k = {^w}P'_k - {^w}P'_{k-1}
$$

$$
\overrightarrow r_k = s_k \cdot \overrightarrow r'_k
$$



MoCap 데이터의 링크 길이들과 JAXON 로봇의 링크 길이 비율 차이를 보완해주면, MoCap 데이터의 움직임을 JAXON의 형태에 맞춰줄 수 있게 됩니다.


$$
C_1 = \sum_k\left\| {^r} P_k - p_k(\theta) \right\|^2 \quad \text{position goal for key joints}
$$

$$
C_2 = \sum_e\left\| {^r} P_e - p_e(\theta) \right\|^2 \quad \text{pose goal for end-effectors}
$$

$$
C_3 = \left\| \theta_t - \theta_{t-1} \right\|^2 \quad \text{minimal displacement goal}
$$


$$
C = \text{argmin}_\theta \sum_i \kappa_i C_i(\theta)
$$


로봇의 움직임과 MoCap 데이터의 움직임을 비교할 때, 각 관절의 좌표와 방위를 보고 판별하지 않고 로봇의 관절값을 보고 바로 구분을 시킬 것이기 때문에 역기구학을 통해 로봇의 관절값을 찾아줍니다. 이때 기하학적으로 역기구학을 계산하지 않고 위의 세 가지 목적함수를 사용해 기울기 기반의 최적화 방법을 사용해 구합니다. 


$$
\theta_{\text{min}} \leq \theta_t \leq \theta_{\text{max}}
$$

$$
\dot \theta_{\text{min}} \leq 
\frac{\theta_t - \theta_{t-1}}{\Delta t} 
\leq\dot \theta_{\text{max}}
$$



이때 위의 제약사항들을 고려해 관절 위치와 관절 속도의 한계를 고려합니다. 







## Wasserstein Adversarial Imitation

<img src="/images/2025-03-19-Paper_HumanMimic/image-20250320164452193.png" alt="image-20250320164452193" style="zoom:50%;" />

위는 Wasserstein Adversarial Imitation Learning Framework의 전체 도식입니다. 액터 크리틱 네트워크와 Wasserstein critic, 그리고 직접적인 제어는 PD 컨트롤러로 수행됩니다.





### Velocity-Conditioned Reinforcement Learning



























