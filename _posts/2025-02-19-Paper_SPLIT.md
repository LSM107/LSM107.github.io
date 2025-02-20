---
layout: single

title:  "SPLIT: SE(3)-diffusion via Local Geometiry-based Score Prediction for 3D Scene-to-Pose-Set Matching Problems"

categories: Paper

tag: [Robotics, Collision]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
published: True
---





이 포스팅은 '**SPLIT: SE(3)-diffusion via Local Geometiry-based Score Prediction for 3D Scene-to-Pose-Set Matching Problems**'에 대한 내용을 담고 있습니다.



논문 주소: <https://arxiv.org/pdf/2411.10049>









# SPLIT: SE(3)-diffusion via Local Geometiry-based Score Prediction for 3D Scene-to-Pose-Set Matching Problems

특정 태스크를 수행할 때,  말단부의 포즈를 추론하는 일은 항상 중요하면서도 잘 해결되지 않는 문제입니다. 가장 전통적인 방법에서는 특정 태스크를 수행하기 위한 모든 세부적인 알고리즘을 설정하고, 이 알고리즘들의 조합으로 태스크를 해결했습니다. 당연히 잘 동작하지 않았고요, 여러 다양한 태스크들에 일반적으로 사용될 수 없다는 문제점이 있습니다. 

요즘 많이 사용되는 학습 기반 방법론으로 접근해도 적절한 포즈를 추론하는 일은 쉽게 해결되지 않습니다. 학습 기반 방식에서는 먼저 물체를 이미지에서 찾고, 찾은 물체에서 들기 쉬워보이는 부분들인 Keypoints를 검출합니다. 그리고 이 Keypoints에 적절한 포즈를 추론하는데요, 이 Keypoints가 제대로 찾아지지 않으면 그 다음 단계인 포즈 검출도 이상하게 되기 때문에 Keypoints를 잘 찾는게 굉장히 중요합니다. 그런데 문제는 이게 잘 안됩니다. 때문에 최근에는 Keypoints를 찾지 않고, 바로 포즈를 추론하는 접근법이 더 선호됩니다.

이 논문에서는 다양한 태스크에 모두 적용될 수 있도록 **SPLIT(Score Prediction with Local Geometry Information for Rigid Transformation)**이라는 새로운 모델을 제안합니다. 이 SPLIT에서는 **Score-based Generative Model**을 사용해, 랜덤한 노이즈로부터 점점 태스크 성공 확률이 높은 포즈를 검출해 나갑니다. 구체적으로 이 모델이 어떻게 구성돼 있는지 아래에서 살펴봅니다.







## Introduction

전통적인 방법들은 다양한 태스크를 범용적으로 수행하는데 큰 어려움이 있습니다. 왜냐하면 개별적인 알고리즘들을 조합하는 방식으로 태스크를 해결했기 때문입니다. 일반적으로 잘 동작하게 만들기 위해서는 이전과는 다른 접근이 필요합니다.

이 논문에서는 *3D scene-to-pose-set matching* 이라는 새로운 문제를 정의합니다. 이 문제 정의에서는 기하학적인 특징은 물체의 전역적인 형태가 아니라,  국소적인 형태에 의해 결정된다고 가정하고 이를 **Spatial Locality** 라고 부릅니다. 예를 들어, 우리가 컵을 들어올리는 태스크를 수행한다고 할 때, 로봇이 컵을 잡을 특정한 부분의 모양만 알면 되지, 컵 전체의 모양이 크게 중요하지는 않습니다. 그리고 이런 접근 방식, Spatial Locality만을 보고 포즈를 결정하는 방식은 사전 지식을 필요로 하지 않는, 좀 더 유연한 프레임워크를 제공합니다. 







## Preliminaries

SPLIT 모델에서 사용되는 배경 지식을 먼저 살펴봅니다.





### Score-based Generative Models

이 논문에서는 성공 확률이 높은 포즈를 추론하기 위해 **SGM(Score-based Generative Model)**을 사용합니다. SGM도 랜덤 노이즈를 시작으로 점차 성공 확률이 높은 포즈를 찾아가는 방식으로 반복적인 샘플링이 수행하는 일종의 확산 모델입니다. 


$$
\mathcal{L}_{\mathrm{DSM}} 
= \frac{1}{L} \sum_{k=0}^{L} 
\mathbb{E}_{x, \tilde{x}} 
\Bigl[
\bigl\|
s_{\theta}(\tilde{x})
- \nabla_{\tilde{x}}
\log q_{\sigma}(\tilde{x} \mid x)
\bigr\|_2^2
\Bigr]
$$

$$
q_{\sigma}(\tilde{x} \mid x) = \mathcal{N}(\tilde x ; x, \sigma^2I)
$$

$$
x_{k-1}
= x_k
+ \frac{\alpha_k}{2}s_{\theta}\bigl(x_k, \sigma_k\bigr)
+ \sqrt{\alpha_k}\,\epsilon,
\quad \epsilon \sim \mathcal{N}(0, I). \quad \text{(Langevin Dynamics)}
$$



손실함수에서 알 수 있는 것은, 이 모델에서 학습하는 것은 '**노이즈가 어떤 방향으로 추가되었는가?**' 입니다. 정확하게는 원레 데이터로 복원하도록 하는 반대방향을 학습합니다. 기울기에 대해 제대로 학습이 됐다면, **랭주뱅 동역학(Langegin Dynamics)**을 통해 업데이트를 수행합니다. 



<img src="https://blog.kakaocdn.net/dn/6gkAi/btsge6cduQo/jNbjcW1g6dbSPuYc9VPkP0/img.gif" alt="img" style="zoom:50%;" />

- <https://dlaiml.tistory.com/entry/Score-based-Generative-Models과-Diffusion-Probabilistic-Models과의-관계>



위 도식은 랭주뱅 동역학이 일어나는 과정을 직관적으로 보여줍니다. 시간이 지남에 따라 모이는 점들이 물체를 잡을 성공 확률이 높은 지점이 됩니다. 풀리지 않는 의문은 **물체를 잡을 성공 확률이 높은 지점으로 이동하게 하는 기울기를 어떻게 구하는가** 입니다. 말단부의 포즈는 4 by 4의 동차행렬로 표현되는데, 이 동차행렬의 공간에서 성공 확률이 높은 포즈로의 이동 방향, 기울기를 형식적으로 나타낼 수 있어야 합니다.





### SE(3) Lie Group

로보틱스에서 물체의 자세는 4 by 4의 동차행렬로 표현될 수 있습니다. 이 동차행렬은 수학적으로 모든 부분에서 미분 가능한 manifold로 **SE(3)**이라고도 표현됩니다. 특정 자세 **H**는 이 manifold 상의 공간을 돌아다닐 텐데요, 특정 자세 H에서 SE(3)에 대한 **접공간(Tangent Space)**을 아래와 같이 표현합니다.


$$
T_HSE(3) \in \mathbb R^6
$$


그리고 특히 이동을 하지 않는, 단위 행렬인 동차 행렬 $\varepsilon$에서의 접공간을 아래와 같이 표현합니다.


$$
\mathfrak {se}(3):= T_\varepsilon SE(3) \in \mathbb R^6
$$


단위 행렬의 접공간과 동차행렬 공간 사이의 mapping을 아래와 같이 정의할 수 있고, 이 논문에서는 이를 **Global Parameterization** 이라고 합니다.


$$
\text{Exp}: \mathbb R^6 \rightarrow SE(3)
$$

$$
\text{Log}: SE(3)\rightarrow \mathbb R^6
$$



비슷하게 **Local Parmeterization**을 정의할 수 있습니다. Global Parameterization이 단위 행렬에서의 접공간 mapping한 거라면, Local Parameterization은 동차행렬 H의 접공간에 mapping 시킵니다. 그리고 동차행렬 공간에서 H와 단위 행렬 사이의 변환을 할 수 있기 때문에, Local Parameterization은 아래와 같이 Global Parameterization을 사용해 표현될 수 있습니다.


$$
{^H}\text{Exp}(h') := H\text{Exp}(h')
$$

$$
{^H}\text{Log}(H') := \text{Log}(H^{-1}H')
$$

$$
where \quad h' = {^H}\text{Log}(H')
$$



동차행렬 공간에서의 가우시안 분포를 어떻게 정의를 해야합니다. 왜냐하면 동차행렬로 주어진 한 점에 계속해서 가우시안 노이즈를 더해나갈 것이기 때문이죠. 그런데 위와 같이 동차행렬 공간과 6차원 실수공간 사이의 관계를 정의하고 나면, 동차행렬 공간에서의 가우시안 분포를 정의할 수 있습니다. 


$$
q\bigl(H \mid \overline{H}, \Sigma\bigr)
\;\propto\;
\exp\!\Bigl(
   -0.5 \,\bigl\|{^\overline{H} }\,\mathrm{Log}(H)\bigr\|_{\Sigma^{-1}}^2
\Bigr).
$$

$$
q_\sigma(H \mid \bar H) \quad \text{where} \quad \Sigma = \sigma I
$$



주된 아이디어는 동차행렬 공간에서 바로 가우시안 분포를 정의하는게 아니라, **'동차행렬을 실수 공간으로 옮긴 뒤, 실수공간에서 가우시안 분포를 적용시켜 다시 동차행렬 공간으로 옮긴다'** 입니다. 




$$
{^H}\nabla f(H') =
\frac{ {^H}D f(H)}{DH} \big|_{H=H'} = 
\frac{ \partial  f(h)}{\partial h} \big|_{h=h'}
$$

$$
h'
= {^H} \mathrm{Log}(H').
$$



그리고 위와 같이 동차행렬 공간의 특정 포즈가 입력으로 들어가는 함수의 기울기도 구할 수 있습니다.







## Method

위의 Preliminaries에서 다룬 내용들을 바탕으로 **SPLIT** 모델을 제안합니다.





### 3D Scene-to-pose-set matching: Towards a Generalizable Pose Detection Formulation

먼저 3D Scene-to-pose-set matching 이라는 새로운 문제를 정의합니다. 이 문제에는 중요한 가정 두 가지가 있는데, 아래와 같습니다.

- 목표 포즈는 장면의 특정 부분, 국부 기하 정보와만 관계가 있다(=**Local Geometric Context**).

- 동일한 국부 기하 정보는 동일한 포즈를 유도한다.


$$
\mathcal F_\phi(H, X) = {^H}z
$$


























