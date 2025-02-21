---
layout: single

title:  "DeepCert: Verification of Contextually Relevant Robustness for Neural Network Image Classifiers"

categories: Paper

tag: [verification, robustness, neural network, adversarial attack]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
---





이 포스팅은 '***DeepCert: Verification of Contextually Relevant Robustness for Neural Network Image Classifiers***' 논문에 대한 소개를 담고 있습니다.



출처: <https://theory.stanford.edu/~barrett/pubs/PWG+21.pdf>









## 배경

신경망을 기반으로 한 이미지 분류 모델의 정확도는 근 몇년간 굉장히 빠른 속도로 상승했고, 심지어 일부 벤치마크에서는 인간 수준의 정확도를 뛰어넘는 성능을 보이기도 합니다. 이렇듯 신경망은 정말 강력한 도구이지만, **적대적 공격(adversarial attack)**에 있어 취약한 모습을 보입니다.



![image-20240215230659921](/images/2024-02-15-DeepCert/image-20240215230659921.png)

- 그림 출처: <https://pytorch.org/tutorials/beginner/fgsm_tutorial.html>

위의 사례에서 신경망은 사람 눈에는 분명 같아보이는 두 이미지를 다른 클래스로 분류합니다. 이런 일이 발생하는 이유는, 사람이 인식하기 힘든 수준의 정말 작은 perturbation을 이미지에 가해졌기 때문입니다. 이미지에 가해진 perturbation은 속이고자 하는 특정 신경망을 목표로 적대적으로 결정됩니다. 그러한 perturbation이 가해져 목표한 신경망을 속이는 예제를 **적대적 예제(adversarial example)**라고 부릅니다.



특정 신경망의 강건성을 적대적 예제를 통해 판단할 수 있습니다. perturbation에 범위를 제한하지 않는다면, perturbation이 가해진 이미지는 어떤 이미지도 될 수 있기 때문에 적대적 예제는 무조건 존재합니다. 따라서 적대적 예제를 이용해서 강건성을 평가할 때, 아래와 같은 기준들을 사용합니다.

1. $0<\epsilon < p$​ 의 범위에서 적대적 예제가 존재하지 않는다면, 모델은 충분히 강건하다.
   - $\epsilon$은 모델에 가해지는 perturbation의 크기입니다.
2. 모델에 대한 적대적 예제들 중, perturbation이 가장 작은 적대적 예제의 그 크기를 통해 모델의 강건성을 평가한다.



적대적 예제에 가해지는 perturbation의 크기는 대개 $L_p-norm$ 으로 측정됩니다. 때문에 적대적 예제는 다소 pixel-wise한 perturbation이 가해져 생성됩니다. 이런 방식으로 생성된 적대적 예제는 ACAS Xu 벤치마크와 같은 상황에서는 의미있는 강건성 검증의 기준이 될 수 있습니다. **하지만 이런 방식으로 생성되는 적대적 예제는 현실 세계에서 발생하는 노이즈를 반영하기 힘들 수 있습니다**.



![img](/images/2024-02-15-DeepCert/1*1FMKI3BuS-ZQFvF4FElxIQ.png)

- 그림 출처: <https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9>

위의 사진은 데이터 증강을 위해 인위적으로 만들어진 예제이지만, 현실 세계에서도 충분히 있을 법한 상황입니다(물체가 카메라를 통해 다양한 각도로 촬영되는 상황). 특히, 위와 같은 수박 사진을 분류한는데 있어서는 회전 각도가 물체의 클래스를 분류하는데 있어 영향을 주지 않아야 합니다. 하지만,  $L_p-norm$​​으로 원본 이미지와의 거리를 측정하면, 그 값이 상당히 크게 나타납니다. 이미지 전체가 이동하는 방식으로 perturbation이 가해지기 때문입니다. **따라서 전통적인 적대적 예제를 통한 강건성 검증은 회전 perturbation에 대한 강건성을 보장할 수 없습니다. 이런 문제는 회전 효과(rotation) 뿐만 아니라 안개 효과(hazing), 대조 효과(contrast), 흐림 효과(blurring)에서도 마찬가지로 나타납니다.**

- 논문에서는 회전 효과에 대해서는 다루지 않습니다.



따라서 이 논문에서는 현실 세계에서 발생하는 노이즈에 대한 강건성을 의미하는 **contextually relevant robustness**에 대한 검증 기법을 제안합니다.





## 핵심 아이디어

이 논문의 main contribution은 아래와 같습니다.

1. contextually relevant한 이미지 perturbation이 $0<\epsilon < 1$ 범위 내에서 가해지며, 이를 formal encoding 합니다.

2. **$\epsilon$ 이 증가함에 따라 DNN 정확도가 얼마나 감소하는가를 통해 DNN의 contextually relevant robustness를 검증합니다.**

   - DeepCert는 이를 test-based, 혹은 formal verification으로 수행합니다.

     

3. contextually relevant counter example을 생성합니다.
4. operational context에 적절한 DNN을 선택합니다.



위 4가지 contribution 중 2번이 이 논문의 핵심 아이디어입니다. 이 논문에서는 contextually relevant robustness를 test based verification과 formal verification, 총 2가지 방법으로 검증합니다.



### test-based verification

![image-20240216100039982](/images/2024-02-15-DeepCert/image-20240216100039982.png)

- 표기법에 대한 자세한 설명은 **논문 2~3페이지의 2.1 Overview**를 참조하세요.



1. A에서 label이 존재하는 이미지에 $\epsilon$​만큼의 Contextual Perturbation을 가합니다. 가해지는 Contextual Perturbation의 종류는 **Haze encoding, Contrast variation encoding, Blur encoding**입니다.

   ![image-20240216112756698](/images/2024-02-15-DeepCert/image-20240216112756698.png)

   1. **Haze encoding**
   
      
      $$
      x_{i,j}' = (1 - \varepsilon)x_{i,j} + \varepsilon C^f
      $$
      

      현실 세계의 안개는 좀 더 복잡한 모델을 필요로 하지만, 이 논문에서는 보다 간단한 안개 모델을 사용합니다. $\epsilon$의 크기가 커질수록 안개 효과가 커집니다.
   
      
   
   2. **Contrast variation encoding**

      
      $$
      x_{i,j}' = \max\left(0, \min\left(1, \frac{x_{i,j} - (0.5 * \varepsilon)}{1 - \varepsilon}\right)\right)
      $$
      
   
      $\epsilon$의 크기가 커질수록 이미지의 대조가 증가합니다.
   
      
   
   3. **Blur encoding**
   
      
      $$
      x_{i,j}' = \sum_{k=-k_d}^{k_d} \sum_{l=-k_d}^{k_d} \alpha_{k,l} \cdot x_{i+k, j+l}
      $$
      
   
      $\alpha_{k,l}$는 window의 중앙 픽셀로부터의 거리에 따라 변화하는 값으로, 가우시안 곡선을 따릅니다. 즉, 가우시안 블러링 필터를 통해 이미지의 흐림 효과를 증가시킵니다. $\epsilon$의 크기가 커질수록 이미지의 흐림 효과가 증가합니다.



2. Contextual Perturbation이 가해진 이미지를 모델에게 분류시킵니다. 모델이 이미지를 잘 분류하면 True, 잘못 분류하면 False입니다.



3. $\epsilon$-Search Heuristic에 따라 $\epsilon$의 범위([$\underline{\epsilon}$, $\overline{\epsilon}$])를 조정합니다.

   - 2단계에서 모델의 분류 결과가 True인 경우 $\underline{\epsilon}$값을 $\epsilon$으로 업데이트합니다.

   - 2단계에서 모델의 분류 결과가 False인 경우 $\overline{\epsilon}$값을 $\epsilon$​​​으로 업데이트합니다.

     

4. $\overline{\epsilon} - \underline{\epsilon}$의 크기가 미리 설정한 $\omega$​​이하가 될 때까지, 1~3단계를 반복합니다.

   - [$\underline{\epsilon}$, $\overline{\epsilon}$​]은 처음에 [0, 1]로 시작합니다. 

     

5. $\epsilon$의 범위($\ r_{i, j}$ = [$\underline{\epsilon}$, $\overline{\epsilon}$​])를 반환합니다.



![image-20240216113540555](/images/2024-02-15-DeepCert/image-20240216113540555.png)

![image-20240216113628985](/images/2024-02-15-DeepCert/image-20240216113628985.png)

위의 단계들을 거쳐 test-based verification을 수행합니다. 각 예제에 대한 counterexample은 $\overline{\epsilon}$의 perturbation을 가함으로써 쉽게 생성할 수 있습니다. 여러 모델들에 대해서 $\epsilon$의 변화에 따른 정확도나, 한 모델에 대한 각 클래스의 정확도 변화를 그래프를 통해 시각화할 수 있습니다.

$\epsilon$​의 크기가 변화함에 따라 분류를 잘 수행하는 모델이 변화할 수 있습니다. 이 논문에서는 각 구간에서 가장 잘하는 모델로 분류를 수행함으로써 정확도를 향상시킬 수 있다고 주장합니다.



### formal verification

test-based verification이 계산상으로 효율적이지만, 완전성(completeness)를 보장하지 못합니다. 이유는 즉슨 애초에 그 효율성이, 완전성을 희생시킨 대가로 얻은 것이기 때문입니다. 따라서, test-based verification을 통해 $\epsilon$값을 $p$로 결정지어도, $p$​​보다 작은 perturbation에 대해 모델이 항상 강건하다고 보장할 수 없습니다. 그러한 보장을 하기 위해서는 formal verification이 필수적이며, DeepCert는 formal verfication을 통한 강건성 검증이 가능합니다.



$$
Y = M(X)
$$

$$
0 \leq\epsilon\leq p
$$

$$
\bigwedge\limits_{i\leq|X|} (x_{i} = (1 - \varepsilon)x_i + \varepsilon C^f)
$$

$$
\bigvee\limits_{\substack{i<=|Y|\\ y_i \neq y_{real}}}
y_i \geq y_{real}
$$

위의 수식들은 Haze encoding에 대한 formal verification의 constraints입니다. 다른 종류의 Contextual Perturbation에 대한 formal verfication을 수행하기 위해선, Haze encoding constraint(위에서 3번째 수식)를 검증하고자 하는 Contextual Perturbation을 표현하는 수식으로 교체해야 합니다. constraints의 satisfiability는 MILP-solver 혹은  Reluplex procedure를 통해 확인할 수 있습니다.

- satisfiable: counterexample이 존재합니다.
- unsatisfiable: $0 \leq\epsilon\leq p$ 범위 내의 counterexample이 존재하지 않으며, locally robust합니다.





## 실험 결과

### Road Traffic Speed Sign Classification

<img src="/images/2024-02-15-DeepCert/output-8187423.png" alt="output" style="zoom:33%;" />

GTSRB는 교통 표지판 인식 알고리즘의 성능을 평가하는 데 사용되는 벤치마크입니다. 총 43개의 클래스로 구성되어 있고, 다양한 조명 조건과 배경에서 촬영된 이미지를 제공합니다. 



<img src="/images/2024-02-15-DeepCert/output-1.png" alt="output-1" style="zoom:25%;" />

이 논문에서는 43개의 클래스 중 제한 속도 표시와 관련된 7개의 클래스만을 사용합니다.

- 하단의 (a)에 해당하는 클래스를 사용합니다.



![image-20240218003828852](/images/2024-02-15-DeepCert/image-20240218003828852.png)

(b)에 해당하는 6개의 모델에 대해 검증을 수행합니다. 검증은 논문에서 제안한 DeepCert를 통해 수행되며, test-based verification과 formal verification, 두 가지 방식을 사용합니다.



#### test-based verification

##### model switching

![image-20240218003804311](/images/2024-02-15-DeepCert/image-20240218003804311.png)

각각의 모델에서의 hazing perturbation의 $\epsilon$ 변화에 따른 정확도를 나타낸 그래프입니다. $\epsilon$구간 별로 분류 정확도의 순서가 변화하는 것을 확인할 수 있습니다. 예를 들어 Model 3a와 Model 3b를 비교할 때, Model 3a가 $0\leq\epsilon\leq0.75$ 구간에서 Model 3b 보다 정확도가  높지만, $0.75\lt\epsilon\leq1$​ 구간에서 Model 3b가 Model 3a의 정확도를 앞섭니다. 각 구간에서 분류 정확도가 더 높은 모델을 선택한다면, perturbation에 대한 robustness를 높일 수 있습니다.



![image-20240218012340353](/images/2024-02-15-DeepCert/image-20240218012340353.png)

contrast와 blur도 마찬가지입니다. 각 구간에서 분류 정확도가 가장 높은 모델을 선택해, robustness의 상승을 꾀할 수 있습니다.



##### identification of class robustness

![image-20240218012329693](/images/2024-02-15-DeepCert/image-20240218012329693.png)

각 클래스 별로 contextual perturbation에 얼마나 강인한지 확인할 수 있습니다. 위 그래프는 haze perturbation의 변화에 따른 각 클래스 별 정확도의 변화를 나타냅니다. 런타임 혹은 재학습할 때의 완화 전략을 수립할 때 위의 정보를 이용할 수 있습니다.

- class1의 정확도가 유지되는 이유는, 모델이 흰색 이미지를 1번 클래스로 분류하기 때문입니다. 



##### generation of meaningful counterexamples

![image-20240218012318621](/images/2024-02-15-DeepCert/image-20240218012318621.png)

위 그림은 Model 3a의 hazing perturbation counterexample입니다. perturbation이 가해진 의미있는 counterexample의 생성은 도메인 전문가로 하여금 모델의 robustness를 판단할 수 있는 단서를 제공할 수 있습니다.



#### formal verification

![image-20240218012305814](/images/2024-02-15-DeepCert/image-20240218012305814.png)

hazing perturbation에 대한 formal verification과 test-based verification을 함께 나타낸 표입니다. formal verification과 test-based verification를 통해 구한 최소 perturbation의 값에 차이가 나타나지 않았습니다.

- test-based verification이 completeness를 보장하지 않음에도, formal verification과 비교할 때 차이가 관찰되지 않은 점은 특이한 사실입니다.

- CIFAR-10에서는 test-based verification 보다 formal verfication에서 $\epsilon$​이 더 낮게 찾아진 사례가 관찰됩니다.

  

또 한 가지 우리가 주목해야할 점은, **non-contextual point robustness가 contextual robustness를 설명하기에 충분하지 않다는 사실입니다**. sample #52는 Model 1A에서보다 Model 1B에서 더 non-contextual point robustness가 더 높습니다. 하지만, contextual robustness는 오히려 Model 1A에서 더 높게 나타납니다. 따라서 contextual robustness를 검증하기 위해서는, 이를 위한 실험을 따로 수행할 필요가 있겠습니다.



### CIFAR-10

CIFAR-10 데이터셋을 이용한 실험에서도 비슷한 결과를 보입니다. 자세한 내용은 논문에서 확인하실 수 있습니다.





## 느낀 점

DeepCert는 신경망의 Contextual-robustness를 검증할 때 꽤 유용한 도구인 것 같습니다. 논문에서 제시한 model switching은 분명 유효한 전략이지만, 여러 모델을 가지고 있어야 한다는 점에서 모델 크기의 면밀한 비교가 수반되어야 할 것입니다.



![image-20240218225935879](/images/2024-02-15-DeepCert/image-20240218225935879.png)

Contextual-perturbation은 non-contextual point perturbation와 비교했을 때, perturbation encoding이 결정되어있기 때문에 비교적 더 큰 범위의 탐색이 비교적 가능합니다. non-contextual point perturbation와 달리 contextual perturbation의 경우는 perturbation이 가해진 이후 이미지가 클래스 도메인에서 얼마나 멀어지고 있는 지 꾸준하게 확인할 필요가 있어 보입니다.



#### 글을 끝마치며..

contextual robustness라는 아이디어를 쉽게 알아갈 수 있었던 것 같고, 내용이 크게 복잡하지 않아서 이 도메인을 처음 접하시는 분들도 어렵지 않게 읽으실 수 있을 거라고 생각합니다.

첫 논문 포스팅을 적어봤는데요, 글에 이상한 점이나 이해하기 어려운 내용이 있다면, 편하게 댓글란에 의견 남겨주세요. 감사합니다.
