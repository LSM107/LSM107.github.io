---
layout: single

title:  "라플라스 변환(Laplace Transform)"

categories: Engineering_Mathematics

tag: [Laplace Transform]

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

이 포스팅은 '라플라스 변환'에 대한 내용을 담고 있습니다.



자료 출처: <https://www.youtube.com/playlist?list=PLbJ_QJGE4c4ioWGWpam2yVva7b3DzwlrE>









# 라플라스 변환(Laplace Transform)

**라플라스 변환**은 공학에서 정말 많이 사용되는 도구 중 하나입니다. 라플라스 변환을 쓰는 가장 큰 이유는 미적분 방정식을 쉽게 풀 수 있다는 점에 있습니다. 그리고 주파수 응답을 시각적으로 쉽게 이해할 수 있도록 직관을 제공하는 장점도 있습니다.


$$
\mathcal{L}[f(t)] = \int_0^\infty f(t)e^{-st} dt = F(s)
$$


라플라스 변환은 위와 같이 정의됩니다. $t$에 대한 함수 뒤에 지수 함수를 곱한 다음, 0부터 무한대까지 적분하여 구합니다. 따라서 지수항의 계수인 $s$만 남게 되어 결과적으로는 $s$에 대한 함수가 됩니다. 다양한 함수들의 라플라스 변환을 어떻게 하는지, 이 라플라스 변환이 구체적으로 어떤 의미를 갖는지 이어서 살펴보겠습니다.







## 대표적인 라플라스 변환 공식


$$
\mathcal{L}[C] = \frac{C}{s}
$$

$$
\mathcal{L}[t^n] = \frac{n!}{s^{n+1}}
$$

$$
\mathcal{L}[e^{at}] = \frac{1}{s-a}
$$

$$
\mathcal{L}[\cos\omega t] = \frac{s}{s^2 + \omega^2}
$$

$$
\mathcal{L}[\sin\omega t] = \frac{\omega}{s^2 + \omega^2}
$$



위 4개 공식들은 대표적인 라플라스 변환 공식입니다. 가장 기본적인 형태의 라플라스 변환이기 때문에 많은 곳에서 자주 사용됩니다. 따라서 위의 변환 공식들은 반드시 알아두어야 합니다.







## 라플라스 변환의 선형성

라플라스 변환이 선형성을 가지는지 확인해보겠습니다.


$$
\mathcal{L}[\alpha f(t) + \beta g(t)] = \alpha \mathcal{L}[f(t)] + \beta\mathcal{L}[g(t)]
$$


선형성을 가진다는 것은 '위의 관계식이 성립하는가?'입니다. 라플라스 변환은 적분으로 정의되기 때문에 사실 선형성이 거의 보장되는데요, 실제로 그러한지 살펴보겠습니다.


$$
\mathcal{L}[\alpha f(t) + \beta g(t)]
$$

$$
\int_0^\infty (\alpha f(t) + \beta g(t))e^{-st} dt
$$

$$
\int_0^\infty \alpha f(t)e^{-st} + \beta g(t)e^{-st} dt
$$

$$
\int_0^\infty \alpha f(t)e^{-st}dt + \int_0^\infty\beta g(t)e^{-st} dt
$$

$$
\int_0^\infty \alpha f(t)e^{-st}dt + \int_0^\infty\beta g(t)e^{-st} dt
$$

$$
\alpha\int_0^\infty  f(t)e^{-st}dt + \beta\int_0^\infty g(t)e^{-st} dt
$$

$$
\alpha \mathcal{L}[f(t)] + \beta\mathcal{L}[g(t)]
$$



위의 유도를 통해 라플라스 변환이 선형성을 가짐을 알 수 있습니다. 앞서 라플라스 변환이 미적분 방정식을 쉽게 풀 수 있게 해준다고 했는데, 바로 이 선형성이 미적분 방정식의 풀이에 큰 도움을 줍니다.


$$
y'' + y' + y = \sin3t
$$


예를 들어, 위의 함수가 있을 때 $y$가 어떤 형태일지 구하기가 정말 막막한데요, 양변에 라플라스 변환을 취하면 아래와 같습니다.


$$
\mathcal{L}[y'' + y' + y] = \mathcal{L}[\sin3t]
$$

$$
\mathcal{L}[y''] + \mathcal{L}[y'] + \mathcal{L}[y] = \mathcal{L}[\sin3t]
$$

$$
\mathcal{L}[y''] + \mathcal{L}[y'] + \mathcal{L}[y] = \frac{3}{s^2 + 9}
$$



위와 같이 정리하고 나니까 조금은 간단해진 모습입니다. 그런데 도함수의 라플라스 변환을 어떻게 해야하는 지가 문제입니다. 다행히도 도함수의 라플라스 변환은 간단하게 구할 수 있습니다.







## 도함수의 라플라스 변환

일계도함수의 라플라스 변환 공식 유도는 아래와 같습니다.


$$
\mathcal{L}[f'] = \int_0^\infty f'(t)e^{-st}dt
$$

$$
u' = f'(t)
$$

$$
v = e^{-st}
$$



부분적분을 통해 정리하면 아래와 같습니다.


$$
\mathcal{L}[f'] = \int_0^\infty u'v \space dt
$$

$$
\int_0^\infty u'v \space dt = [uv]^b_a - \int_a^buv'
$$

$$
[uv]^b_a - \int_a^buv' = [f(t)e^{-st}]_a^b - \int_a^bf(t)\times(-s)(e^{-st})dt
$$

$$
[f(t)e^{-st}]_a^b - \int_a^bf(t)\times(-s)e^{-st}dt = -f(0) + s\int_0^\infty f(t)(e^{-st})dt
$$

$$
-f(0) + s\int_0^\infty f(t)(e^{-st})dt = s\mathcal{L}[f] - f(0)
$$



따라서 일계도함수의 라플라스 변환은 아래와 같이 원시 함수의 라플라스 변환으로 표현됩니다.


$$
\mathcal{L}[f'] = s\mathcal{L}[f] - f(0)
$$


일계도함수의 라플라스 변환을 위와 같이 정리했는데, 더 확장해서 고계도함수의 라플라스 변환 공식은 아래와 같습니다(유도 과정은 일계도함수의 유도 과정을 연속해서 적용).


$$
\mathcal{L}[{f^{(n)}}] = s^n\mathcal{L}[f] - s^{n-1}f(0) - s^{n-2}f'(0) - \ldots -f^{(n-1)}(0)
$$






## 헤비사이드 부분분수 분해(Heaviside Method)

헤비사이드 부분분수 분해는 부분분수 분해를 할 때, 각 부분분수의 계수를 쉽게 구할 수 있게 해주는 일종의 도구입니다. 


$$
\frac{s^2 + s + 1}{(s-2)(s-1)(s^2 + 1)} = \frac{a}{s-2} + \frac{b}{s - 1} + \frac{cs + d}{s^2 + 1}
$$


부분분수 분해는 위와 같이 통분된 형태를 각 인수를 분모로 하는 분수의 합, 차로 바꾸어 표현하는 것을 의미합니다. 왼쪽의 식을 오른쪽 식의 형태로 바꾸는 일은 생각보다 간단합니다. 이때 각 계수를 구할 때 헤비사이드 방법론을 사용합니다.

먼저 $a$를 구하기 위해 양 변에 $s-2$를 곱해줍니다.


$$
\frac{s^2 + s + 1}{(s-1)(s^2 + 1)} = a + \frac{b(s-2)}{s - 1} + \frac{(cs + d)(s-2)}{s^2 + 1}
$$


이제 위의 식에 $s=2$를 대입하면 $a$값을 쉽게 구할 수 있습니다.


$$
\frac{7}{5} = a
$$


$b$를 구하려면 다시 양 변에 $s-1$를 곱해주고 $s=1$을 넣어줍니다.


$$
b = -\frac{3}{2}
$$
 

$c$와 $d$를 구하는 것도 마찬가지로 양 변에 $s^2+1$를 곱해준 다음 $s = i$를 대입합니다.


$$
c = \frac{1}{10}, \space\space\space\space\space\space\space\space\space\space
d = -\frac{3}{10}
$$


위와 같은 방식으로 부분분수의 계수를 구하는 방법을 **헤비사이드 방법론**이라고 합니다.







