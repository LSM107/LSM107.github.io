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





























