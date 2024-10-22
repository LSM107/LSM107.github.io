---
layout: single

title:  "Streamlit을 활용한 웹 서비스 구축"

categories: Web

tag: [Streamlit, Web]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
---



**글에 들어가기 앞서...**

이 포스팅은 '**Streamlit**'에 대한 내용을 담고 있습니다.



자료 출처

- <https://www.youtube.com/watch?v=F8a-0JFHfOo&list=PLR4H8X__Qok291XbTXW-9u_u-9XHynfmT>
- <https://medium.com/@ericdennis7/beautify-streamlit-using-tailwind-css-5b5c725c3dfc>









# Streamlit

**Streamlit**은 파이썬 언어만으로 웹 서비스를 만들 수 있도록 다양한 편의 기능을 제공해주는 라이브러리입니다. **Steamlit**은 다양한 장점들을 가지고 있는데요, 다른 웹 개발 툴을 사용해 본 적은 없지만, 그들과 비교했을 때 코드가 짧고 간결하다고 합니다. 그리고 **Streamlit Cloud** 기능을 이용한 쉬운 배포 기능도 **Streamlit**의 큰 장점입니다. 때문에 파이썬에서 데이터 분석, 인공지능 모델이 적용된 서비스를 개발해 배포하는데 굉장히 적합한 툴입니다. 







## Streamlit 개발 환경 설정

<img src="images/2024-10-22-Streamlit/image-20241019130235602.png" alt="image-20241019130235602" style="zoom:50%;" />

 `app.py`라는 파이썬 파일을 생성하고, 해당 파일 내에 위와 같이 입력합니다. `set_page_config`는 페이지 탭의 이름과 그림 등을 설정하는 함수입니다.

`app.py`를 실행시키면, 아래의 창이 생성되는 것을 확인할 수 있습니다. `app.py`에 적은 내용들을 화면에 출력할 때에는 아래의 명령어를 터미널에 입력합니다.

```
streamlit run app.py
```



<img src="/images/2024-10-22-Streamlit/image-20241019130940184.png" alt="image-20241019130940184" style="zoom:50%;" />







## Streamlit Widget

- **Streamlit Widget** 공식 문서: <https://docs.streamlit.io/develop/api-reference/widgets>



**Streamlit**의 공식 문서 사이트를 들어가보면, 사이트를 장식할 수 있는 다양한 기능들을 확인할 수 있습니다. 소개된 기능들이 구현된 코드를 살펴보겠습니다.

