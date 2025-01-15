---
layout: single

title:  "Streamlit을 활용한 웹 서비스 구축"

categories: Tools

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

<img src="/images/2024-10-22-Streamlit/image-20241019130235602.png" alt="image-20241019130235602" style="zoom:50%;" />

 `app.py`라는 파이썬 파일을 생성하고, 해당 파일 내에 위와 같이 입력합니다. `set_page_config`는 페이지 탭의 이름과 그림 등을 설정하는 함수입니다.

`app.py`를 실행시키면, 아래의 창이 생성되는 것을 확인할 수 있습니다. `app.py`에 적은 내용들을 화면에 출력할 때에는 아래의 명령어를 터미널에 입력합니다.

```
streamlit run app.py
```



<img src="/images/2024-10-22-Streamlit/image-20241019130940184.png" alt="image-20241019130940184" style="zoom:50%;" />







## Streamlit Widget

- **Streamlit Widget** 공식 문서: <https://docs.streamlit.io/develop/api-reference/widgets>



**Streamlit**의 공식 문서 사이트를 들어가보면, 사이트를 장식할 수 있는 다양한 기능들을 확인할 수 있습니다. 소개된 기능들이 구현된 코드를 살펴보겠습니다.

```python
import streamlit as st

print("page reloaded")
st.set_page_config(
    page_title="football agent")

st.title("Football Players")
st.markdown("**축구선수**에 대한 정보를 직접 추가하고 수정할 수 있습니다.")

type_emoji_dict = {
    "GK": "🧤",

    "DF": "🛡️",
    "CB": "🛡️",
    "SW": "🛡️",
    "FB": "🛡️",
    "LB": "🛡️",
    "RB": "🛡️",
    "WB": "🛡️",
    "LWB": "🛡️",
    "RWB": "🛡️",

    "MF": "⚽",
    "CM": "⚽",
    "DM": "⚽",
    "AM": "⚽",
    "LM": "⚽",
    "RM": "⚽",
    "LW": "⚽",
    "RW": "⚽",

    "FW": "🗡️",
    "CF": "🗡️",
    "SS": "🗡️",
    "LW": "🗡️",
    "RW": "🗡️",
    "F9": "🗡️"
    
}

initial_players = [
    {
        "name": "Lionel Messi",
        "type": ["FW", "CF", "SS"],
        "image_url": "https://i.namu.wiki/i/WrefVOGncDZ3Lw81dS9p5P6eAMcCAZr2_BL1VEO8xzodFcF9bcznNvg0U7j7Xx1d4D5ovzvkmaZYEO95PWlqFYCCi-XkTjeG0ZKQz-5SfUAkvA3c36xPqwjU78BftdQtd6xO873LgjSgaV14MyQHDw.webp"

    },

    {
        "name": "Cristiano Ronaldo",
        "type": ["FW", "CF", "SS"],
        "image_url": "https://i.namu.wiki/i/EaJDRxRUqWSVBwK2ZpVXmpEs_-M89y4roCm-MfzOJ4N4QHlXRJISpvr0xFazHzGEB-AnaDQQnwCUXW0kmE-aSxZzbfTmB7t8OMIurvx91VhKiPOGfO_4qExb1bVkFPzg03mdgbXcyxD_MXmhhd2Dow.webp"
    },

    {
        "name": "Neymar Jr.",
        "type": ["FW", "CF", "SS", "LW"],
        "image_url": "https://i.namu.wiki/i/eXbUi0xjfgSxq5zrbo5DXi7QfPfI1_ugB2xu-O0vVlcfqrQJ_h6BvdkzbIDRDeQ-G72QtuRBMO2CD7RO000nqgmc4m1HFGZbTtPcSx1qqlsjAtaYAtcmekHuO6uHwM3DrOlgtDEneCUocmVOOqa7jw.webp"
    },

    {
        "name": "Kylian Mbappé",
        "type": ["FW", "CF", "SS", "LW"],
        "image_url": "https://i.namu.wiki/i/ZepWQsT5z4gS5QPHfWivdn6CNjfkVVd_9G_F3N5bYwBfXwhhHlkx8wAiZ7O7xyoHsVGaBTWv3KIHvnUR_9qFKWHrND7L8OmrSR2TE7pT6aIKl1LKncFBol62CmUT9D9jinjqbDlrHUiEONt5zZlbKg.webp"
    },

    {
        "name": "Robert Lewandowski",
        "type": ["FW", "CF", "SS"],
        "image_url": "https://i.namu.wiki/i/TWTBfovDxIfqZB3i1XUPjaAlm-UdjxiFcYWZmGtNii7ZPVA8M73_m90ZPWHnJm0ZhRWfgxcMXTTuqUYFMiLyNYIp3D5lvxsvRtKvv52BMaQCYrmVpGAn_yx-M48xI4CT2JeNmLKn0MKM3HjHDrDteQ.webp"
    },

    {
        "name": "Kevin De Bruyne",
        "type": ["MF", "CM", "DM", "AM"],
        "image_url": "https://i.namu.wiki/i/Jye-vHg14en0q3oe0AUw-ZXzLbj_ZGr83EzxpnQlRV5mcZKBimeNvWRPGm08z_Z6bHfOxzOGrAcUROcHL3LHvletzLe2s5jnqHDu9H4QoDubxKNydUWHRN3tn95aHljPZqvauZtyNKZVSg3yChuTUw.webp"
    },
]

example_player = {
    "name": "ronaldiho",
    "type": ["FW", "CF", "SS"],
    "image_url": "https://i.namu.wiki/i/0eqIdYexQd3BPqs2UIhyvQSVZLgdEELrTRo_cUFL6QB-d6kjnQUsp4qme4w6gly2LMBdl-ftV5rqj1cISN7ogG0k5KsSVo4bSGPSfKlWBGZrr327m6Lnt5GOEi0nR8ewLuwgYbXZk1obNOmT98Memw.webp"
}

if "players" not in st.session_state:
    st.session_state.players = initial_players


auto_complete = st.toggle("예시 데이터로 채우기")
print("page reloaded, auto_complete:", auto_complete)

with st.form(key="form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input(
            label="선수 이름",
            value=example_player["name"] if auto_complete else ""
        )
    with col2:
        types = st.multiselect(
            label="선수 포지션", 
            options=list(type_emoji_dict.keys()),
            max_selections=4,
            default=example_player["type"] if auto_complete else []
        )
    image_url = st.text_input(
        label='선수 이미지 URL',
        value=example_player["image_url"] if auto_complete else ""
        )
    submit = st.form_submit_button(label="추가")
    if submit:
        if not name:
            st.error("선수 이름을 입력해주세요.")
        elif len(types) == 0:
            st.error("선수 포지션을 한 개 이상 선택해주세요.")
        else:
            st.success("선수가 추가되었습니다.")
            st.session_state.players.append({
                "name": name,
                "type": types,
                "image_url": image_url if image_url else "./images/default.png"
            })


        print("name:", name)
        print("types:", types)
        print("image_url:", image_url)




for i in range(0, len(st.session_state.players), 3):
    row_players = st.session_state.players[i:i + 3]
    cols = st.columns(3)

    for j in range(len(row_players)):
        with cols[j]:
            player = row_players[j]
            with st.expander(label=f'**{i+j+1}. {player["name"]}**', expanded=True):
                st.image(player["image_url"])
                emoji_types = [f"{type_emoji_dict[t]} {t}" for t in player["type"]]
                st.text(" / ".join(emoji_types))
                delete_button = st.button(label="삭제", key = i+j, use_container_width=True)
                if delete_button:
                    del st.session_state.players[i+j]
                    st.rerun()

```

  위 코드는 저장된 선수들의 정보를 화면에 보여주고, 사용자에게 선수들의 정보를 받을 수도 있는 웹 앱 예시입니다.  위의 코드에 사용된 다양한 Streamlit의 메서드들의 기능은 앞서 소개한 사이트에서 확인할 수 있습니다. 사용자에게 입력을 받아오는 경우, 페이지 리로드가 자동으로 실행되어 예상치 못한 동작이 발생할 수 있습니다. 때문에 이를 반드시 확인하고, 필요시 `st.rerun()`을 사용해 오작동을 방지해야 합니다.



<img src="/images/2024-10-22-Streamlit/image-20241019160532865.png" alt="image-20241019160532865" style="zoom:50%;" />







## Streamlit에 Custom CSS 적용하기

**Streamlit**을 통해서만 웹 앱을 디자인하는 데에는 한계가 있습니다. 우리가 원하는 디자인 형식이 라이브러리에 미리 구현되어있지 않은 경우에 이를 화면에 표현하기 어려운데요, Streamlit은 이러한 점을 보완하기 위해 CSS 코드를 적용할 수 있는 기능을 제공합니다.



```python
st.markdown("""
<style>
# 이곳에 CSS 코드를 작성합니다.
</style>
""", unsafe_allow_html=True)
```

CSS 코드를 적용하기 위해서는 `markdown`함수를 사용합니다. 위의 코드가 CSS 코드를 적용하기 위한 기본 형식입니다. html style 태그 사이에 우리가 변경하고자 하는 디자인 요소를 설정합니다.

디자인 요소를 변경할 때에는 개발자 도구를 켜서 우리가 바꾸고 싶은 요소가 html상에서 어떤 클래스로 지정돼있는지 확인합니다.





### 제목 변경

<img src="/images/2024-10-22-Streamlit/image-20241019162845587.png" alt="image-20241019162845587" style="zoom:50%;" />

제목의 디자인을 바꾸기 위해 커서를 올려놓은 상태에서 우클릭, 그리고 검사를 선택합니다. 그러면 위와 같은 개발자 도구로 연결되는데요, 위의 html 코드를 보면 Football Players라는 제목이 `h1`태그 밑에 정의되어 있는게 확인됩니다.



```python
st.markdown("""
<style>
h1 {
	color: red;
}
</style>
""", unsafe_allow_html=True)
```

위와 같이 CSS 코드를 작성하고 적용하면, 제목이 빨간색으로 설정됩니다.



<img src="/images/2024-10-22-Streamlit/image-20241019163315583.png" alt="image-20241019163315583" style="zoom:50%;" />



### 이미지

웹에 나타나는 이미지는 `img`태그를 달고 있습니다. 이미지의 크기를 일괄적으로 맞추고 싶다면, `img`태그를 지정해 변경할 수 있습니다.



```python
st.markdown("""
<style>
img {
	max-height: 300px;
}
</style>
""", unsafe_allow_html=True)
```

그런데, 이 CSS 코드를 사용해 이미지의 디자인을 바꾸면 페이지 간에 이동을 할 때 딜레이, CSS 코드 적용 이전의 이미지 형태가 잠깐 나타났다가 적용 이후로 바뀌는 문제가 발생한다는 문제점이 발생합니다. 따라서 이미지에 대해 CSS 코드를 적용할 때에 별도의 눈속임 장치를 고려할 필요가 있습니다.





### 버튼 비활성화



Expander를 사용해 선수를 나열하도록 만들었기 때문에, Expander의 기본적인 기능들이 자동으로 구현되어 있습니다. 그 중 하나가 요소를 접을 수 있는 기능입니다. html 코드를 살펴보면 `data-testid="stExpanderToggleIcon"`으로 정의된 것이 확인됩니다. 이 정보를 활용해 버튼을 비활성화시키는 CSS 코드를 작성합니다.

<img src="/images/2024-10-22-Streamlit/image-20241019164322722.png" alt="image-20241019164322722" style="zoom:50%;" />

```python
st.markdown("""
<style>
[data-testid="stExpanderToggleIcon"] {
	visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

```

정리하면, 바꾸고자 하는 대상의 **클래스**, 또는 **data-testid**를 **검사**를 통해 확인하고 이를 변경하기 위한 CSS 코드를 **markdown** 함수의 인자로 적어내 실행하면 CSS 코드 내용이 반영됩니다.







## Tailwind CSS 적용

Tailwind CSS는 미리 다양한 디자인으로 만들어진 클래스를 사용해서 쉽게 스타일링을 할 수 있도록 하는 프레임워크입니다. 즉, 디자인 요소를 따로 적는 것이 아니라 class에서 지정하도록 해, 사용성이 훨씬 좋고 코드가 간결해집니다.



```html
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
```

Tailwind CSS를 사용하기 위해서는 위와 같이 head 부분에 `<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">`를 추가해야 합니다. 이 코드가 추가되면 Tailwind CSS에서 제공하는 다양한 클래스 편의기능을 사용할 수 있습니다. body에서 사용된 클래스들은 모두 다 Tailwind CSS의 클래스들인데요, Streamlit을 돌렸을 때에도 잘 적용되는 것을 확인할 수 있습니다.







## Capstone Demo CSS

초기 화면에 있는 여러 요소들의 CSS 태그들을 살펴보고 간단한 디자인 변경 예시들을 살펴봅니다.





### About

<img src="/images/2024-10-22-Streamlit/image-20241021153138166.png" alt="image-20241021153138166" style="zoom:50%;" />

가장 초기 화면에 baLLaMA 프로젝트의 간단한 설명이 있습니다. 가장 상단 제목은 `h2`태그로 변경할 수 있습니다. 위의 화면에서 이미 변화가 적용된 모습을 확인할 수 있는데요, 코드는 아래와 같습니다. 



```python
import streamlit as st

import utils.utils as utils

def body():
    st.divider()
    head = '## About baLLaMA Project'
    st.markdown(head)
    body = '본 프로젝트는 축구 도메인 데이터를 기반으로 대화형 인터페이스를 제공하는 LLM 기반 인공지능 에이전트를 개발하는 것을 목표로 합니다.'
    st.markdown(body)

utils.set_global_page_config()
body()

st.markdown("""
<style>
            
h2 {
    color: #ff0000;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    font-family: 'Arial';
}

</style>
"""
, unsafe_allow_html=True)
```

`main_page.py`의 가장 하단에 CSS를 받을 수 있는 `markdown`함수를 추가해 놓았는데, 이 부분을 반드시 `utils.set_global_page_config()` 하단에 위치시켜야 합니다. 글의 정렬과 색상, 폰트 등등을 바꾼 예시입니다. 위와 같이 HTML 태그에 직접 접근해 변경하는 경우, `h2`태그를 쓰는 다른 요소들에도 동일한 변경 사항들이 적용됩니다. 



<img src="/images/2024-10-22-Streamlit/image-20241021154043900.png" alt="image-20241021154043900" style="zoom:50%;" />

제목 밑에 있는 markdown의 Streamlit 컴포넌트는 `data-testid="stMarkdown"`입니다. `data-testid`속성은 Streamlit이 내부적으로 각각의 컴포넌트에 대해 추가하는 HTML 속성인데, 이 속성을 통해 특정 컴포넌트에만 변화를 적용할 수 있습니다.



```python
st.markdown("""
<style>
            
[data-testid="stMarkdown"] {
    color: red;   
}
            
</style>   
"""
, unsafe_allow_html=True)
```

위와 같이 `data-testid`속성으로 디자인 요소를 변경하는 경우, 해당 컴포넌트에만 변화가 적용됩니다. 



```python
st.markdown("""
<style>
            
.stMarkdown {
    color: red;   
}
            
</style>   
"""
, unsafe_allow_html=True)
```

클래스를 통해 디자인을 변경하는 경우, 위와 같이 클래스 이름 앞에 온점을 찍어 표시합니다. 클래스를 통해 디자인을 변경하면, 동일한 클래스를 사용하는 다른 컴포넌트들에도 변경 사항들이 일괄 적용됩니다.



```python
st.markdown("""
<style>
            
p {
    color: red;   
}
            
</style>   
"""
, unsafe_allow_html=True)
```

`p`태그를 통해 디자인을 변경할 수도 있는데, 이 경우 화면의 거의 모든 글자들의 색상이 빨간 색으로 변경됩니다. 거의 모든 마크다운들이 `p`태그를 가지기 때문에 모든 글들에 일괄적인 변화가 적용됩니다. **따라서 페이지의 글의 크기나, 색상, 폰트, 정렬 등을 변경하고 싶다면, 이 태그를 통해 한 번에 변경할 수 있습니다.**



<img src="/images/2024-10-22-Streamlit/image-20241021163859606.png" alt="image-20241021163859606" style="zoom:50%;" />





### 사이드 바

<img src="/images/2024-10-22-Streamlit/image-20241021163935343.png" alt="image-20241021163935343" style="zoom:50%;" />

사이드바에 사용된 축구공을 든 라마 이미지는 `img`태그를 사용합니다.



```python
st.markdown("""
<style>
            
img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 50%;
    border-radius: 50%;
}
            
</style>   
"""
, unsafe_allow_html=True)
```

이미지의 크기를 조절하고 이미지가 원모양으로 나타나게 변경한 CSS 코드입니다.



<img src="/images/2024-10-22-Streamlit/image-20241021164303687.png" alt="image-20241021164303687" style="zoom:50%;" />

이미지의 전체적인 사이즈가 감소하고, 라마의 이미지가 원 모양으로 나타나게 변경된 것을 확인할 수 있습니다.



<img src="/images/2024-10-22-Streamlit/image-20241021164409815.png" alt="image-20241021164409815" style="zoom:50%;" />

사이드 바를 최소화할 수 있는 버튼이 존재하는데요, 이 기능을 원치 않는 경우 이를 비활성화할 수 있습니다. 개발자 도구에서 해당 버튼의 `data-testid`를 확인할 수 있는데, 이 정보를 사용해 버튼을 없애는 코드를 적어보면 아래와 같습니다.



<img src="/images/2024-10-22-Streamlit/image-20241021164821084.png" alt="image-20241021164821084" style="zoom:50%;" />

```python
st.markdown("""
<style>
            
[data-testid="baseButton-header"] {
    display: none;
}
            
</style>   
"""
, unsafe_allow_html=True)
```

이미지를 디자인을 변경하는 부분을 제외하고 버튼을 없애는 코드만 적용했기 때문에, 이미지가 다시 원래 디자인으로 돌아왔는데요, 화면을 보시면 사이드 바를 최소화할 수 있도록 원래 존재했던 버튼이 사라진 것을 확인할 수 있습니다.



<img src="/images/2024-10-22-Streamlit/image-20241022101310498.png" alt="image-20241022101310498" style="zoom:15%;" />

이미지에 마우스를 위치시키면, 눈에 거의 보이지 않지만 우측 상단에 어떤 버튼이 나타나는 것을 확인할 수 있습니다. 이 버튼의 정체는 이미지 최대화 버튼인데요, Streamlit을 통해 이미지를 화면에 위치하면 이미지를 최대화할 수 있는 기능이 자동으로 추가됩니다. 사실상 필요가 없는 기능이기 때문에 이 버튼을 제거하겠습니다.



<img src="/images/2024-10-22-Streamlit/image-20241021165331068.png" alt="image-20241021165331068" style="zoom:50%;" />

이미지 최대화 버튼의 `data-testid`는 `"StyledFullScreenButton"`입니다. 이 정보를 이용해 이 버튼을 제거하는 코드는 아래와 같습니다.



```python
st.markdown("""
<style>
            
[data-testid="StyledFullScreenButton"] {
    display: none;
}
            
</style>   
"""
, unsafe_allow_html=True)
```

이제 더 이상 마우스를 이미지 위에 놓아도 최대화 버튼이 나타나지 않습니다.