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



![image-20241019163315583](/images/2024-10-22-Streamlit/image-20241019163315583.png)



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







## Tailwind CSS 적용

Tailwind CSS는 미리 다양한 디자인으로 만들어진 클래스를 사용해서 쉽게 스타일링을 할 수 있도록 하는 프레임워크입니다. 즉, 디자인 요소를 따로 적는 것이 아니라 class에서 지정하도록 해, 사용성이 훨씬 좋고, 코드도 간결해집니다.



```python
import json
import streamlit as st

# Data to be inserted into the chart
data = [300, 50, 100]

# Convert the data list to a JSON string
data_json = json.dumps(data)

html_string = f"""
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 p-8">
    <div class="shadow-lg rounded-lg overflow-hidden">
    <div class="py-3 px-5 bg-gray-50">Doughnut chart</div>
    <canvas class="p-10" id="chartDoughnut"></canvas>
    </div>

    <!-- Required chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <!-- Chart doughnut -->
    <script>
    const dataDoughnut = {{
        labels: ["JavaScript", "Python", "Ruby"],
        datasets: [
        {{
            label: "My First Dataset",
            data: {data_json},
            backgroundColor: [
            "rgb(133, 105, 241)",
            "rgb(164, 101, 241)",
            "rgb(101, 143, 241)",
            ],
            hoverOffset: 4,
        }},
        ],
    }};

    const configDoughnut = {{
        type: "doughnut",
        data: dataDoughnut,
        options: {{}},
    }};

    var chartBar = new Chart(
        document.getElementById("chartDoughnut"),
        configDoughnut
    );
    </script>
</body>
"""

st.components.v1.html(html_string, height=1000)
```

위 코드는 Tailwind CSS에서 정의된 클래스를 사용해 쉽고 간편하게 도넛형 그래프를 화면에 표현해주는 코드입니다.