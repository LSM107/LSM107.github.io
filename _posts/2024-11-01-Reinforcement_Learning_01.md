---
layout: single

title:  "강화학습 01: 강화학습이란?"

categories: Reinforcement Learning

tag: [Reinforcement Learning]

typora-root-url: ../

toc: true

author_profile: false

sidebar:
    nav: "docs"

# search: false
use_math: true
---



**글에 들어가기 앞서...**

이 포스팅은 '**강화학습**'에 대한 내용을 담고 있습니다.



자료 출처: 단단한 강화학습, Reinforcement Learning An Introduction , 2nd edition. 리처드 서튼, 앤드류 바트로, 김성우(옮긴이)









# 강화학습

강화학습이란 주어진 상황에서 최대한의 보상을 가져다주도록 하는 행동의 학습을 의미합니다. 여기에서의 학습의 주체는 학습자(Agent)는 지침을 받지 않은 상태에서 어떤 행동을 취할지를 오로지 **시행착오**를 통해서 학습합니다. 보상이라는 것은 즉각적으로 주어지는 것이 아닌데요, 예로 바둑과 같은 게임에서는 게임이 마지막에 종료되어야 보상을 받습니다(승리 또는 패배). 게임이 진행되는 와중에 **즉각적으로 행동에 대한 보상을 받지 못한다는 점**은 학습자에게 있어 굉장히 어려운 점이면서 강화학습이 가지는 독특한 특성입니다.

**지도학습과는 어떤 차이점이 있을까요?** 지도학습이란 외부 전문가의 지침인 레이블(Label)이 포함된 훈련 예제로부터 학습하는 것을 의미합니다. 이러한 종류의 학습 방식은 환경과 상호작용을 하는 경우에는 적합하지 않은데요, 다시 말해 동적인 환경에서 지도학습은 잘 동작하기 어렵습니다. 

**강화학습은 비지도학습과도 구분됩니다.** 비지도학습은 데이터 집합 내에서 숨겨진 구조를 찾는 방식으로 동작합니다. 숨겨진 구조를 찾는 것은 분명 강화학습에서 큰 도움이 될 수 있지만, 그것만으로는 강화학습 문제를 풀지 못합니다. 







## Exploitation & Exploration

강화학습 학습자는 환경에서 시행착오를 직접 겪으면서 학습하게 됩니다. 학습자가 어느정도 환경을 겪으면, 어느 행동이 어느정도의 보상을 가져올 지 대충 알게 되는데요, 여기에서 학습자는 2가지 선택지를 갖습니다.

1. 더 높은 보상을 얻을 수 있는 가능성이 있는 경험하지 않은 행동을 **탐험(Exploration)**한다.
2. 이미 경험해 본 행동들을 최대한으로 **활용(Exploitation)**해 높은 보상을 얻을 수 있도록 한다.



탐험과 활용, 둘 중 어느 하나만 추구하는 학습자는 강화학습 문제를 해결하기 어렵습니다. 이 둘을 어떻게 조절해야 하는가는 오랫동안 수학자들의 연구 대상이었으나 여전히 풀리지 않은 문제입니다.







## 강화학습의 구성 요소

강화학습에는 네 가지 주된 구성 요소가 있습니다.

1. **정책(Policy)**: 특정 시점에 학습자가 취해야 하는 행동을 정의합니다. 단순한 함수 혹은 Look-up table일수도 있습니다. 정책은 결정론적으로 행동을 선택할 수도, 확률론적으로 행동을 선택할 수도 있지만, 일반적으로는 확률적으로 행동을 선택합니다.

   

2. **보상 신호(Reward Signal)**: 매 타임 스텝마다 주변 환경에서 학습자에게 보여하는 보상의 정도입니다. 학습자의 유일한 목표는 장기간에 걸쳐 얻게되는 보상 신호의 합을 최대화하는 것입니다. 보상 신호 역시 확률론적일 수 있습니다.

   

3. **가치 함수(Value Function)**: 장기적인 관점에서 무엇이 좋은가를 나타냅니다. **가치**란 해당 상태로부터 일정 시간 동안 학습자가 기대할 수 있는 보상의 총량을 의미합니다. 즉, 장기적 관점으로 평가한 상태의 장점을 의미합니다. 가치 함수란 이 가치를 평가하는 함수입니다.

   

4. **모델(Model)**: 모델은 환경의 변화를 모사하는 역할을 수행합니다. 모델이 있는 경우에는 **계획(Planning)**을 사용해 문제를 해결할 수 있습니다. 모델이 없는 경우에는 전적으로 시행착오 학습자가 취하는 행동을 기반으로 학습해야 합니다.



가치는 보상에 대한 예측이기 때문에 어떤 측면에서는 부수적이지만, 행동을 결정할 때 가장 많이 고려되는 것은 가치입니다. 이게 당연한 것은 장기적으로 최대한 많은 보상을 받을 수 있어야 하기 때문입니다. 불행히도 보상을 결정하는 일보다 가치를 결정하는 것이 훨씬 더 어렵습니다. 왜냐하면 가치는 학습자의 전 생애주기 동안 학습자가 관찰한 것들로부터 반복 추정되어야 하기 때문입니다. **효과적으로 가치를 추정하는 방법을 알아내는 것이 강화학습의 핵심이라고 말할 수 있습니다.**

이 책에서 다루는 거의 대부분의 내용은 가치를 추정하는 방법에 관한 것이지만, 가치의 추정이 반드시 강화학습 문제를 풀기 위해 필수적이지는 않습니다. 대표적으로 유전자 알고리즘, 모의 담금질과 같은 진화적 최적화 방법에서는 가치 함수를 추정하지 않고, 때로는 이러한 방식이 효과적입니다(정책의 개수가 충분히 적을 때). 하지만 진화적 방법은 환경과의 상호작용이 갖는 세부 정보들을 활용하지 않기 때문에 대부분의 경우에는 비효율적입니다.







## 틱택토 게임

상대방이 이길 수 있게 해주는 덜 숙달된 사람과 게임을 하고 있다고 가정해 보겠습니다. 그리고 또 무승부나 패배나 동일하게 안 좋은 결과라고 생각해 보겠습니다. 어떻게 해야 상대방의 잘못된 선택을 찾고, 승리 확률을 최대화할 수 있을까요?



**미니맥스**

```python
import math

def print_board(board):
    print("\n현재 보드 상태:")
    for row in board:
        print("| " + " | ".join(row) + " |")
    print("\n")

def check_winner(board):
    # 행과 열 확인
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != ' ':
            return board[i][0]  # 행 승리
        if board[0][i] == board[1][i] == board[2][i] != ' ':
            return board[0][i]  # 열 승리
    # 대각선 확인
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return board[0][0]  # 대각선 승리
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return board[0][2]  # 반대 대각선 승리
    return None  # 승자가 없음

def is_board_full(board):
    for row in board:
        if ' ' in row:
            return False
    return True

def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner == 'O':
        return 1  # AI 승리
    elif winner == 'X':
        return -1  # 사용자 승리
    elif is_board_full(board):
        return -1  # 무승부

    if is_maximizing:
        best_score = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ' '
                    best_score = max(best_score, score)
        return best_score
    else:  # 최소화 단계
        best_score = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ' '
                    best_score = min(best_score, score)
        return best_score

def find_best_move(board):
    best_score = -math.inf
    move = (0, 0)  # 초기값 설정
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                score = minimax(board, 0, False)
                board[i][j] = ' '
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return move

def main():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    print("틱택토 게임을 시작합니다. 당신은 'X', AI는 'O'입니다.")
    print_board(board)

    while True:
        # 사용자 입력
        while True:
            try:
                user_input = input("당신의 턴입니다. 행과 열을 입력하세요 (예: 0 1): ")
                row, col = map(int, user_input.strip().split())
                if board[row][col] == ' ':
                    board[row][col] = 'X'
                    break
                else:
                    print("이미 선택된 위치입니다. 다른 위치를 선택하세요.")
            except (ValueError, IndexError):
                print("올바른 형식으로 입력하세요. 예: 0 1")
        print_board(board)

        # 승리 조건 확인
        if check_winner(board) == 'X':
            print("축하합니다! 당신이 승리했습니다!")
            break
        elif is_board_full(board):
            print("무승부입니다!")
            break

        # AI의 턴
        print("AI의 턴입니다.")
        ai_row, ai_col = find_best_move(board)
        board[ai_row][ai_col] = 'O'
        print_board(board)

        # 승리 조건 확인
        if check_winner(board) == 'O':
            print("AI가 승리했습니다!")
            break
        elif is_board_full(board):
            print("무승부입니다!")
            break

if __name__ == "__main__":
    main()

```

위 코드는 미니맥스 방식을 사용해서 틱택토 학습자를 구현한 파이썬 코드입니다. 위 코드를 통해 플레이하는 학습자는 미숙한, 덜 숙달된 플레이어을 상대로 최선의 행동을 할 수 없습니다. 이 문제의 핵심은 덜 숙달된 사람이 플레이하는 정책을 파악하고, 이를 토대로 상대방의 실수를 응징할 수 있는 학습자를 만드는 것입니다. 미니맥스 방식에서는 항상 최선의 플레이를 하는 상대방을 가정하는데요, 미숙한 플레이어의 정책 분포를 활용하지 않는 학습자는 속된 말로, 꼼수를 사용한 승리를 하기 어렵습니다. 



**가치 함수**

틱택토 게임에서 가치 함수를 사용해 문제를 해결한다면, 여기에서의 가치 함수는 게임에서 나타날 수 있는 모든 상태가 입력값이 되고, 그 출력은 **해당 상태에서 승리할 확률에 대한 가장 최신의 추정**입니다. 가치 함수의 초기값은 0.5로 설정되고, 아래의 식에 따라 점진적으로 업데이트됩니다.


$$
V(S_t) \leftarrow V(S_t) + \alpha[V(S_{t+1}) - V(S_t)]
$$


학습자는 가치 함수에 나타나는 확률값에 따라 탐욕적인 선택을 하게 됩니다. 그리고 경험을 바탕으로 위의 식에 따라 조금씩 업데이트됩니다. 위 식에서 $\alpha$는 **시간 간격 파라미터(Step-Size Parameter)**로 학습의 속도에 영향을 미칩니다. 위 식에 따라 학습되는 방식을 **시간차 학습(Temporal-difference Learning)**이라고 하는데, 두 연속적인 시각의 추정값의 차이에 기반하여 학습되기 때문에 시간차 학습이라고 불립니다.

가치함수를 통해 업데이트를 하는 방식은 진화적 방법과 명확한 차이점이 있습니다. 그 점은 **진화적 방법에서는 게임이 수행된 후의 최종 결과만을 사용한다는 사실입니다.** 반면, 가치 함수를 이용하는 방법은 개별적인 상태들을 평가하는 것을 허용하기 때문에, 게임 도중의 정보를 보다 잘 활용합니다.



