# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 배성렬
- 리뷰어 : 정주열


# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 챗봇 성능을 체크하기 위한 테스트가 질문의 맥락에 맞게 답변이 잘 나온 것을 확인했습니다.
    <img width="534" height="698" alt="image" src="https://github.com/user-attachments/assets/543b3344-bb1d-4fa8-b716-942baddc803b" />

- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - Transformer 모델을 이용해서 챗봇을 만드는 프로젝트에서 모델 정의하는 부분이 가장 핵심이라 생각했습니다. 그리고 주석이 잘 달려있어서 어떻게 모델이 구성되어 있는지 파악하기 쉬웠습니다.
    <img width="851" height="828" alt="image" src="https://github.com/user-attachments/assets/0d643b43-7d53-43dd-a3a3-c5e10fa7e6c6" />

- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - learning rate 처리하는 부분에서 모델 학습이 잘안되는 것을 확인하고, 40을 곱해서 보정한 것을 보여줬습니다.
    <img width="753" height="272" alt="image" src="https://github.com/user-attachments/assets/6546f91f-043a-4c30-a07c-afcfc2a81ecd" />

- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 파이프라인 플로우 차트가 코드 흐름을 시각화 하고 있어서 마무리로 잘 정리가 되는것 같습니다. 또한 회고 내용도 결과에 맞게 잘 작성되어 있습니다.
    <img width="1421" height="713" alt="image" src="https://github.com/user-attachments/assets/72560538-c6e9-4a5c-a489-048e8ca0ab6f" />
    <img width="1394" height="127" alt="image" src="https://github.com/user-attachments/assets/b2d1f2b5-4a7b-4b05-9a7f-95b488f43b35" />

- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 기능별로 함수화 모듈화가 잘되어 있습니다.
    <img width="755" height="428" alt="image" src="https://github.com/user-attachments/assets/c8c1e55f-d492-4921-aaf6-d655ab41c607" />



# 회고(참고 링크 및 코드 개선)
코드 내용을 보면서 주석이 잘 작성되어 있어서 코드를 이해하기 쉬웠습니다. 또한 마지막에 플로우차트를 보면서 지금까지 본 코드내용이 정리되면서 흐름을 복습하기 좋았습니다. 코드 리뷰를 하면서 얻어가는게 정말 많아서 좋았습니다.
