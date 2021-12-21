### 이미지 캡셔닝을 이용한 크롬 확장프로그램(Extension) img2U 개발을 위한 API

-----

- **A. 파일구조**

  - [API](https://github.com/img2U/img2U-API/tree/main/API) (folder): Flask테스트를 위해서 생성된 폴더, 실제 API 작동에는 사용하지 않음
  - [API_demo](https://github.com/img2U/img2U-API/tree/main/API_demo) (folder): API실제 작동 코드 저장
    - `app.py`: server 실행 코드 
    - `image_captioning.py`: image captioning 모델 구조 코드

  - [checkpoints/train](https://github.com/img2U/img2U-API/tree/main/checkpoints/train) (folder): 모델 저장을 위한 폴더

-----

- **B. 사용방법**
  - **0. 사전준비**
    - Python+Flask를 통하여 구현된 코드가 실행가능한 서버구축(必)
      - **필수 라이브러리**: tensorflow/flask/numpy/werkzeug/flask_cors/json/urllib
    - Papago 개발자센터에서 Client ID 및 Client Secret 발급(必)
