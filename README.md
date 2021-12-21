### 이미지 캡셔닝을 이용한 크롬 확장프로그램(Extension) img2U 개발을 위한 API

-----
- **:warning:사전설명:**
  - **연구개발 진행은 연구실 서버를 통하여 진행하였으며, 2021년 12월 1일기준으로 서버를 반납하여 실제 본 저장소의 API를 사용하기 위해서는 사전의 서버구축 및 세팅이 필요합니다.**

- **최종보고서:** 
  - :page_facing_up: [최종보고서](https://github.com/img2U/img2U-API/blob/main/img2U_%EC%B5%9C%EC%A2%85%EB%B3%B4%EA%B3%A0%EC%84%9C.pdf)

- **Extension Repository:**
  - :page_facing_up: [img2U-Extension](https://github.com/img2U/img2U-Extension)
- **영상링크:**
  - :film_strip: [Youtube](https://www.youtube.com/watch?v=3N2XEK0JUtE)

-----
- **모델 구조 및 API 구조**
    <details> 
      <summary><b>모델구조: (클릭하여 확인) </b></summary>
        <img src="https://github.com/img2U/img2U-API/blob/main/structure_2.png">
    </details>

    <details> 
      <summary><b>API구조: (클릭하여 확인)</b></summary>
        <img src="https://github.com/img2U/img2U-API/blob/main/structure_1.png">
    </details>

-----

- **A. 파일구조**

  - [API](https://github.com/img2U/img2U-API/tree/main/API) (folder): ~~Flask테스트를 위해서 생성된 폴더,~~ InceptionV3 모델 저장되어 있음
  - [API_demo](https://github.com/img2U/img2U-API/tree/main/API_demo) (folder): API실제 작동 코드 저장
    - `app.py`: server 실행 코드 
    - `image_captioning.py`: image captioning 모델 구조 코드

  - [checkpoints/train](https://github.com/img2U/img2U-API/tree/main/checkpoints/train) (folder): 모델 저장을 위한 폴더
  - [captions.pkl](https://github.com/img2U/img2U-API/blob/main/captions.pkl) (file): caption을 pickle파일로 저장한 파일

-----

- **B. 사용방법**
  - **:star: 사전준비**
     - :heavy_check_mark:	(필수) Python+Flask를 통하여 구현된 코드가 실행가능한 서버구축
       - **필수 라이브러리**: `tensorflow`/`flask`/`numpy`/`werkzeug`/`flask_cors`/`json`/`urllib`/`matplotlib`/`pickle`
       - **Python 버전**: >3.6
       - **tensorflow 버전**: >=2.6
     - :heavy_check_mark:	(필수) Papago 개발자센터에서 Client ID 및 Client Secret 발급
     - :+1:  (추천) GPU가 탑제된 서버
   
  - **:star: 코드설정**
     - **1. API_demo/app.py:**
        - line 26, 27: Papago API에서 발급받은 Client ID값과 Secret값으로 변경
        - line 52: checkpoints에 저장된 모델을 호출할 수 있음 (새로운 모델이 생성되었으면 해당 폴더에 넣어 모델파일 위치 수정)
        - line 95~147: POST방식으로 API호출하는 부분(Papago와 연동되어 있지 않음)
        - line 151: `app.run(host='0.0.0.0', debug=False, port=8888)` 여기서 host와 port를 서버에 맞게 설정하기
        - API호출 주소: GET방식으로 호출하며 `https:// 서버주소 /?img=` (= 뒤에 이미지 주소 넣어서 호출)
      - **2. API_demo/image_captioning.py:**
        - line 35, 36: InceptionV3 모델 저장된 폴더에서 불러오기
        - line 66: 저장된 caption pickle파일 불러오기

-----

- **C. 실행방법:**
    - **본 API폴더 에서 `python app.py`로 API서버 실행**
