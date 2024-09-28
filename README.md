# deeplearning-repo-4
딥러닝 프로젝트 4조. googeese

# Title : 찾아라! 얄리

## 1. 기능 리스트

| 항목 | 세부 기능 |
|-----|----------|
| 1. 공항 구역 안내 기능 | 사용자에게 공항 구역 안내 기능 |
| 2. 실종 접수 기능 | 접수 항목<br>&ensp;• 사진 (필수)<br>&ensp;• 의상정보<br>  &ensp;&ensp; *  상의 (색상 코드값)<br>  &ensp;&ensp; * 하의 (색상 코드값) |
| 3. 실종자 탐지 | 접수 정보에 일치하는 사람 자동 확인<br>&ensp;• 확인 기준 (얼굴: 95%) |
| 4. 실종자 추적 | 탐지 이후의 동선감<br>• 추적 실패 기준: 확인이 3초이상 사라진 경우 중단된 |
| 5. 실종자 재추적 | 재추적 |
| 6. 실종 중장자 발견시 체크 기능 | 자동확인 얼굴 보호자 재확인<br>&ensp;• True: 실종자 매칭 성공<br>&ensp;• False: 제외 목록추가 (재 탐지 방지) |
| 7. 실종자 안내 기능 | 보호자가 로봇 위치로 오서 로봇에게 안내 받음<br>&ensp;• Baby Goose GUI 직업 안로 버튼 클릭 |
| 8. 공항 순찰 기능 | 순찰 항목<br>&ensp;• 쓰러진 사람 (우선순위 1)<br>&ensp;• 졸은 바닥 (우선순위 2) |
| 9. 순찰 이벤트 감지 대응 기능 | 주변 알림<br>&ensp;• Baby Goose GUI 직업 알림 강화임<br>&ensp;• 이벤트 발생 message 표시<br>  &ensp;&ensp; * 쓰러진 사람<br>  &ensp;&ensp; * 젖은 바닥 |
| 10. 순찰 이벤트 종료 기능 | 담당자 완결(로봇 위치) 조치 후 상황 종료 입력<br>&ensp;• Baby Goose GUI 직업 안로 버튼 클릭 |

## 2. 시스템 구성도
### 2.1 HW 구성도

![HW_diagram (1)](https://github.com/user-attachments/assets/68ed9b1c-494d-459b-98e4-343a0d8898ac)


### 2.2 SW 구성도

![sw_diagram](https://github.com/user-attachments/assets/7612e9aa-8028-4bb0-abe0-2289dc0fdeee)

## 3. 활용 기술
| 항목 | 사용 기능 |
|-----|----------|
|OS|<img src="https://github.com/user-attachments/assets/7fec89aa-dab6-4232-8da7-d225dd7e35a1" alt="ubuntu" style="width: 75px; height: 30;"> <img src="https://github.com/user-attachments/assets/f6edbdc4-608f-4e22-80ea-6f6b8666d8d5" alt="Raspberry_Pi_OS" style="width: 120px; height: 30;">
|프로그래밍 언어|<img src="https://github.com/user-attachments/assets/21038c16-552e-44a4-bc86-67315ae989a5" alt="python" style="width: 110px; height: 30;">
|개발환경|<img src="https://github.com/user-attachments/assets/ae10e02a-d00e-40da-8f16-2ec80fca3b95" alt="git" style="width: 80px; height: 30;"> <img src="https://github.com/user-attachments/assets/d5e11b6c-0666-4301-99ba-ce95d4843558" alt="vscode" style="width: 52px; height: 30;"> <img src="https://github.com/user-attachments/assets/3d95b8e3-5a8b-4fe9-8168-561fc1035437" alt="arduino" style="width: 100px; height: 30;">|
|문서/프로젝트 관리|<img src="https://github.com/user-attachments/assets/5d352779-2778-470a-b0be-2de26dd234b7" alt="github" style="width: 100px; height: 30;"> <img src="https://github.com/user-attachments/assets/29c3b80c-9e19-47c7-99b9-adcf0ac6c00f" alt="confluence" style="width: 120px; height: 30;"> <img src="https://github.com/user-attachments/assets/1191b03d-5ab4-4d53-8a61-fe547aa1f852" alt="jira" style="width: 150px; height: 30;"> <img src="https://github.com/user-attachments/assets/264cdbcc-7c25-4525-b187-8e2debf06571" alt="slack" style="width: 90px; height: 30;">
|컴퓨터 비전 & 딥러닝|<img src="https://github.com/user-attachments/assets/e4c87c04-00cd-4869-ab0c-609d98c08e48" alt="opencv" style="width: 100px; height: 30;"> <img src="https://github.com/user-attachments/assets/3ddfb6ed-a0e6-46e8-b5c9-142a96eda3d9" alt="yolo8" style="width: 80px; height: 30;"> <img src="https://github.com/user-attachments/assets/9ef310af-a57f-4242-8dfe-c723591355ef" alt="movenet" style="width: 130px; height: 30;"> <img src="https://github.com/user-attachments/assets/4048bef5-36ae-4859-bc93-d45a08fa450d" alt="deepsort" style="width: 150px; height: 30;"> <img src="https://github.com/user-attachments/assets/66db1d2a-8971-4d2f-b581-3839d918cb42" alt="rekognition" style="width: 150px; height: 30;"> <img src="https://github.com/user-attachments/assets/be9077da-a3ba-4f5a-99cb-fe3e631219cb" alt="deeface" style="width: 110px; height: 30;">
|Database|<img src="https://github.com/user-attachments/assets/8d88dd9a-ff7d-4c33-a5ec-36a066b21685" alt="RDS" style="width: 90px; height: 30;">|

## 3. 딥러닝
### 3.1 전이 학습
### 3.1.1 Movenet - Fall Down Detection

#### Movenet
* 사람의 17개 주요 관절(코, 어깨, 엉덩이, 무릎 등) 실시간으로 추적하는 고속 추적 모델
* 영상 Dataset 으로부터 프레임을 추출하여 각 영상에 대한 CSV Label 정보를 사용하여 훈련 가능

![스크린샷 2024-09-28 165033](https://github.com/user-attachments/assets/d3829fe1-e736-420d-8792-71164a7a5b57)

*[Image Source: smilgate.ai](https://smilegate.ai/2021/05/20/movenet-a-javascript-pose-estimator/)*


#### 학습
* 영상 Dataset 을 사용, 각 영상에 대한 10개 class Label 을 부여하여 학습을 진행, 최종적으로 ‘Falling’ class 검출 사용 
* Data Size : 64 X 64 , 34,172 frames
* Class : Drinking, Eating, Exercising, Falling, Reading, Sitting, Sleeping, Standing, Unknown, Walking
* 학습 결과
[![image](https://github.com/user-attachments/assets/aae859de-bc7e-482e-8ad9-c22d3bdbb572)
](https://www.youtube.com/watch?v=3YcNn8kiVh8)
* 학습 결과 영상
* 
### 3.1.2 YOLOv8n - Wet Floor Detection
#### 학습

* 6,784장의 wet floor 이미지 사용하여 전이학습 진행
* 100 epoch 진행 시  최종 train/cls_loss : 0.79381  /   val/cls_loss : 0.93078 
* 높은 loss 대비 실제 영상에서 정확도 높은 검출 능력을 확인
* 학습 결과 영상
![스크린샷 2024-09-28 170448](https://github.com/user-attachments/assets/57895cd6-df4d-4931-97b8-b1002ad90e9d)

[![스크린샷 2024-09-28 171509](https://github.com/user-attachments/assets/a4a317d6-8776-4ba7-97b6-af6bdac29cc3)](https://www.youtube.com/watch?v=yUxwa1hmskQ)

### 3.1.3 YOLOv8s-seg - Clothes Segmentation
#### 학습
* 6,405장의 의상 이미지 사용하여 전이학습 진행
* class : 0: one-piece, 1: top, 2: under
* 200 epoch 진행 시  최종 train/cls_loss : 0.1886  /   val/seg_loss : 0.5023 
* 학습 결과 영상1
[![스크린샷 2024-09-28 170848](https://github.com/user-attachments/assets/3442e56d-91f6-44ab-b388-d19e9f3a842e)](https://www.youtube.com/watch?v=H0P24YpGDXE)

* 학습 결과 영상2
[![스크린샷 2024-09-28 171942](https://github.com/user-attachments/assets/0adec5ba-a576-4d0b-9665-4989d13e2af6)](https://www.youtube.com/watch?v=YqKes7pbTkk)


### 3.2 Face Detection & Recognition
#### YOLOv8-face
* YOLOv8 기반 얼굴 감지를 목적으로 훈련된 모델
* 작은 메모리 및 자원 소모, 실시간 처리에 적합
* 얼굴의 세부적인 특징, 임베딩 추출 불가

#### DeepFace
* 얼굴인식 관련  다양한 딥러닝 모델 제공 오픈 소스 라이브러리
* 얼굴 임베딩 추출, 감정 분석, 성별 예측 등의 기능 제공
* Local 작업 가능, 정확도 높은 분석 불가

#### AWS Rekognition
* Amazon Web Services 제공 클라우드 기반의 컴퓨터 비전 서비스
* 코사인 유사도를 사용해 두 임베딩의 유사도 (0~100%) 계산
* 높은 정확도, 클라우드 서비스의 실시간 사용 한계, 유료

* 얼굴 인식 프로세스 
![face1](https://github.com/user-attachments/assets/71177a0b-b1de-4edb-9d0f-f4966d36dff1)
*[Image Source: inkistyle.com](https://inkistyle.com/wp-content/uploads/2022/11/221112-IVE-Fashion-Incheon-Airport-1.jpg
)*

### 3.3 Rear Time DeepSort for Re-Identification
#### DeepSortTracker
* 객체 탐지 모델 + 임베딩 모델 -> 실시간 다중 객체 추적 및 Re-Identification(재식별)을 수행
* 각 객체에 고유 ID를 부여하고, 추적 및 재식별

#### 3.3.1 객체 탐지 : RetinaNet ResNet50 FPN v2
* RetinaNet은 이중 피라미드 네트워크(FPN) 구조를 통해 멀티스케일 객체 탐지
* Fast-RNN 대비 높은 정확도, 실시간성 확인

#### 3.3.2 객체 임베딩 : OSNet-AIN x1.0
* 다중 스케일의 특징을 학습하여 다양한 환경 변화와 시점 차이에도 강력한 재식별 성능 제공
* Adaptive Instance Normalization (AIN): 각 객체의 스타일 차이를 보정하여 일관된 특징 추출이 가능

#### DeepSortTracker 
* RetinaNet ResNet50 FPN v2 모델이 객체를 탐지하고, 바운딩 박스를 반환
* **임베딩 모델(OSNet-AIN)**이 각 객체의 특징 벡터를 생성하여 재식별 작업 수행
* DeepSort 알고리즘이 각 객체에 고유 ID를 부여하고, 지속적으로 추적
* 실시간으로 FPS를 측정하고, 결과를 시각적으로 표시
* 입력 받은 매 프레임 마다 의류 세그멘테이션을 수행
* id를 기준으로 하나의 정보로 묶어 딕셔너리 형태로 저장하여 DeepSort로 매겨진 ID를 가진 사람이 입은 상의와 하의의 색깔은 무엇이며, 그 사람은 프레임상 어디에  위치해있는지 알 수 있음.
  
<img src="https://github.com/user-attachments/assets/45ed3c35-6a4b-473e-9403-75985109ce99" alt="lego" style="width: 250px; height: auto;">|

## 4. 시스템 동작

![gui](https://github.com/user-attachments/assets/e25f3807-b5cf-4f65-b9c0-2201ab83c00f)

## 5. 동작영상 
[![스크린샷 2024-09-28 182248](https://github.com/user-attachments/assets/ea0bedaa-e362-4add-b5d7-f648ec6206fa)](https://www.youtube.com/watch?v=ptQEoJdGH7c)
[![스크린샷 2024-09-28 182218](https://github.com/user-attachments/assets/f9d7aa30-6603-44b1-875d-018ff2fb2884)](https://www.youtube.com/watch?v=swjwcJ5nP4I)