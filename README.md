# EyeTrackingDemo

few-shot gaze를 이용하여 EyeTracking 을 web에서 실행할 수 있도록 하는 프로그램이다. 실시간으로 영상을 주고받아야 하기 때문에 websocket을 사용함


## Release

|날짜|내용
|---|---|
|2021.07.10| 기존에 Linux에서만 작동하던 monitor.py를 Windows 환경에서 사용할 수 있도록 코드 수정
|2021.10.17| websocket을 이용하여 Client의 webcam을 받아 Calibrate 및 EyeTracking Server 일부를 구현
|2021.10.18| weboscket을 이용하여 Client의 webcam을 받아 Data collect 및 Train Server 일부를 구현
|2021.10.20| Data Collect Server에서 frame을 받을 때 바로 data로 변환하게 수정, 일부 바뀐 함수 수정
|2021.10.21| Data Collect Server에서 만드는 데이터의 수를 늘려서 받는것에 맞추어 수정

## References

Few-shot-gaze : https://github.com/NVlabs/few_shot_gaze


## Memo

#### 2021.10.20
~~1. Collect Data부분이 내가 생각했던 것과 다름. 내가 생각했던 것은 그 점과 그 점을 바라보는 사용자의 카메라를 하나만 사용하는 줄 알았는데 버튼을 누를때 까지 frame을 받고 버튼을 누르면 뒤에서 10개의 frame을 데이터로 사용함. 그래서 총 점의 개수 * 10개의 데이터를 Fine-Tuning함~~(2021.10.21 해결)


2. 좌표를 받고 monitor_to_camera 함수를 적용해야함. + 화면 가운데가 0임
