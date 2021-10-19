# EyeTrackingDemo

few-shot gaze를 이용하여 EyeTracking 을 web에서 실행할 수 있도록 하는 프로그램이다. 실시간으로 영상을 주고받아야 하기 때문에 websocket을 사용함


## Issue

|날짜|내용
|---|---|
|2021.07.10| 기존에 Linux에서만 작동하던 monitor.py를 Windows 환경에서 사용할 수 있도록 코드 수정
|2021.10.17| websocket을 이용하여 Client의 webcam을 받아 Calibrate 및 EyeTracking Server 일부를 구현
|2021.10.18| weboscket을 이용하여 Client의 webcam을 받아 Data collect 및 Train Server 일부를 구현

## References

Few-shot-gaze : https://github.com/NVlabs/few_shot_gaze
