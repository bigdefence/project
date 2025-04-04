from TTS.api import TTS
import os
import torch

# torch.load를 임시로 재정의하여 weights_only=False로 설정
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)

# XTTS-v2 모델 초기화 (GPU 사용)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# 한국어 텍스트
text = "안녕하세요, XTTS-v2를 우분투에서 테스트 중입니다!"
output_file = "output/korean_test_output.wav"

# 출력 디렉토리 생성
os.makedirs("output", exist_ok=True)

# 옵션 1: 기본 제공 화자 사용
# 사용 가능한 화자는 모델 로드 후 확인 가능 (tts.synthesizer.tts_model.speaker_manager.speakers)
speaker = "Ana Florence"  # 예시 화자 이름, 실제 사용 가능 화자는 모델 문서나 테스트로 확인 필요

# 한국어로 음성 생성 (기본 화자 사용)
tts.tts_to_file(text=text, file_path=output_file, language="ko", speaker=speaker)
print(f"음성 파일이 {output_file}에 저장되었습니다.")

# 옵션 2: 특정 음성 복제 (주석 처리됨, 필요 시 사용)
# speaker_wav = "/path/to/your/speaker.wav"  # 복제할 음성 파일 경로
# tts.tts_to_file(text=text, file_path=output_file, language="ko", speaker_wav=speaker_wav)

# torch.load를 원래 상태로 복구
torch.load = original_load
