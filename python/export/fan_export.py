import face_alignment
import torch

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cuda', face_detector='blazeface', face_detector_kwargs={'back_model': True})

# 내부 PyTorch 모델 가져오기
model = fa.face_alignment_net

# pt 파일로 저장
torch.save(model.state_dict(), "fan_3d_gpu.pt")