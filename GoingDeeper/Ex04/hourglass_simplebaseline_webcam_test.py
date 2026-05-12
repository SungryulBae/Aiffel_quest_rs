import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import time

R_ANKLE = 0
R_KNEE = 1
R_HIP = 2
L_HIP = 3
L_KNEE = 4
L_ANKLE = 5
PELVIS = 6
THORAX = 7
UPPER_NECK = 8
HEAD_TOP = 9
R_WRIST = 10
R_ELBOW = 11
R_SHOULDER = 12
L_SHOULDER = 13
L_ELBOW = 14
L_WRIST = 15

MPII_BONES = [
    [R_ANKLE, R_KNEE],
    [R_KNEE, R_HIP],
    [R_HIP, PELVIS],
    [L_HIP, PELVIS],
    [L_HIP, L_KNEE],
    [L_KNEE, L_ANKLE],
    [PELVIS, THORAX],
    [THORAX, UPPER_NECK],
    [UPPER_NECK, HEAD_TOP],
    [R_WRIST, R_ELBOW],
    [R_ELBOW, R_SHOULDER],
    [THORAX, R_SHOULDER],
    [THORAX, L_SHOULDER],
    [L_SHOULDER, L_ELBOW],
    [L_ELBOW, L_WRIST]
]
JOINT_NAMES = [
    "R_Ankle", "R_Knee", "R_Hip", "L_Hip", "L_Knee", "L_Ankle",
    "Pelvis", "Thorax", "Neck", "Head",
    "R_Wrist", "R_Elbow", "R_Shoulder", "L_Shoulder", "L_Elbow", "L_Wrist"
]
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=1, downsample=False):
        super(BottleneckBlock, self).__init__()
        self.downsample = downsample
        # 만약 downsample이라면 identity branch에 1x1 conv 적용하여 채널 수와 spatial size 조정
        if self.downsample:
            self.downsample_conv = nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False)

        # main branch
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, filters // 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn2 = nn.BatchNorm2d(filters // 2, momentum=0.9)
        # kernel_size=3, padding=1로 'same' padding 효과
        self.conv2 = nn.Conv2d(filters // 2, filters // 2, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(filters // 2, momentum=0.9)
        self.conv3 = nn.Conv2d(filters // 2, filters, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample_conv(x)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += identity
        return out

class HourglassModule(nn.Module):
    def __init__(self, order, filters, num_residual):
        super(HourglassModule, self).__init__()
        self.order = order

        # Up branch: BottleneckBlock 1회 + num_residual회 반복
        self.up1_0 = BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
        self.up1_blocks = nn.Sequential(*[
            BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
            for _ in range(num_residual)
        ])

        # Low branch: MaxPool + num_residual BottleneckBlock
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1_blocks = nn.Sequential(*[
            BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
            for _ in range(num_residual)
        ])

        # Recursive hourglass or additional BottleneckBlocks
        if order > 1:
            self.low2 = HourglassModule(order - 1, filters, num_residual)
        else:
            self.low2_blocks = nn.Sequential(*[
                BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
                for _ in range(num_residual)
            ])

        # 후처리 BottleneckBlock 반복
        self.low3_blocks = nn.Sequential(*[
            BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
            for _ in range(num_residual)
        ])

        # UpSampling (최근접 보간법)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # up branch
        up1 = self.up1_0(x)
        up1 = self.up1_blocks(up1)

        # low branch
        low1 = self.pool(x)
        low1 = self.low1_blocks(low1)
        if self.order > 1:
            low2 = self.low2(low1)
        else:
            low2 = self.low2_blocks(low1)
        low3 = self.low3_blocks(low2)
        up2 = self.upsample(low3)

        return up2 + up1
    
class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)

        # He (Kaiming) 초기화 적용
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class StackedHourglassNetwork(nn.Module):
    def __init__(self, input_shape=(256, 256, 3), num_stack=4, num_residual=1, num_heatmap=16):
        super(StackedHourglassNetwork, self).__init__()
        self.num_stack = num_stack

        in_channels = input_shape[2]  # 3
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)

        # Bottleneck blocks 초기화
        # BottleneckBlock의 첫번째 호출: 64 → 128, downsample=True
        self.bottleneck1 = BottleneckBlock(in_channels=64, filters=128, stride=1, downsample=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 두 번째: 128 → 128, downsample=False
        self.bottleneck2 = BottleneckBlock(in_channels=128, filters=128, stride=1, downsample=False)
        # 세 번째: 128 → 256, downsample=True
        self.bottleneck3 = BottleneckBlock(in_channels=128, filters=256, stride=1, downsample=True)

        # 스택 구성 요소들
        self.hourglass_modules = nn.ModuleList()
        self.residual_modules = nn.ModuleList()  # hourglass 후 residual block들 (num_residual회)
        self.linear_layers = nn.ModuleList()
        self.heatmap_convs = nn.ModuleList()
        # 마지막 스택을 제외한 중간 피쳐 결합용 1x1 conv
        self.intermediate_convs = nn.ModuleList()
        self.intermediate_outs = nn.ModuleList()

        for i in range(num_stack):
            # order=4인 hourglass 모듈 (앞에서 정의한 HourglassModule 사용)
            self.hourglass_modules.append(HourglassModule(order=4, filters=256, num_residual=num_residual))
            # hourglass 후 residual block들
            self.residual_modules.append(nn.Sequential(*[
                BottleneckBlock(in_channels=256, filters=256, stride=1, downsample=False)
                for _ in range(num_residual)
            ]))
            # Linear layer: 1x1 conv + BN + ReLU (앞에서 정의한 LinearLayer 사용)
            self.linear_layers.append(LinearLayer(in_channels=256, out_channels=256))
            # 최종 heatmap을 생성하는 1x1 conv
            self.heatmap_convs.append(nn.Conv2d(256, num_heatmap, kernel_size=1, stride=1, padding=0))

            if i < num_stack - 1:
                self.intermediate_convs.append(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.intermediate_outs.append(nn.Conv2d(num_heatmap, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.bottleneck1(x)
        x = self.pool(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)

        outputs = []
        for i in range(self.num_stack):
            hg = self.hourglass_modules[i](x)
            res = self.residual_modules[i](hg)
            lin = self.linear_layers[i](res)
            heatmap = self.heatmap_convs[i](lin)
            outputs.append(heatmap)

            if i < self.num_stack - 1:
                inter1 = self.intermediate_convs[i](lin)
                inter2 = self.intermediate_outs[i](heatmap)
                x = inter1 + inter2  # 다음 스택의 입력으로 사용

        return outputs

class SimplePoseNet(nn.Module):
    def __init__(self, backbone_name='resnet50', num_heatmap=16, pretrained=False):
        super(SimplePoseNet, self).__init__()
        # torchvision의 resnet 활용
        import torchvision.models as models
        resnet = getattr(models, backbone_name)(pretrained=pretrained)
        
        # ResNet의 마지막 layer4까지만 추출 (2048채널, 8x8 feature map @ 256x256 input)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # 3개의 Deconvolution 레이어 (해상도 4배 -> 32배로 복구)
        self.deconv_layers = nn.Sequential(
            self._make_deconv_layer(2048, 256),
            self._make_deconv_layer(256, 256),
            self._make_deconv_layer(256, 256)
        )
        # 최종 히트맵 생성용 1x1 Conv
        self.final_layer = nn.Conv2d(256, num_heatmap, kernel_size=1, stride=1, padding=0)

    def _make_deconv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

def find_max_coordinates(heatmaps):
    # heatmaps: (H, W, C)
    H, W, C = heatmaps.shape
    # reshape to (H*W, C)
    flatten_heatmaps = heatmaps.reshape(-1, C)
    # 각 채널 별 최대값 인덱스 (flattened index)
    indices = torch.argmax(flatten_heatmaps, dim=0)
    # y 좌표: index // H, x 좌표: index - H * y
    y = indices // H
    x = indices - H * y
    # 반환: (C, 2) 텐서, 각 행이 [x, y] 좌표
    return torch.stack([x, y], dim=1)


def extract_keypoints_from_heatmap(heatmaps):
    """
    heatmaps: (H, W, C) 텐서 (예: (64,64,16))
    """
    H, W, C = heatmaps.shape
    max_keypoints = find_max_coordinates(heatmaps)  # shape: (C, 2) with [x, y] per channel

    # pad heatmaps: 먼저 (C, H, W)로 변환한 후 pad, 다시 (H+2, W+2, C)
    heatmaps_permuted = heatmaps.permute(2, 0, 1)  # (C, H, W)
    padded = F.pad(heatmaps_permuted, (1, 1, 1, 1))  # pad (left, right, top, bottom)
    padded_heatmaps = padded.permute(1, 2, 0)  # (H+2, W+2, C)

    adjusted_keypoints = []
    for i, keypoint in enumerate(max_keypoints):
        # 기존 keypoint의 좌표에 패딩 오프셋 추가
        max_x = int(keypoint[0].item()) + 1
        max_y = int(keypoint[1].item()) + 1

        # 3x3 패치를 추출 (채널 i)
        patch = padded_heatmaps[max_y-1:max_y+2, max_x-1:max_x+2, i]  # (3,3)
        # 중앙 값 제거
        patch[1, 1] = 0
        # 패치 내 최대값의 index를 찾음
        flat_patch = patch.reshape(-1)
        index = torch.argmax(flat_patch).item()

        next_y = index // 3
        next_x = index % 3
        delta_y = (next_y - 1) / 4.0
        delta_x = (next_x - 1) / 4.0

        adjusted_x = keypoint[0].item() + delta_x
        adjusted_y = keypoint[1].item() + delta_y
        adjusted_keypoints.append((adjusted_x, adjusted_y))

    # 리스트를 텐서로 변환하고 clip
    adjusted_keypoints = torch.tensor(adjusted_keypoints)
    adjusted_keypoints = torch.clamp(adjusted_keypoints, 0, H)
    normalized_keypoints = adjusted_keypoints / H
    return normalized_keypoints


# 전처리 파이프라인 정의
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),      # 모델 입력 크기에 맞게 조절 
    transforms.ToTensor(),              # 0~1 사이 값으로 변환 및 텐서화
    transforms.Normalize(               # ImageNet 정규화 값 적용
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def get_fps(model, input_tensor, device):
    # 1. GPU 연산이 끝날 때까지 대기 (CPU라면 무시됨)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter() # 정밀 타이머 사용
    
    with torch.no_grad():
        output = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    # 계산된 시간 차이
    dt = end_time - start_time
    
    # 핵심: dt가 너무 작으면(0 포함) 아주 작은 최소값으로 강제 고정
    # 0.001초(1ms)보다 작으면 1ms로 처리해서 최대 FPS를 1000으로 제한
    dt = max(dt, 0.001) 
    
    return output, 1.0 / dt

def run_comparison_side_by_side(model_hg, model_sb, mpii_bones, device):
    cap = cv2.VideoCapture(0)
    
    # FPS 부드럽게 표시하기 위한 변수
    avg_fps_hg = 0
    avg_fps_sb = 0
    smoothing = 0.9

    while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            input_tensor = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)

            # --- Hourglass 추론 및 시간 측정 (정밀화) ---
            if device.type == 'cuda': torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.no_grad():
                out_hg = model_hg(input_tensor)
                if isinstance(out_hg, list): out_hg = out_hg[-1] # 리스트 에러 방지
            if device.type == 'cuda': torch.cuda.synchronize()
            dt_hg = time.perf_counter() - t1
            
            # FPS 폭주 방지: dt가 너무 작으면 최소 0.001초로 고정
            curr_fps_hg = 1.0 / max(dt_hg, 0.001) 
            avg_fps_hg = avg_fps_hg * smoothing + curr_fps_hg * (1 - smoothing)

            # --- Simple Baselines 추론 및 시간 측정 (정밀화) ---
            if device.type == 'cuda': torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.no_grad():
                out_sb = model_sb(input_tensor)
            if device.type == 'cuda': torch.cuda.synchronize()
            dt_sb = time.perf_counter() - t2
            
            curr_fps_sb = 1.0 / max(dt_sb, 0.001)
            avg_fps_sb = avg_fps_sb * smoothing + curr_fps_sb * (1 - smoothing)

            # --- 좌표 추출 ---
            kp_hg = extract_keypoints_from_heatmap(out_hg.squeeze(0).cpu().permute(1, 2, 0))
            kp_sb = extract_keypoints_from_heatmap(out_sb.squeeze(0).cpu().permute(1, 2, 0))

            # --- 시각화 함수 (0,0 필터링 강화) ---
            def draw_refined(canvas, kps, color, label, fps):
                h, w, _ = canvas.shape
                # 임계값을 0.05 정도로 설정하여 좌상단(0,0) 근처의 부실한 점들 제거
                min_cutoff = 0.05 

                for bone in mpii_bones:
                    p1, p2 = kps[bone[0]], kps[bone[1]]
                    # 두 점이 모두 화면 구석(0,0)을 벗어나 유효할 때만 선을 그림
                    if p1[0] > min_cutoff and p1[1] > min_cutoff and p2[0] > min_cutoff and p2[1] > min_cutoff:
                        cv2.line(canvas, (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h)), color, 2, cv2.LINE_AA)

                for p in kps:
                    if p[0] > min_cutoff and p[1] > min_cutoff:
                        cv2.circle(canvas, (int(p[0]*w), int(p[1]*h)), 4, (255, 255, 255), -1)

                # FPS 배경 박스
                cv2.rectangle(canvas, (5, 5), (260, 85), (0, 0, 0), -1)
                cv2.putText(canvas, f"{label}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(canvas, f"FPS: {fps:.1f}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            canvas_hg = frame.copy()
            canvas_sb = frame.copy()
            draw_refined(canvas_hg, kp_hg, (255, 0, 0), "Hourglass", avg_fps_hg)
            draw_refined(canvas_sb, kp_sb, (0, 255, 255), "SimpleBaseline", avg_fps_sb)

            cv2.imshow('Final Comparison', cv2.hconcat([canvas_hg, canvas_sb]))
            if cv2.waitKey(1) == ord('q'): break




# 1. 경로 및 설정값 정의
PROJECT_PATH = 'mpii'
num_heatmap = 16

# 가중치 경로
HG_WEIGHTS = os.path.join(PROJECT_PATH, 'models', 'model-epoch-5-loss-1.1280.pt')
SB_WEIGHTS = os.path.join(PROJECT_PATH, 'models','simpleBaseline', 'simple_baseline_epoch-5-loss-0.2743.pt')

# 장치(Device) 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 선언 및 로드
model_hg = StackedHourglassNetwork(num_stack=4, num_heatmap=16).to(device)
model_hg.load_state_dict(torch.load(HG_WEIGHTS, map_location=device))

model_sb = SimplePoseNet(num_heatmap=16).to(device)
model_sb.load_state_dict(torch.load(SB_WEIGHTS, map_location=device))

# 비교 실행
run_comparison_side_by_side(model_hg, model_sb, MPII_BONES, device)