import os
import cv2
import torch
import argparse
import numpy as np
from model.model import Model


# =====================================
#  Utility: 이미지 읽기 및 텐서 변환
# =====================================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (torch.from_numpy(img).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
    return img


# =====================================
#  Main Function
# =====================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', nargs=2, required=True, help='보간할 이미지 두 장 (frame0, frame1)')
    parser.add_argument('--multiplier', type=int, default=2, help='보간 배수 (2=2배, 3=3배, 4=4배)')
    parser.add_argument('--model', type=str, default='train_log', help='모델 가중치 폴더 경로')
    args = parser.parse_args()

    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # 출력 폴더
    os.makedirs('output', exist_ok=True)

    # 이미지 로드
    img0 = load_image(args.img[0]).to(device)
    img1 = load_image(args.img[1]).to(device)

    # 모델 로드
    model = Model().to(device)
    ckpt_path = os.path.join(args.model, 'flownet.pkl')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ERROR] 모델 가중치를 찾을 수 없습니다: {ckpt_path}")

    model.flownet.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[INFO] 모델 로드 완료: {ckpt_path}")

    # =====================================
    #  보간 시점 계산
    # =====================================
    n = args.multiplier
    if n < 2:
        raise ValueError("[ERROR] multiplier 값은 2 이상이어야 합니다.")
    time_list = [(i + 1) / n for i in range(n - 1)]
    print(f"[INFO] 보간 시점: {time_list}")

    # =====================================
    #  보간 실행 루프
    # =====================================
    torch.cuda.empty_cache()
    for idx, t in enumerate(time_list):
        print(f"[INFO] {idx+1}/{len(time_list)} 프레임 보간 중... (t={t:.3f})")
        with torch.no_grad():
            out = model.inference(img0, img1, timestep=float(t))
            out_img = (out[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            out_path = os.path.join('output', f'img{idx+1}.png')
            cv2.imwrite(out_path, out_img)
            print(f"[SAVED] {out_path}")

    print("[DONE] 모든 보간 프레임 저장 완료 ✅")


if __name__ == '__main__':
    main()
