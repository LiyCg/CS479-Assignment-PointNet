# CS479 Assignment 1 - PointNet 제출 노트

## 구현 시 주의사항

### 코드 수정 규칙 (중요 - 감점 방지)
- **TODO 구역 밖의 코드는 절대 수정하지 않음**
- README에 명시: "you modify any code outside of the section marked with TODO → zero score"
- git diff로 TODO 구역 외 변경사항 없음 확인 완료

### 구현 파일
- `pointnet/model.py` — PointNetFeat, PointNetCls, PointNetPartSeg 구현
- `pointnet/train_cls.py` — step() 함수 구현 (cross_entropy + orthogonal loss)
- `pointnet/train_seg.py` — step() 함수 구현 (cross_entropy + orthogonal loss)

### 달성 성능
| Task | 결과 | 만점 기준 | 판정 |
|------|------|-----------|------|
| Classification (Acc) | 80.88% | 75% | 만점 ✓ |
| Segmentation (mIoU) | 81.48% | 70% | 만점 ✓ |

---

## 제출물 정리 (README Submission Guidelines 준수)

### 제출 파일 목록
1. `model.py`, `train_cls.py`, `train_seg.py` — 구현 코드 3개
2. `20265334_cls.ckpt` — Classification best checkpoint (epoch12, acc 80.88%)
3. `20265334_seg.ckpt` — Segmentation best checkpoint (epoch14, mIoU 81.48%)
4. `inyuplee_20265334.pdf` — 학습 종료 스크린샷 (cls + seg 각 1장)

### 최종 압축 파일
- 파일명: `inyuplee_20265334.zip` (26MB)
- 위치: `/source/inyup/CS479-Assignment-PointNet/`

### .gitignore 처리
- epoch별 기존 ckpt (`checkpoints/classification/**`, `checkpoints/segmentation/**`) → ignore
- renamed ckpt (`checkpoints/20265334_cls.ckpt`, `checkpoints/20265334_seg.ckpt`) → 추적

---

## GitHub 설정
- Fork: `https://github.com/LiyCg/CS479-Assignment-PointNet.git`
- git config: `LiyCg / leeinyup123@gmail.com`

---

## 다음 Assignment 체크리스트

1. GitHub에서 원본 레포 fork → `https://github.com/LiyCg/{repo}.git`
2. `git remote set-url origin https://github.com/LiyCg/{repo}.git`
3. `git config user.name "LiyCg" && git config user.email "leeinyup123@gmail.com"`
4. TODO 구역 안에서만 코드 수정 (`git diff`로 반드시 확인)
5. 제출 전 성능 기준 확인 (README Grading 섹션)
6. 체크포인트 rename: `{학번}_cls.ckpt`, `{학번}_seg.ckpt`
7. PDF: 학습 종료 터미널 스크린샷 2장 (cls + seg)
8. 최종 zip: `이름_학번.zip`
