## ECOBIT Solar Dashboard
부산 버스정류장의 태양광 발전 여건을 비교해 ECOBIT 버스정보안내기(BIT) 설치 후보지를 사전 검토하는 웹 대시보드입니다. 지자체·운영기관과의 상담에서 후보지를 좁히고 현장 실측 우선순위를 정하는 데 활용할 수 있습니다.

### 대시보드 열기
https://bravo-jino.github.io/ecobit-solar-dashboard/

주요 기능
- 부산시 구·군별 정류장 탐색 및 정류장명·번호 검색
- 정류장별 월별 일평균 예상 발전량과 태양 경로 확인
- 위성 1 m 수관 높이(Meta & WRI CHMv2) 기반 가로수·주변 수목 음영 전수 필터링 및 2D/3D·계절별 발전량 보정
- 배터리 용량과 일일 소비전력에 따른 예상 잔량 시뮬레이션

### 수목 데이터 갱신 (전수 자동화)
```bash
uv run --with rasterio --with pyproj --with numpy --with scipy scripts/build_tree_data.py
```
AWS Open Data 공개 타일(Meta & WRI Canopy Height Maps v2, CC BY 4.0)에서 부산 전 정류장의 반경 30 m 수관 차폐 고도와 28 m 수관 클러스터를 자동 추출해 `tree-data.js`를 생성합니다.

시뮬레이션 결과는 사전 검토용 추정치입니다. 최종 설치 여부는 실제 패널 설치 위치에서 일조와 음영 조건을 확인하고 현장 실측을 거쳐 판단합니다.
