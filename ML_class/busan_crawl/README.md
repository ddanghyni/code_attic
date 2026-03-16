# 부산 카페/음식점 네이버 지도 크롤러

부산 지역 카페/음식점의 평점, 리뷰 수, 메뉴(가격), 업종, 주소 등을 네이버 지도에서 수집합니다.

## 프로젝트 구조

```
busan_crawl/
├── crawler.py          # 메인 크롤러 (네이버 지도 크롤링)
├── preprocess.py       # 전처리 스크립트 (가격 중앙값, 가격대 범주화 등)
├── requirements.txt    # 패키지 목록
├── README.md           # 이 파일
├── output/             # (자동 생성) 크롤링 원본 데이터
│   ├── restaurants_YYYYMMDD_HHMMSS.csv
│   ├── menus_YYYYMMDD_HHMMSS.csv
│   └── raw_data_YYYYMMDD_HHMMSS.json
└── processed/          # (자동 생성) 전처리된 분석용 데이터
    ├── all_stores.csv
    ├── analysis_data.csv
    └── menus_clean.csv
```

## 수집 데이터 항목

### 가게 테이블 (restaurants)
| 컬럼 | 설명 | 예시 |
|---|---|---|
| place_id | 네이버 고유 ID | 1234567890 |
| name | 상호명 | 맛있는집 |
| category | 업종 (네이버 분류) | 한식 |
| rating | 평점 (별점) | 4.32 |
| visitor_review_count | 방문자 리뷰 수 | 152 |
| blog_review_count | 블로그 리뷰 수 | 45 |
| address | 전체 주소 | 부산 해운대구 우동 ... |
| district | 행정구 | 해운대구 |
| business_status | 영업 상태 | 영업중 |
| menu_count | 메뉴 수 | 12 |
| search_query | 검색어 | 부산 해운대구 맛집 |
| crawl_date | 크롤링 날짜 | 2026-03-13 |

### 메뉴 테이블 (menus)
| 컬럼 | 설명 | 예시 |
|---|---|---|
| place_id | 네이버 고유 ID | 1234567890 |
| store_name | 상호명 | 맛있는집 |
| menu_name | 메뉴명 | 된장찌개 |
| price | 가격 (원) | 8000 |

## 설치 방법

### 1단계: Python 확인

Python 3.9 이상이 필요합니다.

```bash
python --version
# 또는
python3 --version
```

Python이 없다면: https://www.python.org/downloads/ 에서 설치

### 2단계: 프로젝트 폴더 준비

원하는 위치에 프로젝트 폴더를 만들고, 파일들을 넣어주세요.

```bash
mkdir busan_crawl
cd busan_crawl
# crawler.py, preprocess.py, requirements.txt 파일을 이 폴더에 넣기
```

### 3단계: 가상환경 생성 (권장)

가상환경을 만들면 다른 프로젝트와 패키지가 충돌하지 않아요.

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# 활성화되면 터미널 앞에 (venv) 표시가 뜹니다
```

### 4단계: 패키지 설치

```bash
pip install -r requirements.txt
```

### 5단계: Playwright 브라우저 설치

Playwright는 실제 브라우저(Chromium)를 제어해서 크롤링합니다. 브라우저를 별도로 설치해야 해요.

```bash
playwright install chromium
```

이 과정에서 Chromium 브라우저가 다운로드됩니다 (약 200MB).

> 만약 `playwright install`에서 에러가 나면:
> ```bash
> # 시스템 의존성도 설치 (Linux만 해당)
> playwright install-deps chromium
> ```

## 실행 방법

### 크롤링 실행

```bash
python crawler.py
```

실행하면 3가지 모드를 선택할 수 있어요:

```
실행 모드를 선택하세요:
  1) 전체 크롤링 (부산 16개 구·군 × 맛집/카페)
  2) 특정 지역만 크롤링
  3) 직접 검색어 입력
```

#### 모드 1: 전체 크롤링
- 부산 16개 행정구 × 2개 키워드(맛집, 카페) = 32개 검색어
- 소요 시간: 약 3~5시간 (네트워크, 데이터 양에 따라 다름)
- **권장: 처음에는 모드 2로 1~2개 구만 테스트 후 전체 실행**

#### 모드 2: 특정 지역만 (테스트용 추천)
```
부산 행정구 목록:
   1) 해운대구
   2) 부산진구
   ...

크롤링할 구 번호 (쉼표로 구분, 예: 1,2,5): 1,2
```

#### 모드 3: 직접 검색어 입력
```
검색어를 입력하세요: 부산 서면 맛집
```

### 전처리 실행

크롤링이 끝난 후 실행합니다.

```bash
python preprocess.py
```

이 스크립트가 하는 일:
1. `output/` 폴더에서 가장 최근 크롤링 데이터를 로드
2. 폐업/휴업 가게 제거
3. 메뉴 가격 중앙값 계산 → 가게 테이블에 병합
4. 가격대 범주화 (저가/중가/고가)
5. 업종 대분류 (카페/음식점)
6. `processed/` 폴더에 분석용 데이터 저장

## 출력 파일 설명

### output/ (크롤링 원본)
- `restaurants_YYYYMMDD_HHMMSS.csv`: 가게 정보 원본
- `menus_YYYYMMDD_HHMMSS.csv`: 메뉴 정보 원본
- `raw_data_YYYYMMDD_HHMMSS.json`: JSON 백업
- `*_intermediate.csv`: 크롤링 중간 저장 (안전장치)

### processed/ (전처리 후 - 분석에 사용)
- `analysis_data.csv`: **분석용 최종 데이터** (평점 있는 가게만)
- `all_stores.csv`: 전체 가게 (평점 없는 것 포함)
- `menus_clean.csv`: 정리된 메뉴 데이터

## 주의사항

### 크롤링 관련
- **요청 간격**: 코드에 1~2.5초 랜덤 딜레이가 설정되어 있습니다. 너무 빠르게 하면 IP 차단될 수 있어요.
- **IP 차단 시**: 30분~1시간 기다린 후 재실행, 또는 VPN 사용
- **중간 저장**: 10개 검색마다 자동 저장되므로, 중간에 오류가 나도 데이터가 보존됩니다.
- **headless 모드**: `NaverMapCrawler(headless=True)`로 변경하면 브라우저 창 없이 실행 (속도 약간 빠름)

### 네이버 지도 CSS 셀렉터 관련
- 네이버 지도의 HTML 구조(CSS 클래스명)는 네이버가 업데이트하면 바뀔 수 있습니다.
- 크롤링이 제대로 안 되면 (데이터가 0개 등), 네이버 지도를 직접 열어서 개발자 도구(F12)로 셀렉터를 확인해야 할 수 있습니다.
- 주요 셀렉터:
  - 검색결과 목록: `li.UEzoS`
  - 가게명: `span.TYaxT`
  - 업종: `span.KCMnt`
  - 평점: `span.h69bs`
  - 주소 (상세페이지): `span.LDgIH`

### 데이터 관련
- 모든 가게에 평점이 있는 것은 아닙니다 (신규 가게 등). 전처리 시 필터링됩니다.
- 메뉴 가격이 없는 가게도 많습니다. 이 경우 가격 관련 분석에서 제외됩니다.
- 같은 가게가 여러 검색어에서 나올 수 있는데, place_id 기반으로 자동 중복 제거됩니다.

## 가격대 기준 (preprocess.py에서 수정 가능)

| 가격대 | 기준 (메뉴 가격 중앙값) |
|---|---|
| 저가 | ~10,000원 |
| 중가 | 10,001원 ~ 25,000원 |
| 고가 | 25,001원~ |

이 기준은 `preprocess.py`의 `categorize_price()` 함수에서 수정할 수 있습니다. 크롤링 후 데이터 분포를 보고 사분위수 기반으로 조정하는 것을 권장합니다.

## 트러블슈팅

### "playwright install" 실패
```bash
# pip 업그레이드 후 재시도
pip install --upgrade pip
pip install playwright
playwright install chromium
```

### "TimeoutError" 자주 발생
- 인터넷 연결 확인
- TIMEOUT 값을 `crawler.py` 상단에서 15000 → 30000으로 늘려보기
- headless=False로 실행해서 브라우저가 실제로 어디서 멈추는지 확인

### 데이터가 0개 수집됨
- 네이버 지도 CSS 셀렉터가 변경되었을 수 있음
- headless=False로 실행해서 브라우저 화면 직접 확인
- 검색 결과가 실제로 있는 검색어인지 확인

### "ModuleNotFoundError: No module named 'playwright'"
```bash
# 가상환경이 활성화되어 있는지 확인
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install playwright
playwright install chromium
```
