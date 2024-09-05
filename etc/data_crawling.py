import os
import time
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# 검색할 키워드 목록
keywords = [
    "T-shirt", 
    "Short-sleeve shirt", 
    "Polo shirt"
]

# 원본 데이터를 저장할 디렉토리 경로
source_dir = "crawled_images"
os.makedirs(source_dir, exist_ok=True)

# 사용자 에이전트 설정 (필요에 따라 수정)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

def download_image(image_url, save_path, min_size=(100, 100)):
    try:
        response = requests.get(image_url, headers=headers)
        image = Image.open(BytesIO(response.content))

        # 로고나 작은 이미지를 제외하기 위해 크기 체크
        if image.size[0] < min_size[0] or image.size[1] < min_size[1]:
            print(f"Skipped small image: {image.size}")
            return False

        image.save(save_path, "PNG")
        print(f"Image saved at {save_path}")
        return True
    except Exception as e:
        print(f"Failed to save image {image_url}: {e}")
        return False

def crawl_images_for_keyword(keyword):
    # Selenium 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    search_url = f"https://www.google.com/search?tbm=isch&q={keyword.replace(' ', '+')}"
    driver.get(search_url)

    image_urls = set()
    scroll_attempts = 0
    max_scroll_attempts = 10  # 최대 스크롤 시도 횟수

    while len(image_urls) < 100 and scroll_attempts < max_scroll_attempts:
        # 스크롤하여 새로운 이미지를 로드
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # 페이지 소스 가져오기
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        image_tags = soup.find_all("img", {"src": True})

        for img_tag in image_tags:
            img_url = img_tag.get("src")
            if img_url and img_url.startswith("http"):
                image_urls.add(img_url)
                if len(image_urls) >= 100:
                    break

        scroll_attempts += 1

    # 드라이버 종료
    driver.quit()

    # 이미지 다운로드
    for i, img_url in enumerate(image_urls):
        img_name = f"{keyword.replace(' ', '_')}_{i}.png"
        img_save_path = os.path.join(source_dir, img_name)

        download_image(img_url, img_save_path)

def main():
    for keyword in keywords:
        print(f"Crawling images for keyword: {keyword}")
        crawl_images_for_keyword(keyword)

if __name__ == "__main__":
    main()
