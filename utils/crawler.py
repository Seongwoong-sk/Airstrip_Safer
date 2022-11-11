## 👉 Web Crawler ## 

# Libraries
import os
import re 
import time
import socket
import argparse
import urllib.request
from PIL import Image
from tqdm import tqdm
from pygame import mixer
from datetime import date

from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor

from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    ElementNotInteractableException,)
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains



def parse_args():
    parser = argparse.ArgumentParser(
        description='Web Image Crawler')
    parser.add_argument('--path', default='C:/Users/rlatj/Desktop/Crawled_Images/', help='Directory to save images')
    parser.add_argument('--webdriver',default= "C://chromedriver.exe", help='Chromedriver path')
    parser.add_argument('--mp3',default='./sound.mp3',help='mp3 path')
    parser.add_argument('--filter_size',type=int, help='Minimum image size to keep')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    class Crawler:
        def __init__(self):
            
            #  이미지들이 저장될 경로 및 폴더 이름
            self.path = args.path
            
            # 검색어 입력 및 중복 검사
            self.query = input('검색 키워드 입력 : ')
            
            # 웹 브라우저의 개발자 모드(F12)를 열어 console에 navigator.userAgent라고 입력 후 출력되는 값을 복사
            user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
            opts = Options()
            opts.add_argument(f"user-agent={user_agent}")
            opts.add_argument('headless')  # 창 띄울지 말지 
            opts.add_argument('window-size=1920x1080')
            opts.add_argument('disable-gpu')
            opts.add_argument('ignore-certificate-errors')
            
            # 드라이버 생성
            self.driver = webdriver.Chrome(args.webdriver, options=opts)
            
            # clickAndRetrieve() 과정에서 urlretrieve이 너무 오래 걸릴 경우를 대비해 타임 아웃 지정
            socket.setdefaulttimeout(30)
            
            # 크롤링한 이미지 수 
            self.crawled_count = 0
            
            # mp3 파일 경로
            self.mp3 = args.mp3

    #####################################################################################################################
        def scroll_down(self):
            scroll_count = 0
            
            print("-- 스크롤 다운 시작 --")
            
            # 스크롤 위치값 얻고 last_height에 저장
            last_height = self.driver.execute_script("return document.body.scrollHeight")
        
            # '결과 더보기' 버튼을 클릭했는 지 유무
            after_click = False
            
            while True:
                print(f"-- 스크롤 횟수 : {scroll_count} --")
                
                # 스크롤 다운
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                scroll_count += 1
                time.sleep(1.5)
                
                # 스크롤 위치값 얻고 new_height 에 저장
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                
                # 스크롤이 최하단이며
                if last_height == new_height:
                    
                    # '결과 더보기' 버튼을 클릭한 적이 있는 경우
                    if after_click is True:
                        print(" -- 스크롤 다운 종료 --")
                        break
                    
                    # '결과 더보기' 버튼을 클릭한 적이 없는 경우
                    elif after_click is False:
                        if self.driver.find_element_by_css_selector(".mye4qd").is_displayed():
                            self.driver.find_element_by_css_selector(".mye4qd").click()
                            print("-- '결과 더보기' 버튼 클릭 --")
                            after_click = True
                        elif NoSuchElementException:
                            print(' -- NoSuchElementException --')
                            print(' -- 스크롤 다운 종료 --')
                            break
                last_height = new_height
                
    #####################################################################################################################
        
        def click_and_retrieve(self,index, img, img_list_length):
            
            try:
                ActionChains(self.driver).click(img).perform()
                time.sleep(1.5)
                self.driver.implicitly_wait(3)
                imgurl = self.driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute("src")
                
                # 확장자 
                if re.search(r"jpeg|png",imgurl):
                    ext = re.search(r"jpeg|png", imgurl).group()
                else:
                    ext = 'jpg'
                
                # 저장될 이미지 파일 경로
                dst = os.path.join(self.path,self.query,f"{self.query}_{self.crawled_count+1}.{ext}")
                self.crawled_count += 1
                
                urlretrieve(imgurl, f"{dst}")
                self.driver.implicitly_wait(3)
                print(f"{index + 1} / {img_list_length} 번째 사진 저장 {ext}")
                
            except HTTPError:
                print("ㅡ HTTPError & 패스 ㅡ")
                pass

    #####################################################################################################################
            
        def crawling(self):
            print('-- 크롤링 시작 --')
            
            self.driver.get("https://www.google.co.kr/imghp?hl=ko&authuser=0&ogbl")
            time.sleep(2)
            self.driver.find_element_by_css_selector("input.gLFyf").send_keys(self.query) #send keyword
            self.driver.find_element_by_css_selector("input.gLFyf").send_keys(Keys.RETURN)##send Keys.RETURN
            
            self.scroll_down()
            
            # class_name에 공백이 있는 경우 여러 클래스가 있는 것이므로 아래와 같이 css_selector로 찾음
            img_list = self.driver.find_elements_by_css_selector("img.rg_i.Q4LuWd") # 썸네일
            
            # 디렉토리 생성
            directory_path = os.path.join(self.path, self.query)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                print(f'{directory_path} 폴더 생성')
            else : print(f"{directory_path} Directory Already Exists")
            
            for index, img in enumerate(img_list):
                try:
                    self.click_and_retrieve(index, img, len(img_list))

                except ElementClickInterceptedException:
                    print("ㅡ ElementClickInterceptedException ㅡ")
                    self.driver.execute_script("window.scrollTo(0, window.scrollY + 100)")
                    print("ㅡ 100만큼 스크롤 다운 및 3초 슬립 ㅡ")
    #                 img.click()
                    ActionChains(self.driver).click(img).perform()
                    time.sleep(3)
                    self.click_and_retrieve(index, img, len(img_list))

                except NoSuchElementException:
                    print("ㅡ NoSuchElementException ㅡ")
                    self.driver.execute_script("window.scrollTo(0, window.scrollY + 100)")
                    print("ㅡ 100만큼 스크롤 다운 및 4초 슬립 ㅡ")
                    time.sleep(4)
    #                 img.click()
                    ActionChains(self.driver).click(img).perform()
                    self.click_and_retrieve(index, img, len(img_list))

                except ConnectionResetError:
                    print("ㅡ ConnectionResetError & 패스 ㅡ")
                    pass

                except URLError:
                    print("ㅡ URLError & 패스 ㅡ")
                    pass

                except socket.timeout:
                    print("ㅡ socket.timeout & 패스 ㅡ")
                    pass

                except socket.gaierror:
                    print("ㅡ socket.gaierror & 패스 ㅡ")
                    pass

                except ElementNotInteractableException:
                    print("ㅡ ElementNotInteractableException ㅡ")
                    break

            try:
                print("ㅡ 크롤링 종료 (성공률: %.2f%%) ㅡ" % (self.crawled_count / len(img_list) * 100.0))

            except ZeroDivisionError:
                print("ㅡ img_list 가 비어있음 ㅡ")

            self.driver.quit()

    #####################################################################################################################

        def filtering(self, width_threshold, height_threshold):
            print("ㅡ 필터링 시작 ㅡ")
            filtered_count = 0
            dir_name = os.path.join(self.path, self.query)
            
            for index, file_name in enumerate(os.listdir(dir_name)):
                try:
                    file_path = os.path.join(dir_name, file_name)
                    ext = file_name.split('.')[-1]
                    img = Image.open(file_path)

                    # 이미지 해상도의 가로와 세로가 모두 800이하인 경우
                    if (img.width < width_threshold and img.height < height_threshold):
                        img.close()
                        os.remove(file_path)
                        print(f"{index} 번째 사진 삭제 -->  width : {img.width} height : {img.height}")
                        filtered_count += 1

                # 이미지 파일이 깨져있는 경우
                except OSError:
                    os.remove(file_path)
                    filtered_count += 1

            print(f"ㅡ 필터링 종료 (총 갯수: {self.crawled_count - filtered_count}) ㅡ")
                    
    #####################################################################################################################      
        def change_extension(self):
            print("ㅡ 필터링 시작 ㅡ")
            changed_count = 0
            dir_name = os.path.join(self.path, self.query)
            
            for index, file_name in enumerate(os.listdir(dir_name)):
                ext = file_name.split('.')[-1]
                img_name = file_name.split('.')[0]
                if ext != 'jpg':
                    os.rename(os.path.join(dir_name,file_name), os.path.join(dir_name,img_name+'.jpg'))
                    print(f"{index} 번째 사진 확장자 변경 -->  before : {ext} after : jpg")
                    changed_count+=1
                else: pass
            
            print(f"ㅡ 확장자 변환 종료 (총 변환 갯수: {changed_count}) ㅡ")

    #####################################################################################################################
        def checking(self):
            # 입력 받은 검색어가 이름인 폴더가 존재하면 중복으로 판단
            for dir_name in os.listdir(self.query):
                file_list = os.listdir(os.path.join(self.query, dir_name))
                if self.query in file_list:
                    print(f"ㅡ 중복된 검색어: ({dir_name}) ㅡ")
                    return True
                
        def playing_mp3(self):
            mixer.init()
            mixer.music.load(self.mp3)
            mixer.music.play()
            while mixer.music.get_busy():
                pass
            print(f"ㅡ 검색어: {self.query} ㅡ")

#####################################################################################################################
    crawler = Crawler()
    crawler.crawling()
    crawler.filtering(args.filter_size,args.filter_size)
    crawler.change_extension()
    crawler.playing_mp3()

if __name__ == '__main__':
    main()
