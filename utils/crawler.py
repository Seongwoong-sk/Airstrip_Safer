## ๐ Web Crawler ## 

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
            
            #  ์ด๋ฏธ์ง๋ค์ด ์ ์ฅ๋  ๊ฒฝ๋ก ๋ฐ ํด๋ ์ด๋ฆ
            self.path = args.path
            
            # ๊ฒ์์ด ์๋ ฅ ๋ฐ ์ค๋ณต ๊ฒ์ฌ
            self.query = input('๊ฒ์ ํค์๋ ์๋ ฅ : ')
            
            # ์น ๋ธ๋ผ์ฐ์ ์ ๊ฐ๋ฐ์ ๋ชจ๋(F12)๋ฅผ ์ด์ด console์ navigator.userAgent๋ผ๊ณ  ์๋ ฅ ํ ์ถ๋ ฅ๋๋ ๊ฐ์ ๋ณต์ฌ
            user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
            opts = Options()
            opts.add_argument(f"user-agent={user_agent}")
            opts.add_argument('headless')  # ์ฐฝ ๋์ธ์ง ๋ง์ง 
            opts.add_argument('window-size=1920x1080')
            opts.add_argument('disable-gpu')
            opts.add_argument('ignore-certificate-errors')
            
            # ๋๋ผ์ด๋ฒ ์์ฑ
            self.driver = webdriver.Chrome(args.webdriver, options=opts)
            
            # clickAndRetrieve() ๊ณผ์ ์์ urlretrieve์ด ๋๋ฌด ์ค๋ ๊ฑธ๋ฆด ๊ฒฝ์ฐ๋ฅผ ๋๋นํด ํ์ ์์ ์ง์ 
            socket.setdefaulttimeout(30)
            
            # ํฌ๋กค๋งํ ์ด๋ฏธ์ง ์ 
            self.crawled_count = 0
            
            # mp3 ํ์ผ ๊ฒฝ๋ก
            self.mp3 = args.mp3

    #####################################################################################################################
        def scroll_down(self):
            scroll_count = 0
            
            print("-- ์คํฌ๋กค ๋ค์ด ์์ --")
            
            # ์คํฌ๋กค ์์น๊ฐ ์ป๊ณ  last_height์ ์ ์ฅ
            last_height = self.driver.execute_script("return document.body.scrollHeight")
        
            # '๊ฒฐ๊ณผ ๋๋ณด๊ธฐ' ๋ฒํผ์ ํด๋ฆญํ๋ ์ง ์ ๋ฌด
            after_click = False
            
            while True:
                print(f"-- ์คํฌ๋กค ํ์ : {scroll_count} --")
                
                # ์คํฌ๋กค ๋ค์ด
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                scroll_count += 1
                time.sleep(1.5)
                
                # ์คํฌ๋กค ์์น๊ฐ ์ป๊ณ  new_height ์ ์ ์ฅ
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                
                # ์คํฌ๋กค์ด ์ตํ๋จ์ด๋ฉฐ
                if last_height == new_height:
                    
                    # '๊ฒฐ๊ณผ ๋๋ณด๊ธฐ' ๋ฒํผ์ ํด๋ฆญํ ์ ์ด ์๋ ๊ฒฝ์ฐ
                    if after_click is True:
                        print(" -- ์คํฌ๋กค ๋ค์ด ์ข๋ฃ --")
                        break
                    
                    # '๊ฒฐ๊ณผ ๋๋ณด๊ธฐ' ๋ฒํผ์ ํด๋ฆญํ ์ ์ด ์๋ ๊ฒฝ์ฐ
                    elif after_click is False:
                        if self.driver.find_element_by_css_selector(".mye4qd").is_displayed():
                            self.driver.find_element_by_css_selector(".mye4qd").click()
                            print("-- '๊ฒฐ๊ณผ ๋๋ณด๊ธฐ' ๋ฒํผ ํด๋ฆญ --")
                            after_click = True
                        elif NoSuchElementException:
                            print(' -- NoSuchElementException --')
                            print(' -- ์คํฌ๋กค ๋ค์ด ์ข๋ฃ --')
                            break
                last_height = new_height
                
    #####################################################################################################################
        
        def click_and_retrieve(self,index, img, img_list_length):
            
            try:
                ActionChains(self.driver).click(img).perform()
                time.sleep(1.5)
                self.driver.implicitly_wait(3)
                imgurl = self.driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute("src")
                
                # ํ์ฅ์ 
                if re.search(r"jpeg|png",imgurl):
                    ext = re.search(r"jpeg|png", imgurl).group()
                else:
                    ext = 'jpg'
                
                # ์ ์ฅ๋  ์ด๋ฏธ์ง ํ์ผ ๊ฒฝ๋ก
                dst = os.path.join(self.path,self.query,f"{self.query}_{self.crawled_count+1}.{ext}")
                self.crawled_count += 1
                
                urlretrieve(imgurl, f"{dst}")
                self.driver.implicitly_wait(3)
                print(f"{index + 1} / {img_list_length} ๋ฒ์งธ ์ฌ์ง ์ ์ฅ {ext}")
                
            except HTTPError:
                print("ใก HTTPError & ํจ์ค ใก")
                pass

    #####################################################################################################################
            
        def crawling(self):
            print('-- ํฌ๋กค๋ง ์์ --')
            
            self.driver.get("https://www.google.co.kr/imghp?hl=ko&authuser=0&ogbl")
            time.sleep(2)
            self.driver.find_element_by_css_selector("input.gLFyf").send_keys(self.query) #send keyword
            self.driver.find_element_by_css_selector("input.gLFyf").send_keys(Keys.RETURN)##send Keys.RETURN
            
            self.scroll_down()
            
            # class_name์ ๊ณต๋ฐฑ์ด ์๋ ๊ฒฝ์ฐ ์ฌ๋ฌ ํด๋์ค๊ฐ ์๋ ๊ฒ์ด๋ฏ๋ก ์๋์ ๊ฐ์ด css_selector๋ก ์ฐพ์
            img_list = self.driver.find_elements_by_css_selector("img.rg_i.Q4LuWd") # ์ธ๋ค์ผ
            
            # ๋๋ ํ ๋ฆฌ ์์ฑ
            directory_path = os.path.join(self.path, self.query)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                print(f'{directory_path} ํด๋ ์์ฑ')
            else : print(f"{directory_path} Directory Already Exists")
            
            for index, img in enumerate(img_list):
                try:
                    self.click_and_retrieve(index, img, len(img_list))

                except ElementClickInterceptedException:
                    print("ใก ElementClickInterceptedException ใก")
                    self.driver.execute_script("window.scrollTo(0, window.scrollY + 100)")
                    print("ใก 100๋งํผ ์คํฌ๋กค ๋ค์ด ๋ฐ 3์ด ์ฌ๋ฆฝ ใก")
    #                 img.click()
                    ActionChains(self.driver).click(img).perform()
                    time.sleep(3)
                    self.click_and_retrieve(index, img, len(img_list))

                except NoSuchElementException:
                    print("ใก NoSuchElementException ใก")
                    self.driver.execute_script("window.scrollTo(0, window.scrollY + 100)")
                    print("ใก 100๋งํผ ์คํฌ๋กค ๋ค์ด ๋ฐ 4์ด ์ฌ๋ฆฝ ใก")
                    time.sleep(4)
    #                 img.click()
                    ActionChains(self.driver).click(img).perform()
                    self.click_and_retrieve(index, img, len(img_list))

                except ConnectionResetError:
                    print("ใก ConnectionResetError & ํจ์ค ใก")
                    pass

                except URLError:
                    print("ใก URLError & ํจ์ค ใก")
                    pass

                except socket.timeout:
                    print("ใก socket.timeout & ํจ์ค ใก")
                    pass

                except socket.gaierror:
                    print("ใก socket.gaierror & ํจ์ค ใก")
                    pass

                except ElementNotInteractableException:
                    print("ใก ElementNotInteractableException ใก")
                    break

            try:
                print("ใก ํฌ๋กค๋ง ์ข๋ฃ (์ฑ๊ณต๋ฅ : %.2f%%) ใก" % (self.crawled_count / len(img_list) * 100.0))

            except ZeroDivisionError:
                print("ใก img_list ๊ฐ ๋น์ด์์ ใก")

            self.driver.quit()

    #####################################################################################################################

        def filtering(self, width_threshold, height_threshold):
            print("ใก ํํฐ๋ง ์์ ใก")
            filtered_count = 0
            dir_name = os.path.join(self.path, self.query)
            
            for index, file_name in enumerate(os.listdir(dir_name)):
                try:
                    file_path = os.path.join(dir_name, file_name)
                    ext = file_name.split('.')[-1]
                    img = Image.open(file_path)

                    # ์ด๋ฏธ์ง ํด์๋์ ๊ฐ๋ก์ ์ธ๋ก๊ฐ ๋ชจ๋ 800์ดํ์ธ ๊ฒฝ์ฐ
                    if (img.width < width_threshold and img.height < height_threshold):
                        img.close()
                        os.remove(file_path)
                        print(f"{index} ๋ฒ์งธ ์ฌ์ง ์ญ์  -->  width : {img.width} height : {img.height}")
                        filtered_count += 1

                # ์ด๋ฏธ์ง ํ์ผ์ด ๊นจ์ ธ์๋ ๊ฒฝ์ฐ
                except OSError:
                    os.remove(file_path)
                    filtered_count += 1

            print(f"ใก ํํฐ๋ง ์ข๋ฃ (์ด ๊ฐฏ์: {self.crawled_count - filtered_count}) ใก")
                    
    #####################################################################################################################      
        def change_extension(self):
            print("ใก ํํฐ๋ง ์์ ใก")
            changed_count = 0
            dir_name = os.path.join(self.path, self.query)
            
            for index, file_name in enumerate(os.listdir(dir_name)):
                ext = file_name.split('.')[-1]
                img_name = file_name.split('.')[0]
                if ext != 'jpg':
                    os.rename(os.path.join(dir_name,file_name), os.path.join(dir_name,img_name+'.jpg'))
                    print(f"{index} ๋ฒ์งธ ์ฌ์ง ํ์ฅ์ ๋ณ๊ฒฝ -->  before : {ext} after : jpg")
                    changed_count+=1
                else: pass
            
            print(f"ใก ํ์ฅ์ ๋ณํ ์ข๋ฃ (์ด ๋ณํ ๊ฐฏ์: {changed_count}) ใก")

    #####################################################################################################################
        def checking(self):
            # ์๋ ฅ ๋ฐ์ ๊ฒ์์ด๊ฐ ์ด๋ฆ์ธ ํด๋๊ฐ ์กด์ฌํ๋ฉด ์ค๋ณต์ผ๋ก ํ๋จ
            for dir_name in os.listdir(self.query):
                file_list = os.listdir(os.path.join(self.query, dir_name))
                if self.query in file_list:
                    print(f"ใก ์ค๋ณต๋ ๊ฒ์์ด: ({dir_name}) ใก")
                    return True
                
        def playing_mp3(self):
            mixer.init()
            mixer.music.load(self.mp3)
            mixer.music.play()
            while mixer.music.get_busy():
                pass
            print(f"ใก ๊ฒ์์ด: {self.query} ใก")

#####################################################################################################################
    crawler = Crawler()
    crawler.crawling()
    crawler.filtering(args.filter_size,args.filter_size)
    crawler.change_extension()
    crawler.playing_mp3()

if __name__ == '__main__':
    main()
