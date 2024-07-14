import time, pyautogui as gui
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from utitlity import shapeIdentifiyer
import pandas as pd


def enterValue(element:WebElement, value:int):
    element.click()
    gui.hotkey("ctrl","a")
    element.send_keys(f"{value}")
    driver.find_element(by=By.CSS_SELECTOR, value='button[class="editorMotion_subTab__5irbg ui-label-regular editorMotion_active__S8SMV"]').click()

url="https://me.meshcapade.com/editor"

driver = Chrome()
driver.get(url)
driver.maximize_window()

gui.alert("Starting...")

height = driver.find_element(By.CSS_SELECTOR, value='''input[name="Height"]''')
chest = driver.find_element(By.CSS_SELECTOR, value='''input[name="Bust_girth"]''')
waist = driver.find_element(By.CSS_SELECTOR, value='''input[name="Waist_girth"]''')
hips = driver.find_element(By.CSS_SELECTOR, value='''input[name="Top_hip_girth"]''')
myData = []
i=1
try:
    for h in range(60,66):
        enterValue(height,h)
        for c in range(30,41):
            enterValue(chest,c)
            for w in range(30,41):
                enterValue(waist,w)
                for hi in range(30,41):
                    enterValue(hips, hi)
                    shape = shapeIdentifiyer(c,w,hi,"inches")
                    print(shape)
                    driver.find_element(by=By.CSS_SELECTOR, value='button[class="editorMotion_subTab__5irbg ui-label-regular editorMotion_active__S8SMV"]').click()
                    weight = float(driver.find_element(By.CSS_SELECTOR, value='''input[name="Weight"]''').get_attribute("value"))
                    x1,y1 = (819,294)
                    x2,y2 = (1160,949)
                    path =f"utils\data\{shape.lower()}\{i}.png"
                    gui.screenshot(path,[x1,y1,x2-x1,y2-y1]) # utils\data\{shape.lower()}\{i}.png
                    myData.append([h,weight,c,w,hi,shape.lower(),path])
                    i+=1
                # gui.alert("Next")
                # print(gui.position())
except Exception as e:
    print(e, e.with_traceback())
finally:
    df = pd.DataFrame(myData, columns="height weight bust waist hip shape img".split())
    df.to_csv("utils\data\data.csv",index=False)

gui.alert("Closing...")