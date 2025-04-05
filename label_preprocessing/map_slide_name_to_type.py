import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import time

# Load slide-uids.json
try:
    with open('slide_uids.json', 'r') as f:
        slide_uids = json.load(f)
except FileNotFoundError:
    print("slide_uids.json not found. Please ensure it's in the correct directory.")
    slide_uids = []

# Remove duplicates based on StudyInstanceUID and SeriesInstanceUID
unique_slide_uids = []
seen_uids = set()
for item in slide_uids:
    uid_pair = (item['studyInstanceUID'], item['seriesInstanceUID'])
    if uid_pair not in seen_uids:
        unique_slide_uids.append(item)
        seen_uids.add(uid_pair)

# Configure Chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--start-maximized")


# chrome_options.add_argument("--headless")  # Uncomment for headless mode

def extract_slide_data(url):
    driver = webdriver.Chrome(options=chrome_options)
    slide_data = []

    try:
        driver.get(url)

        # Wait for viewer to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".ant-menu-item"))
        )

        # Get all slide elements in the left panel
        slides = driver.find_elements(By.CSS_SELECTOR, ".ant-menu-item")
        print(f"Found {len(slides)} slides")

        for index in range(len(slides)):
            # Re-find elements to avoid staleness
            current_slides = driver.find_elements(By.CSS_SELECTOR, ".ant-menu-item")
            slide = current_slides[index]

            # Click the slide
            driver.execute_script("arguments[0].scrollIntoView();", slide)
            slide.click()
            time.sleep(1)  # Allow time for animation

            # Extract slide number
            slide_number = slide.find_element(By.CSS_SELECTOR, ".ant-card-head-title").text

            # Extract description from right panel
            try:
                description_rows = WebDriverWait(driver, 5).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".ant-descriptions-item-content"))
                )
                for row in description_rows:
                    if row.text.startswith("HE"):
                        description = row.text.strip()
                        break
                else:
                    description = "HE description not found"
            except TimeoutException:
                description = "Description not found"

            slide_data.append({
                "slide_name": f"{slide_number}.svs",
                "description": description
            })

            print(f"Processed slide {index + 1}/{len(slides)}: {slide_number}")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        driver.quit()

    return slide_data


# Process each URL and save data every 5 pairs
slide_name_to_type = []
for index, item in enumerate(unique_slide_uids):
    url = f"https://viewer.imaging.datacommons.cancer.gov/slim/studies/{item['studyInstanceUID']}/series/{item['seriesInstanceUID']}"
    print(f"Processing URL: {url}")
    slide_data = extract_slide_data(url)
    slide_name_to_type.extend(slide_data)

    if (index + 1) % 5 == 0:
        with open('slide_name_to_type.json', 'w') as f:
            json.dump(slide_name_to_type, f, indent=4)
        print(f"Data saved to slide_name_to_type.json after processing {index + 1} pairs")

# Final save if not already saved
if len(unique_slide_uids) % 5 != 0:
    with open('slide_name_to_type.json', 'w') as f:
        json.dump(slide_name_to_type, f, indent=4)
    print("Final data saved to slide_name_to_type.json")

print("All data processed and saved")
