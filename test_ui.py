import time
from playwright.sync_api import sync_playwright

def test():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://127.0.0.1:8501")
        print("Waiting for Streamlit app to load...")
        time.sleep(10)
        page.screenshot(path="home.png", full_page=True)
        print("Screenshot saved to home.png")
        browser.close()

if __name__ == "__main__":
    test()
