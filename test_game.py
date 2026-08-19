from playwright.sync_api import sync_playwright
import time

def test_app():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8501")
        print("Waiting for load...")
        time.sleep(10)
        
        page.screenshot(path="screenshot.png", full_page=True)
        print("Saved screenshot.png")
            
        browser.close()

if __name__ == "__main__":
    test_app()
