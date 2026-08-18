import time
from playwright.sync_api import sync_playwright

def get_logs():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        logs = []
        page.on("console", lambda msg: logs.append(f"CONSOLE {msg.type}: {msg.text}"))
        page.on("pageerror", lambda err: logs.append(f"PAGE ERROR: {err}"))
        
        page.goto("http://127.0.0.1:8501")
        print("Waiting for Streamlit app to load...")
        time.sleep(10)
        
        with open("browser_logs.txt", "w") as f:
            f.write("\n".join(logs))
        print("Logs saved to browser_logs.txt")
        browser.close()

if __name__ == "__main__":
    get_logs()
