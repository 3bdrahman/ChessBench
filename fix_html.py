import re

with open('chess_fight/ui/streamlit_app.py', 'r') as f:
    content = f.read()

# Remove specific start divs
content = re.sub(r"^[ \t]*st\.markdown\(['\"]<div class=[\"']cf-card.*?[\"']>['\"], unsafe_allow_html=True\)\n", "", content, flags=re.MULTILINE)
content = re.sub(r"^[ \t]*st\.markdown\(['\"]<div class=[\"']cf-export-buttons[\"']>['\"], unsafe_allow_html=True\)\n", "", content, flags=re.MULTILINE)

# Remove the closing divs
content = re.sub(r"^[ \t]*st\.markdown\(['\"]</div>['\"], unsafe_allow_html=True\)\n", "", content, flags=re.MULTILINE)

# Remove the specific auto-scroll script block
script_pattern = r"^[ \t]*st\.markdown\(\"\"\"\s*<script>\s*\(function\(\) \{.*?\}\)\(\);\s*</script>\s*\"\"\", unsafe_allow_html=True\)\n"
content = re.sub(script_pattern, "", content, flags=re.MULTILINE | re.DOTALL)

with open('chess_fight/ui/streamlit_app.py', 'w') as f:
    f.write(content)
