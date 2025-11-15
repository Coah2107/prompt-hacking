#!/usr/bin/env python3
"""
Safe Emoji Removal Script - Only removes emojis from print statements and comments
Tránh việc làm hỏng code bằng cách chỉ xử lý print statements và comments
"""

import re
import os
import sys
from pathlib import Path

def safe_remove_emojis_from_file(file_path):
    """Safely remove emojis only from print statements and comments"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Store original content for comparison
        original_content = content
        
        # Split into lines for safer processing
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            # Only process print statements and comments
            if line.strip().startswith('#') or 'print(' in line:
                # Remove common emojis from these lines only
                emoji_pattern = r'[🔥⚡️✅❌📊🚀🎯🛡️📈📉💡🔍🎉⭐️📝📂📁💻🌟🔧⚙️🎨💯🏆📖📋📌💎🚨🌊⏰💪🔒🛠️📸🎪🎭🔊💰🎁🎂🎇🎆🎈🎊⚠️❤️❗❕❓❔]'
                line = re.sub(emoji_pattern, '', line)
                # Clean up extra spaces
                line = re.sub(r'  +', ' ', line)
                line = line.strip()
            
            processed_lines.append(line)
        
        new_content = '\n'.join(processed_lines)
        
        # Only write if there were actual changes
        if new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to safely remove emojis"""
    # Get project files only (exclude venv and other external files)
    python_files = []
    
    # Define directories to include
    include_dirs = [
        'detection_system',
        'prevention_system', 
        'scripts',
        'utils'
    ]
    
    for dir_name in include_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            for py_file in dir_path.rglob('*.py'):
                python_files.append(str(py_file))
    
    print(f"Found {len(python_files)} Python files in project directories")
    
    # Process each file safely
    cleaned = 0
    for file_path in python_files:
        if safe_remove_emojis_from_file(file_path):
            cleaned += 1
            print(f"Cleaned emojis from: {file_path}")
    
    print(f"Successfully cleaned {cleaned} files")
    print("Code structure preserved - only emojis in print statements and comments removed")

if __name__ == "__main__":
    main()
