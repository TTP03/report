import re
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

def clean_dataset(file_path, output_path, language="en"):
    with open(file_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Loại bỏ ký tự không mong muốn
            line = re.sub(r'[^\w\s.,!?]', '', line)  # Giữ dấu câu cơ bản

            if line:  # Bỏ qua dòng trống
                # Token hóa
                if language == "english":
                    words = word_tokenize(line, language="english")
                else:  # Tiếng Việt (token hóa cơ bản)
                    words = line.split()

                # Ghi lại câu
                outfile.write(' '.join(words) + '\n')

# File paths
en_input_path = 'data/en.txt'
vi_input_path = 'data/vi.txt'
en_output_path = 'data/clean_en.txt'
vi_output_path = 'data/clean_vi.txt'

# Clean datasets
clean_dataset(en_input_path, en_output_path, language="english")
clean_dataset(vi_input_path, vi_output_path, language="vietnamese")

print(f"Cleaned datasets saved to:\n- {en_output_path}\n- {vi_output_path}")
