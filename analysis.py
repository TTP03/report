import matplotlib.pyplot as plt
from collections import Counter

# Đọc dữ liệu từ file
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

# Đường dẫn file
files = {
    "train_en": "data/train_en.txt",
    "train_vi": "data/train_vi.txt",
    "val_en": "data/val_en.txt",
    "val_vi": "data/val_vi.txt",
    "test_en": "data/test_en.txt",
    "test_vi": "data/test_vi.txt",
}

# Đọc dữ liệu
data = {name: read_file(path) for name, path in files.items()}

# Kiểm tra số lượng câu
for name, sentences in data.items():
    print(f"{name}: {len(sentences)} sentences")

# Độ dài trung bình câu
def calculate_avg_length(sentences):
    lengths = [len(sentence.split()) for sentence in sentences]
    return sum(lengths) / len(lengths), lengths

# Tính độ dài trung bình
avg_lengths = {name: calculate_avg_length(sentences) for name, sentences in data.items()}

for name, (avg, _) in avg_lengths.items():
    print(f"{name}: Average length = {avg:.2f} words")

# Vẽ biểu đồ phân phối độ dài
for name, (_, lengths) in avg_lengths.items():
    plt.hist(lengths, bins=50, alpha=0.6, label=name)

plt.title("Distribution of Sentence Lengths")
plt.xlabel("Sentence Length")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Tìm các từ phổ biến
def get_most_common_words(sentences, top_n=5):
    all_words = " ".join(sentences).split()
    counter = Counter(all_words)
    return counter.most_common(top_n)

# Liệt kê từ phổ biến
for name, sentences in data.items():
    print(f"Most common words in {name}:")
    print(get_most_common_words(sentences))
    print()
