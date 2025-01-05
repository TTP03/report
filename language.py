import json
from typing import List
from nltk.tokenize import word_tokenize

class Language:
    """
    Language abstraction to handle tokenizer, word2idx, idx2word.
    """

    def __init__(
        self,
        language: str,
        path_to_word2idx: str,
        unk_id: int,
        bos_id: int,
        eos_id: int,
    ):
        self.language = language
        self.path_to_word2idx = path_to_word2idx
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        # Load word2idx and prepare idx2word
        with open(path_to_word2idx, mode="r", encoding="utf-8") as fp:
            self.word2idx = json.load(fp)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode_sentence(
        self,
        sentence: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        # Loại bỏ khoảng trắng đầu và cuối
        sentence = sentence.strip()
        if not sentence:  # Xử lý câu trống
            return [self.unk_id]

        try:
            # Sử dụng nltk để tokenize câu
            tokens = word_tokenize(sentence)
        except Exception as e:
            print(f"Error tokenizing sentence: {sentence}. Error: {e}")
            return [self.unk_id]

        # Ánh xạ các từ thành chỉ số (word2idx)
        seq = [self.word2idx.get(word, self.unk_id) for word in tokens]

        # Thêm token <bos> và <eos> nếu được yêu cầu
        if add_bos:
            seq.insert(0, self.bos_id)
        if add_eos:
            seq.append(self.eos_id)

        return seq

