import torch
from language import Language
from network import Seq2SeqModel

# Đường dẫn tới các tệp
MODEL_PATH = "models/seq2seq.pth"
INPUT_LANG_VOCAB_PATH = "vocab/en_vocab.json"
OUTPUT_LANG_VOCAB_PATH = "vocab/vi_vocab.json"

# Hyper-parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_ID = 3
BOS_ID = 1
EOS_ID = 2

# Tải từ điển ngôn ngữ
input_language = Language("en", INPUT_LANG_VOCAB_PATH, unk_id=0, bos_id=BOS_ID, eos_id=EOS_ID)
output_language = Language("vi", OUTPUT_LANG_VOCAB_PATH, unk_id=0, bos_id=BOS_ID, eos_id=EOS_ID)

# Tải mô hình đã huấn luyện
model = Seq2SeqModel(
    encoder_num_embeddings=len(input_language.word2idx),
    encoder_embedding_dim=500,
    encoder_hidden_size=500,
    encoder_num_layers=2,
    encoder_dropout=0.2,
    decoder_num_embeddings=len(output_language.word2idx),
    decoder_embedding_dim=500,
    decoder_hidden_size=500,
    decoder_num_layers=2,
    decoder_dropout=0.2,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Hàm dịch
def translate(sentence):
    # 1. Tokenize và chuyển câu sang chỉ số
    input_seq = input_language.encode_sentence(sentence, add_bos=True, add_eos=True)
    input_tensor = torch.tensor([input_seq]).to(DEVICE)

    # 2. Encoder
    with torch.no_grad():
        decoder_hidden = model.encoder(input_tensor)

        # 3. Decode từng bước
        output_seq = [BOS_ID]
        for _ in range(50):  # Giới hạn độ dài câu đầu ra
            output_tensor = torch.tensor([output_seq]).to(DEVICE)
            logits = model.decoder(output_tensor, decoder_hidden)
            next_token = logits.argmax(dim=-1)[:, -1].item()  # Lấy từ có xác suất cao nhất
            if next_token == EOS_ID:  # Kết thúc khi gặp EOS
                break
            output_seq.append(next_token)

    # 4. Chuyển chỉ số thành từ
    output_sentence = " ".join(output_language.idx2word[idx] for idx in output_seq if idx not in {PAD_ID, BOS_ID, EOS_ID})
    return output_sentence

# Vòng lặp nhập input và dịch
if __name__ == "__main__":
    print("=== Máy dịch Anh-Việt ===")
    print("Nhập câu tiếng Anh để dịch. Gõ 'exit' để thoát.")
    while True:
        input_sentence = input("Bạn: ")
        if input_sentence.lower() == "exit":
            print("Kết thúc chương trình. Tạm biệt!")
            break
        output_sentence = translate(input_sentence)
        print(f"Dịch: {output_sentence}")
