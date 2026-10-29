from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

corpus = Path("data_merged_updated.txt")

if not corpus.exists():
    raise SystemExit("ERROR: data_merged_updated.txt was not found.")

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.NFC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    show_progress=True,
)

print("Training a BPE tokenizer on:", corpus)
tokenizer.train([str(corpus)], trainer)
tokenizer.save("tokenizer_bpe.json")

print("Tokenizer saved to tokenizer_bpe.json")
print("Actual vocabulary size:", tokenizer.get_vocab_size())

for sample in [
    "Artificial intelligence learns patterns from data.",
    "The purpose of a computer is to process information.",
]:
    print("\nText:", sample)
    print("Tokens:", tokenizer.encode(sample).tokens)
