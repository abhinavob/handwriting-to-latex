class Tokenizer:
    def __init__(self):
        self.special_tokens = ["<pad>", "<start>", "<end>"]
        self.token_to_id = {}
        self.id_to_token = {}

    def build_vocab(self, texts):
        tokens = set()
        for text in texts:
            tokens.update(text.split())
        tokens = sorted(list(tokens))
        all_tokens = self.special_tokens + tokens
        self.token_to_id = {tok: i for i, tok in enumerate(all_tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text):
        tokens = text.split()
        tokens = ["<start>"] + tokens + ["<end>"]
        return [self.token_to_id[tok] for tok in tokens]

    def decode(self, ids):
        tokens = [self.id_to_token[i] for i in ids]
        return " ".join(tokens)
