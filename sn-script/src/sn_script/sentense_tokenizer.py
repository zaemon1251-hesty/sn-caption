import nltk
from sn_script.config import Config


def main(half_number: int):
    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        raw_txt_path = Config.base_dir / target / f"{half_number}_224p.txt"
        tokenized_txt_path = Config.base_dir / target / f"{half_number}_tokenized_224p.txt"
        with open(raw_txt_path, "r") as f:
            text = f.read()
        with open(tokenized_txt_path, "w") as f:
            f.write("\n".join(nltk.sent_tokenize(text)))


if __name__ == '__main__':
    nltk.download('punkt')
    half_number = 1
    main(half_number)