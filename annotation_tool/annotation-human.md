# アノテーションガイドライン

## 概要

協力者の皆様には、このガイドラインを読んでいただき、テキストの2値分類をしていただきます。
`annotation-description.txt`にLLM用のプロンプトと同様の指示と例が載っていますので、そちらを参考にしてください。
`annotation-target.txt`には、アノテーション対象のコメントが載っています。
`game`, `previous comments`, の3つの情報を元に、`comment`の `category`（0 or 1）を記入してください
よければ，判断基準や，判断しやすかったかどうかといったコメントを`note`書いてください

## annotation-target のフォーマット

```txt
- game => datetime TeamA match-result TeamB
- previous comments => comment[i-2] and comment[i-1]
- comment => comment[i]
category => (0 or 1)
note => ...
```

## 備考

- 英語は適宜日本語訳してください
- アノテーションが終わったら、`annotation-target.txt`を送り返してください
