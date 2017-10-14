mkdir datasets
cd datasets
curl https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip -o wikitext-2-v1.zip
curl https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -o wikitext-103-v1.zip
unzip wikitext-2-v1.zip
unzip wikitext-103-v1.zip
rm wikitext-2-v1.zip
rm wikitext-103-v1.zip
