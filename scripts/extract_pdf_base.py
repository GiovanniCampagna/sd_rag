import fitz
import json

doc = fitz.open(r'C:\Users\campa\Desktop\materiale tesi\ISP-18 Rifornimento carburante Agg19 01.04.2025.pdf')
text = ''
for page in doc:
    text += page.get_text()

corpus = {'documento_prof': text}

with open('datasets/corpus_prof.json', 'w', encoding='utf-8') as f:
    json.dump(corpus, f, ensure_ascii=False, indent=2)

print(f'OK - {len(doc)} pagine estratte')
print(f'Caratteri totali: {len(text)}')
print(f'Anteprima primi 500 caratteri:')
print(text[:500])