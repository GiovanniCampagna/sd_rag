"""
Script diagnostico — esegui questo PRIMA di usare il nuovo extract_pdf.py.
Stampa le prime 3 righe di ogni tabella trovata nelle prime 15 pagine del PDF,
così puoi verificare che pdfplumber legga correttamente le tabelle di ISP-18.
"""
import pdfplumber

PDF_PATH = r'C:\Users\campa\Desktop\materiale tesi\ISP-18 Rifornimento carburante Agg19 01.04.2025.pdf'

with pdfplumber.open(PDF_PATH) as pdf:
    found = 0
    for i, page in enumerate(pdf.pages[:20]):
        tables = page.extract_tables()
        if tables:
            found += len(tables)
            print(f'--- Pagina {i}: {len(tables)} tabella/e ---')
            for t, table in enumerate(tables):
                print(f'  Tabella {t} ({len(table)} righe x {len(table[0])} colonne):')
                for row in table[:3]:
                    cleaned = [c.replace('\n', ' ') if c else '' for c in row]
                    print('    ' + ' | '.join(cleaned))
            print()
    print(f'Totale tabelle trovate nelle prime 20 pagine: {found}')
