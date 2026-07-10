from pylatexenc.latex2text import LatexNodes2Text


tex_with_tables = ['./tex/unidades-curriculares.tex',
                   './tex/unidades-optativas-especificas.tex',
                   './tex/unidades-optativas-humanidades.tex']

conversor = LatexNodes2Text()

for filename in tex_with_tables:
    with open(filename, 'r+') as f:
        content = f.read()
        cleaned_tex = conversor.latex_to_text(content)
        f.seek(0)
        f.truncate()
        f.write(cleaned_tex)


