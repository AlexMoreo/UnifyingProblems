from pathlib import Path
import os
from typing import List


def tabular2pdf(tabular_path, pdf_path, landscape=False, resizebox=False):
    parent = Path(pdf_path).parent
    if parent:
        os.makedirs(parent, exist_ok=True)
    
    doc_path = pdf_path.replace('.pdf', '.tex')

    tabular_path_rel = os.path.relpath(tabular_path, parent)
    table_str = tex_table(tabular_path_rel, resizebox=resizebox)
    tex_document(doc_path, [table_str], landscape=landscape)
    latex2pdf(pdf_path)



def latex2pdf(pdf_path: str, delete_tex=True, verbose=False):
    assert pdf_path.endswith('.pdf'), f'{pdf_path=} does not seem a valid name for a pdf file'
    tex_path = pdf_path.replace('.pdf', '.tex')

    dir = Path(pdf_path).parent
    pwd = os.getcwd()

    os.chdir(dir)
    cmd = 'pdflatex ' + Path(tex_path).name + ('' if verbose else ' >/dev/null 2>&1')
    if verbose:
        print("[running] $"+cmd)
    os.system(cmd)
    basename = Path(tex_path).name.replace('.tex', '')
    os.system(f'rm {basename}.aux {basename}.log')
    if delete_tex:
        os.system(f'rm {basename}.tex')
    os.chdir(pwd)
    if verbose:
        print('[Done]')


def tex_document(tex_path,
                 tables_str: List[str],
                 landscape=True,
                 dedicated_pages=True,
                 add_package=[]):

    lines = []
    lines.append('\\documentclass[10pt,a4paper]{article}')
    lines.append('\\usepackage[utf8]{inputenc}')
    lines.append('\\usepackage{amsmath}')
    lines.append('\\usepackage{amsfonts}')
    lines.append('\\usepackage{amssymb}')
    lines.append('\\usepackage{graphicx}')
    lines.append('\\usepackage{xcolor}')
    lines.append('\\usepackage{colortbl}')
    lines.append('\\usepackage{booktabs}')
    lines.append('\\usepackage{rotating}')
    lines.append('\\usepackage{multirow}')
    if landscape:
        lines.append('\\usepackage[landscape]{geometry}')
    for package in add_package:
        lines.append('\\usepackage{'+package+'}')
    lines.append('')
    lines.append('\\begin{document}')

    for table_str in tables_str:
        lines.append('')
        lines.append(table_str)
        lines.append('\n')
        if dedicated_pages:
            lines.append('\\newpage\n')
    lines.append('\\end{document}')

    document = '\n'.join(lines)

    parent = Path(tex_path).parent
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(tex_path, 'wt') as foo:
        foo.write(document)

    return document


def tex_table(tabular_path, caption=None, label=None, resizebox=None):

    lines = []
    lines.append('\\begin{table}[h]')
    lines.append('\center')

    if resizebox:
        lines.append('\\resizebox{\\textwidth}{!}{%')

    lines.append(f'\input{{{tabular_path}}}')

    if resizebox:
        lines.append('}%')

    if caption is None:
        caption = tabular_path.replace('_', '\_')

    lines.append(f'\caption{{{caption}}}')

    if label is not None:
        lines.append(f'\label{{{label}}}')

    lines.append('\end{table}')

    table_tex = '\n'.join(lines)

    return table_tex