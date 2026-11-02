"""Rebuild the original Chinese PDF demo fixture. Requires reportlab."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


def main():
    directory = Path(__file__).parent
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    output = canvas.Canvas(str(directory / 'database-notes.pdf'), pagesize=A4)
    output.setTitle('数据库复习手册 - Pagewise 原创演示材料')
    sections = (directory / 'database-notes.txt').read_text(encoding='utf-8').split('\n\n')[1:]
    for page, section in enumerate(sections, 1):
        output.setFont('STSong-Light', 19)
        lines = section.strip().splitlines()
        output.drawString(48, 780, lines[0])
        output.setFont('STSong-Light', 12)
        y = 735
        for paragraph in lines[1:]:
            for start in range(0, len(paragraph), 38):
                output.drawString(48, y, paragraph[start:start + 38])
                y -= 23
            y -= 10
        output.setFont('STSong-Light', 10)
        output.drawString(48, 40, f'Pagewise 原创示例 · {page} / 4')
        output.showPage()
    output.save()


if __name__ == '__main__':
    main()
