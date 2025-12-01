# compare_and_report.py
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import os

def create_report(case_results, out_pdf='outputs/report_summary.pdf'):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    c = canvas.Canvas(out_pdf, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Sonic Boom Propagation - Summary Report")
    y = height - 80
    for case in case_results:
        title = case.get('title', 'Case')
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, title)
        y -= 18
        # metrics
        for k,v in case['metrics'].items():
            c.setFont("Helvetica", 10)
            c.drawString(60, y, f"{k}: {v:.4g}" if isinstance(v,(int,float)) else f"{k}: {v}")
            y -= 14
        # add image if exists
        imgpath = case.get('image')
        if imgpath and os.path.exists(imgpath):
            c.drawImage(imgpath, 50, y-200, width=500, height=200)
            y -= 220
        else:
            y -= 10
        if y < 120:
            c.showPage()
            y = height - 50
    c.save()
    print(f"Report saved to {out_pdf}")
