import os

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

app_tex = "\\addcontentsline{toc}{section}{APPENDICES}\n"
app_tex += "\\section*{APPENDICES}\n\n"

app_tex += "\\subsection*{Appendix A: Python Source Code for Data Preprocessing}\n"
app_tex += "The following script handles the ingestion, cleaning, and mathematical synthesis of the Multi-Label ground truth targets.\n"
app_tex += "\\begin{verbatim}\n"
app_tex += read_file("../datamanupulation/01_preprocessing.py")
app_tex += "\\end{verbatim}\n\\clearpage\n\n"

app_tex += "\\subsection*{Appendix B: Python Source Code for Classifier Chain Modeling}\n"
app_tex += "The following script builds the Classifier Chain meta-architecture and evaluates XGBoost, RF, DT, and MLP.\n"
app_tex += "\\begin{verbatim}\n"
app_tex += read_file("../datamanupulation/02_modeling.py")
app_tex += "\\end{verbatim}\n\\clearpage\n\n"

app_tex += "\\subsection*{Appendix C: Python Source Code for XAI Counterfactual Generation}\n"
app_tex += "The following script implements the $L_2$-constrained gradient-free optimizer to extract actionable pedagogical pathways.\n"
app_tex += "\\begin{verbatim}\n"
app_tex += read_file("../datamanupulation/03_evaluation_xai.py")
app_tex += "\\end{verbatim}\n\\clearpage\n"

with open("text/Appendices.tex", "w") as f:
    f.write(app_tex)
