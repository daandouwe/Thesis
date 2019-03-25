emacs main.tex &
emacs Thesis.bib &
evince ./out/main.pdf &
latexmk -pdf -pvc -f -interaction=nonstopmode -outdir="./out" main.tex

