# Generate PDFs from the Markdown source files
#
# In order to use this makefile, you need some tools:
# - GNU make
# - Pandoc
# - LuaLaTeX
# - DejaVu Sans fonts

name := lisa-mail

all: $(name).pdf

# Recipe for converting a Markdown file into PDF using Pandoc
$(name).pdf: $(name).md
	pandoc \
		--variable mainfont="DejaVu Sans" \
		--variable monofont="DejaVu Sans Mono" \
		--variable fontsize=11pt \
		--variable geometry:"top=1.5cm, bottom=2.5cm, left=1.5cm, right=1.5cm" \
		--variable geometry:a4paper \
		--table-of-contents \
		--number-sections \
		-f markdown  $< \
		--latex-engine=lualatex \
		-o $@

.PHONY : clean

clean:
	rm -f $(name).pdf
