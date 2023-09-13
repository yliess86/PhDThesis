FILENAME     = thesis

OPTIONS      = -s
OPTIONS     += -F pandoc-acro
OPTIONS     += -F pandoc-crossref
OPTIONS     += --citeproc
OPTIONS     += --top-level-division chapter

PANDOC       = pandoc
PANDOC_PDF   = ${PANDOC} ${OPTIONS} ${FILENAME}.md -o docs/${FILENAME}.pdf
PANDOC_HTML  = ${PANDOC} ${OPTIONS} --mathjax ${FILENAME}.md -o docs/index.html

GS_COMPRESS  = gs -dNOPAUSE -dQUIET -dBATCH -dPDFSETTINGS=/ebook -sdocs/${FILENAME}.pdf docs/${FILENAME}.pdf
PDFTK_COVER  = pdftk docs/cover.pdf docs/${FILENAME}.pdf output docs/formatted_${FILENAME}.pdf && mv docs/formatted_${FILENAME}.pdf docs/${FILENAME}.pdf
BOOKLET_PDF  = pdfbook2 docs/${FILENAME}.pdf

PAGE_TOTAL   = 100
PAGE_COUNT  := $(shell exiftool -T -PageCount -s3 -ext pdf docs/${FILENAME}.pdf | tr -d ' ')
PAGE_PROG   := $(shell python -c "print(int(${PAGE_COUNT} / ${PAGE_TOTAL} * 100))")

WORD_TOTAL   = 40000
WORD_COUNT  := $(shell pdftotext -layout docs/${FILENAME}.pdf - | wc -w | tr -d ' ')
WORD_PROG   := $(shell python -c "print(int(${WORD_COUNT} / ${WORD_TOTAL} * 100))")

RGX_SRC      = (https\:\/\/progress\-bar\.dev\/)(([0-9]{0,})\?title=([0-9]{0,})\/([0-9]{0,}))
RGX_PAGES    = ${RGX_SRC} Pages
RGX_WORDS    = ${RGX_SRC} Words
RPL_PAGES    = https://progress-bar.dev/${PAGE_PROG}?title=${PAGE_COUNT}/${PAGE_TOTAL} Pages
RPL_WORDS    = https://progress-bar.dev/${WORD_PROG}?title=${WORD_COUNT}/${WORD_TOTAL} Words
PY_README_R  = r = open('README.md', 'r').read()
PY_README_F  = f = open('README.md', 'w')
RE_PAGES     = python -c "import re; ${PY_README_R}; ${PY_README_F}.write(re.sub(r'${RGX_PAGES}', '${RPL_PAGES}', r))"
RE_WORDS     = python -c "import re; ${PY_README_R}; ${PY_README_F}.write(re.sub(r'${RGX_WORDS}', '${RPL_WORDS}', r))"

all: build

dev:
	echo ${FILENAME}.md | entr -r make html

progress:
	${RE_PAGES}
	${RE_WORDS}

html:
	cp -r figures docs/
	${PANDOC_HTML}

pdf:
	${PANDOC_PDF}
	${GS_COMPRESS}
	${PDFTK_COVER}

booklet:
	${BOOKLET_PDF}
	rm -f tmp-*

build: progress html pdf booklet
	echo "Build Done!\n"