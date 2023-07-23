FILENAME     = thesis

OPTIONS      = -s
OPTIONS     += -F pandoc-acro
OPTIONS     += -F pandoc-crossref
OPTIONS     += --citeproc
OPTIONS     += --toc
OPTIONS     += --top-level-division chapter

PANDOC       = pandoc
PANDOC_PDF   = ${PANDOC} ${OPTIONS} ${FILENAME}.md -o docs/${FILENAME}.pdf
PANDOC_HTML  = ${PANDOC} ${OPTIONS} --mathjax ${FILENAME}.md -o docs/index.html

BOOKLET      = pdfbook2
BOOKLET_PDF  = ${BOOKLET} docs/${FILENAME}.pdf

PAGE_TOTAL   = 120
PAGE_COUNT  := $(shell exiftool -T -PageCount -s3 -ext pdf docs/${FILENAME}.pdf | tr -d ' ')
PAGE_PROG   := $(shell python -c "print(int(${PAGE_COUNT} / ${PAGE_TOTAL} * 100))")

WORD_TOTAL   = 60000
WORD_COUNT  := $(shell pdftotext -layout docs/${FILENAME}.pdf - | wc -w | tr -d ' ')
WORD_PROG   := $(shell python -c "print(int(${WORD_COUNT} / ${WORD_TOTAL} * 100))")

SED          = gsed
RGEX_PAGES   = 's+[0-9]*\?title=[0-9]*\/[0-9]* Pages+${PAGE_PROG}?title=${PAGE_COUNT}/${PAGE_TOTAL} Pages+g'
RGEX_WORDS   = 's+[0-9]*\?title=[0-9]*\/[0-9]* Words+${WORD_PROG}?title=${WORD_COUNT}/${WORD_TOTAL} Words+g'

all: build

dev:
	echo ${FILENAME}.md | entr -r make pdf clean

progress:
	${SED} -i -e ${RGEX_PAGES} -e ${RGEX_WORDS} README.md

html:
	cp -r figures docs/
	${PANDOC_HTML}

pdf:
	${PANDOC_PDF}

booklet:
	${BOOKLET_PDF}

clean:
	rm -f tmp-*

build: progress html pdf booklet clean
	echo "Build Done!\n"