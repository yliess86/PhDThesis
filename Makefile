FILENAME     = thesis

OPTIONS      = -s
OPTIONS     += -F pandoc-acro
OPTIONS     += -F /usr/bin/pandoc-crossref
OPTIONS     += --citeproc
OPTIONS     += --toc
OPTIONS     += --top-level-division part

PANDOC       = /usr/bin/pandoc
PANDOC_PDF   = ${PANDOC} ${OPTIONS} ${FILENAME}.md -o docs/${FILENAME}.pdf
PANDOC_HTML  = ${PANDOC} ${OPTIONS} --mathjax ${FILENAME}.md -o docs/index.html

BOOKLET      = pdfbook2
BOOKLET_PDF  = ${BOOKLET} docs/${FILENAME}.pdf

PAGE_TOTAL   = 170
PAGE_COUNT  := $(shell exiftool -T -PageCount -s3 -ext pdf docs/${FILENAME}.pdf)
PAGE_PROG   := $(shell python3 -c "print(int(${PAGE_COUNT} / ${PAGE_TOTAL} * 100))")

WORD_TOTAL   = 60000
WORD_COUNT  := $(shell pdftotext -layout docs/${FILENAME}.pdf - | tr -d '.' | wc -w)
WORD_PROG   := $(shell python3 -c "print(int(${WORD_COUNT} / ${WORD_TOTAL} * 100))")

all: build

dev:
	echo ${FILENAME}.md | entr -r make pdf clean

progress:
	sed -i -r 's|(https:\/\/progress-bar\.dev\/([0-9]{0,})\?title=([0-9]{0,})\/([0-9]{0,}) Pages)|https:\/\/progress-bar.dev\/${PAGE_PROG}\?title=${PAGE_COUNT}\/${PAGE_TOTAL} Pages|g' README.md
	sed -i -r 's|(https:\/\/progress-bar\.dev\/([0-9]{0,})\?title=([0-9]{0,})\/([0-9]{0,}) Words)|https:\/\/progress-bar.dev\/${WORD_PROG}\?title=${WORD_COUNT}\/${WORD_TOTAL} Words|g' README.md

html:
	cp -r figures docs/
	${PANDOC_HTML}

pdf:
	${PANDOC_PDF}

booklet:
	${BOOKLET_PDF}

clean:
	rm tmp-*

build: progress html pdf booklet clean
	echo "Build Done!\n"