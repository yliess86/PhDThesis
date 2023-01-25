FILENAME     = thesis

OPTIONS      = -s
OPTIONS     += -F pandoc-acro
OPTIONS     += -F /usr/bin/pandoc-crossref
OPTIONS     += --citeproc
OPTIONS     += --toc

PANDOC       = /usr/bin/pandoc
PANDOC_PDF   = ${PANDOC} ${OPTIONS} ${FILENAME}.md -o ${FILENAME}.pdf
PANDOC_HTML  = ${PANDOC} ${OPTIONS} ${FILENAME}.md -o docs/index.html

PAGE_TOTAL   = 251
PAGE_COUNT  := $(shell exiftool -T -PageCount -s3 -ext pdf thesis.pdf)
PAGE_PROG   := $(shell python3 -c "print(int(${PAGE_COUNT} / ${PAGE_TOTAL} * 100))")

all: build

dev:
	echo ${FILENAME}.md | entr -r ${PANDOC_HTML}

progress:
	sed -i -r 's|(https:\/\/progress-bar\.dev\/([0-9]{0,})\?title=([0-9]{0,})\/([0-9]{0,}) Pages)|https:\/\/progress-bar.dev\/${PAGE_PROG}\?title=${PAGE_COUNT}\/${PAGE_TOTAL} Pages|g' README.md

build: progress
	${PANDOC_PDF}
	${PANDOC_HTML}