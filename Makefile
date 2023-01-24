FILENAME     = thesis

OPTIONS      = --standalone
OPTIONS     += -F /usr/bin/pandoc-crossref
OPTIONS     += --citeproc

PANDOC       = /usr/bin/pandoc
PANDOC_PDF   = ${PANDOC} ${FILENAME}.md -o ${FILENAME}.pdf ${OPTIONS}
PANDOC_HTML  = ${PANDOC} ${FILENAME}.md -o ${FILENAME}.html ${OPTIONS}

all: build

dev:
	echo ${FILENAME}.md | entr -r ${PANDOC_HTML}

build:
	${PANDOC_PDF}
	${PANDOC_HTML}