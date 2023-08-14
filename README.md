<p align="center">
    <h1>PhD Thesis</h1>
    <img alt="progress-pages" src="https://progress-bar.dev/108?title=108/100 Pages" />
    <img alt="progress-words" src="https://progress-bar.dev/78?title=31344/40000 Words" />
</p>

This repository contains the source code and the compiled versions of my Ph.D. Thesis.

- [Markdown Source File](/thesis.md)
- [Bibliography](/bibliography.bib)
- [Figures](/figures)
- [PDF File](/docs/thesis.pdf)
- [PDF Booklet File](/docs/thesis-book.pdf)
- [HTML File](/docs/index.html)
- [License](/LICENSE)

## Quickstart

- Prepare a **Dev Environement**:
    - Install the [VSCode deb file](https://code.visualstudio.com/): `sudo dpkg -i code_${VERSION}_amd64.deb`
    - Install the [Pandoc deb file](https://github.com/jgm/pandoc/releases): `sudo dpkg -i pandoc-${VERSION}-amd64.deb`
    - Install the [Pandoc-CrossRef tar file](https://github.com/lierdakil/pandoc-crossref/releases):
        - Extract the files: `tar -xf pandoc-crossref-Linux.tar.xz`
        - Move the files: `mv pandoc-crossref /usr/bin/ && mv pandoc-crossref.1 /usr/bin/`
        - Give execution rights: `sudo chmod a+x pandoc-crossref pandoc-crossref.1`
    - Install the [Pandoc-CiteProc executable](https://github.com/jgm/pandoc-citeproc/releases):
        - Update apt and install: `sudo apt update && sudo apt install pandoc-citeproc`
    - Install the [Pandoc-Acronym module](https://github.com/kprussing/pandoc-acro.git):
        - Git clone: `git clone https://github.com/kprussing/pandoc-acro.git`
        - Install python module: `python3 setup.py install`
    - Install xelatex libs: `sudo apt install texlive-full`
    - Install entr: `sudo apt install entr`
    - Install exiftool: `sudo apt install exiftool`
    - Install librsvg: `sudo apt install librsvg2-bin`
    - Install ghostscript: `sudo apt install ghostscript`

- Launch a **Dev Session**:
    - Launch in terminal: `make dev`
    - Serve the `HTML` file with VSCode or other
- Build the **Thesis Files**: `make`

## License

```
AI-Assisted Creative Expression: a Case for Automatic Lineart Colorization © 2023 by Yliess Hati is licensed under CC BY 4.0
```