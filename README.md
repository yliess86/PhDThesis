<p align="center">
    <h1>PhD Thesis</h1>
    <img alt="progress" src="https://progress-bar.dev/6?title=16/251 Pages" />
</p>

This repository contains the source code and the compiled versions of my Ph.D. Thesis.

- [Markdown Source File](/thesis.md)
- [Bibliography](/bibliography.bib)
- [Figures](/figures)
- [PDF File](/thesis.pdf)
- [HTML File](/thesis.html)
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
    - Install xelatex libs: `sudo apt install texlive-full`
    - Install entr: `sudo apt install entr`
    - Install exiftool: `sudo apt install exiftool`

- Launch a **Dev Session**:
    - Launch in terminal: `make dev`
    - Serve the `HTML` file with VSCode or other
- Build the **Thesis Files**: `make`

## License

```
AI-Assisted Creative Expression: a Case for Automatic Lineart Colorization © 2023 by Yliess Hati is licensed under CC BY 4.0
```