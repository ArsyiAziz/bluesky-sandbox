# Minimal makefile for Sphinx documentation.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean strict livehtml Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build with warnings as errors - what CI should run.
strict:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" -W --keep-going $(SPHINXOPTS) $(O)

# Live preview on http://127.0.0.1:8000, rebuilding on every save.
# Needs sphinx-autobuild, deliberately kept out of docs/requirements.txt so
# Read the Docs does not install a dev-only tool:  pip install sphinx-autobuild
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--open-browser --ignore "$(BUILDDIR)/*" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
