#!/bin/bash

if [ ! -d ./pubmed-rct ]; then
    echo "[+] Downloading dataset."
    git clone https://github.com/Franck-Dernoncourt/pubmed-rct.git
else
    echo "[*] Dataset exists already; skipping."
fi
