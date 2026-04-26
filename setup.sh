#!/bin/bash

# Cria o ambiente virtual
python -m venv .venv

# Ativa o ambiente virtual
# Windows (PowerShell):
#   .\.venv\Scripts\Activate.ps1
# Windows (CMD):
#   .\.venv\Scripts\activate.bat
# Linux/Mac:
#   source .venv/bin/activate

# Atualiza pip
pip install --upgrade pip

# Instala as dependencias
pip install -r requirements.txt
