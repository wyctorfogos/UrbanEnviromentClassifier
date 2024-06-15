# Uso de app para analisar o app

## Descrição

Este aplicativo classifica diferentes tipos de ambientes usando um modelo de Machine Learning baseado no CLIP. As classes suportadas são:

- "person"
- "vehicle"
- "person with knife"
- "on fire"
- "person with gun"
- "graffiti"
- "person graffiting"
- "unknown"

# Para rodar a aplicação:
Crie um arquivo .env dentro da pasta '/docker' com o número da porta da api.

Rode o comando:

docker compose build && docker compose up -d
