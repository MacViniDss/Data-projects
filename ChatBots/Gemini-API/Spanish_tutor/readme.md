# 🇪🇸 DicaBot — Aprenda Espanhol no Telegram

> Bot de dicas diárias de espanhol para brasileiros, powered by Google Gemini AI.

---

## Status Atual: Beta

Esta é a fase **Beta** do DicaBot — a fundação do projeto está construída e funcionando.

O bot é capaz de:
- 🤖 Gerar dicas únicas de vocabulário e gramática com Inteligência Artificial (Gemini 2.5 Flash)
- 📬 Enviar as dicas diretamente no Telegram para os usuários cadastrados
- 🗃️ Gerenciar usuários via banco de dados SQLite

O script é executado **manualmente** nesta fase, servindo como prova de conceito para validar toda a pipeline: IA → banco de dados → Telegram.

---


**Pré-requisito:** O usuário precisa ter enviado `/start` para o bot no Telegram antes de receber mensagens.

---

## 🗺️ Roadmap

### ✅ Beta (atual)
- [x] Integração com Google Gemini AI
- [x] Envio de mensagens via Telegram Bot API
- [x] Banco de dados SQLite com lista de usuários
- [x] Execução manual do script

---

### 🔜 Versão 1.0 — Automação com GitHub Actions

A **grande evolução** da V1 será a automação completa via **GitHub Actions**.

Em vez de rodar o script manualmente, o próprio GitHub irá disparar o bot **todos os dias, automaticamente e de graça** — sem precisar de servidor, sem custo de infraestrutura.

**Como vai funcionar:**

```
⏰ Todo dia às 8h (horário configurável)
        ↓
  GitHub Actions acorda
        ↓
  Executa o main.py
        ↓
  Gemini gera a dica do dia
        ↓
  Todos os usuários recebem no Telegram ✅
```

**O que será adicionado na V1:**
- [ ] Workflow `.github/workflows/daily_tip.yml` com `schedule: cron`
- [ ] `TELEGRAM_TOKEN` e `GEMINI_API_KEY` armazenados como **GitHub Secrets** (seguro)
- [ ] Logs de execução visíveis direto no painel do GitHub
- [ ] Zero custo — GitHub Actions oferece 2.000 minutos/mês gratuitos para repositórios públicos

> 💡 O repositório precisará ser **público** para usar os minutos gratuitos do GitHub Actions, ou ter uma conta com o plano adequado para repositórios privados.

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Uso |
|---|---|
| Python 3 | Linguagem principal |
| Google Gemini 2.5 Flash | Geração das dicas com IA |
| Telegram Bot API | Envio das mensagens |
| SQLite | Armazenamento dos usuários |
| python-dotenv | Gerenciamento de variáveis de ambiente |
| GitHub Actions *(V1)* | Automação e agendamento diário |

---

## 📄 Licença

MIT License — sinta-se livre para usar, modificar e distribuir.
