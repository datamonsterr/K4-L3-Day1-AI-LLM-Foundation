# Trợ lý AI — K4 Lab (Chat Streaming UI)

Chatbot web streaming với system instruction (persona) — bản UI của
`solution/solution.py` (Task 3.1 + 4.1), dùng Next.js 16 + shadcn-chatbot-kit.

## Chạy

```bash
pnpm install
cp .env.example .env.local   # dán OPENAI_API_KEY vào
pnpm dev                     # http://localhost:3000
```

## Cấu hình (.env.local)

| Biến | Ý nghĩa |
| --- | --- |
| `OPENAI_API_KEY` (hoặc `API_KEY`) | Key API — bắt buộc |
| `OPENAI_BASE_URL` (hoặc `BASE_URL`) | Endpoint tương thích OpenAI (NVIDIA NIM, Ollama...) — tùy chọn |
| `LAB_MODEL` | Model, mặc định `gpt-4o` |

## Kiến trúc

- `src/app/api/chat/route.ts` — streaming route: persona từ client, history 3 lượt
  (`slice(-6)`, như `streaming_chatbot`), `maxRetries: 3` (như `retry_with_backoff`).
- `src/app/page.tsx` — UI chat + card System instruction; persona gửi kèm mỗi request.