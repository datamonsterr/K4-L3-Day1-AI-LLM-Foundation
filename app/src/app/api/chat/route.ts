import { createOpenAI } from "@ai-sdk/openai";
import { convertToCoreMessages, streamText } from "ai";

// Persona mặc định — từ Câu 4.1 trong exercises.md
const DEFAULT_SYSTEM =
  "Bạn là trợ giảng AI thông thái, kiên nhẫn và thân thiện của khóa học. " +
  "Hãy giải thích các khái niệm bằng tiếng Việt chuẩn mực, trả lời súc tích trong tối đa 3–4 câu, " +
  "và luôn kèm một ví dụ thực tế dễ hiểu.";

// Cửa sổ trượt: giữ 3 lượt hỏi–đáp gần nhất (6 message) — như streaming_chatbot()
const MAX_HISTORY_MESSAGES = 6;

export async function POST(req: Request) {
  const apiKey = process.env.OPENAI_API_KEY ?? process.env.API_KEY;
  if (!apiKey) {
    return Response.json(
      {
        error:
          "Thiếu OPENAI_API_KEY (hoặc API_KEY). Copy .env.example thành .env.local rồi dán key vào.",
      },
      { status: 500 },
    );
  }

  try {
    const { messages, system } = await req.json();

    const openai = createOpenAI({
      apiKey,
      baseURL: process.env.OPENAI_BASE_URL ?? process.env.BASE_URL,
    });

    const result = await streamText({
      model: openai(process.env.LAB_MODEL ?? "gpt-4o"),
      system: system || DEFAULT_SYSTEM,
      messages: convertToCoreMessages(messages.slice(-MAX_HISTORY_MESSAGES)),
      temperature: 0.7,
      maxRetries: 3,
    });

    return result.toDataStreamResponse();
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : "Lỗi không xác định" },
      { status: 500 },
    );
  }
}