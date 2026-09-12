"use client";

import { useChat } from "ai/react";
import { RotateCcw } from "lucide-react";
import { toast } from "sonner";
import { useState } from "react";

import { Chat } from "@/components/ui/chat";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";

// Persona mặc định — từ Câu 4.1 trong exercises.md
const DEFAULT_SYSTEM =
  "Bạn là trợ giảng AI thông thái, kiên nhẫn và thân thiện của khóa học. " +
  "Hãy giải thích các khái niệm bằng tiếng Việt chuẩn mực, trả lời súc tích trong tối đa 3–4 câu, " +
  "và luôn kèm một ví dụ thực tế dễ hiểu.";

const SUGGESTIONS = [
  "Giải thích blockchain cho trẻ 8 tuổi.",
  "temperature và top_p khác nhau thế nào?",
  "Vì sao streaming quan trọng với chatbot?",
];

export default function Home() {
  const [system, setSystem] = useState(DEFAULT_SYSTEM);

  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    append,
    status,
    stop,
    setMessages,
  } = useChat({
    body: { system },
    onError: (error) => toast.error(error.message),
  });

  const isGenerating = status === "submitted" || status === "streaming";

  return (
    <main className="mx-auto flex h-dvh max-w-3xl flex-col gap-4 p-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle>System instruction (persona)</CardTitle>
          <CardDescription>
            Định hình vai trò của model trước mỗi lượt chat — như Câu 2.1.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <Textarea
            value={system}
            onChange={(e) => setSystem(e.target.value)}
            rows={3}
          />
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSystem(DEFAULT_SYSTEM)}
            >
              Persona mặc định
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setMessages([])}
            >
              <RotateCcw /> Xóa hội thoại
            </Button>
          </div>
        </CardContent>
      </Card>

      <Chat
        className="min-h-0 flex-1"
        messages={messages}
        input={input}
        handleInputChange={handleInputChange}
        handleSubmit={handleSubmit}
        isGenerating={isGenerating}
        stop={stop}
        append={append}
        setMessages={setMessages}
        suggestions={SUGGESTIONS}
      />
    </main>
  );
}