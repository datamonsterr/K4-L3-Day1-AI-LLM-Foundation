# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi tăng temperature từ 0.0 lên 1.5, phản hồi chuyển từ tính tất định, chuẩn mực sang đa dạng và sáng tạo nhưng giảm dần tính nhất quán logic. Ở mức 0.0 và 0.5, mô hình đưa ra câu trả lời rất chính xác, quen thuộc (như xuất khẩu cà phê, hang Sơn Đoòng) và ổn định qua nhiều lần chạy. Ở mức 1.0 đến 1.5, câu chữ phong phú, bất ngờ và độc đáo hơn rõ rệt, nhưng với 1.5 mô hình có xu hướng lan man, cấu trúc câu kém chặt chẽ và dễ bị ảo giác (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature khoảng 0.2 đến 0.3 (thấp). Lý do là chatbot hỗ trợ khách hàng đòi hỏi tính chính xác cao, độ tin cậy tuyệt đối, tuân thủ chính sách/tài liệu sản phẩm và hạn chế tối đa ảo giác (hallucination). Mức nhiệt độ thấp này giúp câu trả lời nhất quán, đúng trọng tâm và có thể dự đoán được, đồng thời vẫn giữ được sự mượt mà tự nhiên trong giao tiếp vừa đủ mà không bị ngẫu hứng hay sai lệch thông tin quy trình.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với giá output $0.010/1K token của GPT-4o so với $0.0006/1K token của GPT-4o-mini, GPT-4o đắt hơn khoảng 16.67 lần (tương đương $105/ngày so với $6.30/ngày cho 10.5 triệu token output). GPT-4o xứng đáng chi phí trong các bài toán đòi hỏi suy luận phức tạp, phân tích hợp đồng pháp lý, chuẩn đoán dữ liệu chuyên sâu hoặc xử lý khiếu nại nhạy cảm cần độ chính xác tối đa. Ngược lại, nên dùng GPT-4o-mini cho các tác vụ thường nhật như FAQ/CSKH cấp 1, phân loại ý định người dùng (intent classification), tóm tắt tin nhắn ngắn hoặc trích xuất thông tin cơ bản để tối ưu chi phí và độ trễ.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi khác biệt rõ rệt về phong cách và cách tiếp cận: bản giáo viên tiểu học dùng từ ngữ mộc mạc, gần gũi với ví dụ trực quan ("cuốn sổ ghi chép chung của cả lớp mà không ai tẩy xóa được"), câu văn ngắn và giọng điệu thân thiện; trong khi bản chuyên gia tài chính sử dụng thuật ngữ chuyên sâu (distributed ledger, mật mã học phi tập trung, cơ chế đồng thuận, tính bất biến và triệt tiêu bên trung gian). System prompt đóng vai trò thiết lập khung nhận thức và ranh giới hành vi cho mô hình ngay từ đầu phiên làm việc. Nó định hình toàn bộ văn phong, độ phức tạp của từ vựng, góc nhìn chuyên môn cũng như xác định đối tượng người nghe mục tiêu để mô hình điều chỉnh nội dung cho phù hợp.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt 97 từ, `count_tokens` (tiktoken với model mặc định `gpt-4o`) ra 115 token, thấp hơn ước lượng `97 / 0.75 ≈ 129` token (chênh lệch khoảng 11%). Số token đo được thấp hơn ước lượng vì công thức `số từ / 0.75` (tương đương 1.33 token/từ) là hệ số ước tính thô chung, trong khi tokenizer `o200k_base` của GPT-4o đã mở rộng từ điển lên 200.000 token giúp nén các từ/âm tiết tiếng Việt phổ thông tốt hơn (~1.18 token/từ; nếu dùng tokenizer cũ `cl100k_base` của GPT-3.5/GPT-4 thì đoạn này tốn tới 214 token, cao hơn nhiều so với 129). Tuy nhiên, xét trên cùng độ dài ký tự hoặc cùng lượng thông tin diễn đạt, tiếng Việt vẫn tốn nhiều token hơn tiếng Anh (~1.5 đến 1.8 lần trên mỗi ký tự) do thuật toán BPE được huấn luyện áp đảo trên kho ngữ liệu tiếng Anh, trong khi các nguyên âm có dấu tiếng Việt (ă, â, ê, ô, ơ, ư, đ cùng các dấu thanh) biểu diễn bằng nhiều byte UTF-8 nên thường xuyên bị bẻ thành nhiều mảnh subword/byte token.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng có giao diện tương tác trực tiếp với người dùng như chatbot, trợ lý ảo hoặc công cụ soạn thảo, nơi việc giảm Thời gian tới Token Đầu tiên (Time-to-First-Token — TTFT) xuống vài trăm mili-giây giúp người dùng có thể đọc phản hồi ngay lập tức thay vì phải chờ đợi trong trạng thái màn hình bị đóng băng suốt nhiều giây. Ngược lại, non-streaming lại phù hợp hơn trong các pipeline xử lý ngầm (background jobs/batch processing), hệ thống chấm điểm tự động, các tác vụ trích xuất dữ liệu có cấu trúc (như JSON Object mà hệ thống cần nhận trọn vẹn để validate schema và parse dữ liệu an toàn), hoặc các kịch bản cần qua lớp kiểm duyệt an toàn (content moderation) trước khi gửi kết quả về cho client.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff có lợi thế vượt trội là kéo giãn khoảng cách giữa các lần thử lại theo cấp số nhân (ví dụ: 0.1s → 0.2s → 0.4s → 0.8s...), giúp giảm tải nhanh chóng tần suất yêu cầu dồn dập và tạo thời gian cho máy chủ tự giải phóng tài nguyên, phục hồi sau cơn nghẽn mạng. Nếu hàng nghìn client cùng sử dụng delay cố định (ví dụ 1 giây), toàn bộ các client sẽ đồng loạt gửi lại request cùng một lúc ở các mốc thời gian tuần hoàn, gây ra hiện tượng "thundering herd" (cộng hưởng sóng truy cập). Điều này khiến máy chủ liên tục bị sốc tải lặp đi lặp lại, dẫn đến tình trạng tê liệt kéo dài và sập hệ thống hoàn toàn (cascading failure).

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona là trợ giảng công nghệ AI thân thiện và kiên nhẫn. System prompt: "Bạn là trợ giảng AI thông thái, kiên nhẫn và thân thiện của khóa học. Hãy giải thích các khái niệm bằng tiếng Việt chuẩn mực, trả lời súc tích trong tối đa 3–4 câu, và luôn kèm một ví dụ thực tế dễ hiểu." Hai lựa chọn từ ngữ quan trọng gồm: (1) "trả lời súc tích trong tối đa 3–4 câu" nhằm hạn chế tràn màn hình terminal dòng lệnh (CLI), giúp người dùng nắm bắt thông điệp cốt lõi nhanh chóng và tiết kiệm tối đa số token output (giảm độ trễ lẫn chi phí API); (2) "bằng tiếng Việt chuẩn mực" để đảm bảo tính nhất quán ngôn ngữ, ngăn chặn việc mô hình tự ý pha trộn hoặc chuyển ngữ đột ngột sang tiếng Anh khi bắt gặp các thuật ngữ kỹ thuật chuyên ngành.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất hiện tại là cơ chế cắt history dạng cửa sổ trượt cố định (sliding window chỉ giữ 3 lượt gần nhất), khiến trợ lý bị mất hoàn toàn thông tin bối cảnh của những lượt trao đổi trước đó và không có bộ nhớ dài hạn xuyên suốt các phiên làm việc. Cải thiện đề xuất: Xây dựng cơ chế "Tóm tắt bộ nhớ đệm" (Conversation Summary Buffer Memory). Cách triển khai: Khi lịch sử vượt quá 3 lượt, thay vì xóa bỏ hoàn toàn tin nhắn cũ, ta kích hoạt một tác vụ chạy ngầm dùng model nhỏ (như GPT-4o-mini) để tóm tắt các lượt hội thoại đã qua thành một đoạn ngữ cảnh ngắn gọn (Summary). Bản tóm tắt này được lưu trong một role system/context phụ ghim ở đầu chuỗi hội thoại, trong khi danh sách history trực tiếp chỉ cần giữ 1–2 lượt mới nhất. Giải pháp này giúp trợ lý nhớ được toàn bộ diễn biến trao đổi dài mà vẫn kiểm soát tối ưu số lượng token đầu vào.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
