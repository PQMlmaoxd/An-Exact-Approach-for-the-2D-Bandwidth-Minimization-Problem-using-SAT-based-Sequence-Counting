

**Hãy đóng vai một kiến trúc sư phần mềm Python chuyên gia và một nhà phát triển cấp cao.**

Nhiệm vụ của bạn là tạo ra một hướng dẫn toàn diện, cấp độ sản xuất về việc triển khai các cơ chế timeout cho các hàm chạy dài trong Python. Phản hồi của bạn phải được cấu trúc như một bài hướng dẫn chi tiết, hoàn chỉnh với các ví dụ mã, giải thích sâu sắc và một phân tích so sánh.

**BỐI CẢNH:**
Tôi đang xây dựng một hệ thống cần thực thi các tác vụ có thời gian thực thi không thể đoán trước. Một số tác vụ nặng về CPU (ví dụ: tính toán khoa học, xử lý dữ liệu), trong khi những tác vụ khác nặng về I/O (ví dụ: gọi API, truy vấn cơ sở dữ liệu). Tôi cần triển khai một cơ chế timeout mạnh mẽ để ngăn các tác vụ này bị treo vô thời hạn, qua đó bảo vệ tài nguyên hệ thống và đảm bảo khả năng phản hồi. Giải pháp phải đáng tin cậy, đa nền tảng (Windows, Linux, macOS) và tuân theo các thực tiễn tốt nhất của Python hiện đại.

**CÁC RÀNG BUỘC:**
- Sử dụng Python 3.9+.
- Tất cả mã phải bao gồm gợi ý kiểu (type hints) và docstrings toàn diện (kiểu Google).
- Tất cả các ví dụ phải độc lập và có thể chạy được.
- Ưu tiên các giải pháp mạnh mẽ và đa nền tảng.
- Giải thích rõ ràng các ưu và nhược điểm của mỗi phương pháp.

Vui lòng cấu trúc phản hồi của bạn theo bốn yêu cầu cốt lõi sau đây:

---

### **Yêu cầu Cốt lõi 1: Chiến lược Dứt khoát cho các Tác vụ Nặng về CPU (`multiprocessing`)**

Đây là yêu cầu quan trọng nhất. Cung cấp một lớp Python hoàn chỉnh, sẵn sàng cho sản xuất có tên là `ProcessTimeoutExecutor`. Lớp này nên thực thi một hàm đã cho trong một tiến trình con riêng biệt và thực thi một timeout nghiêm ngặt.

**Việc triển khai của bạn phải bao gồm:**
1.  Một phương thức `__init__` và hỗ trợ trình quản lý ngữ cảnh (`__enter__`, `__exit__`).
2.  Một phương thức chính, `execute(func, *args, **kwargs, timeout: float)`, chạy hàm mục tiêu.
3.  **Giao tiếp Mạnh mẽ:** Sử dụng một `multiprocessing.Queue` để trả về một cách an toàn kết quả của hàm *hoặc* bất kỳ ngoại lệ nào nó ném ra trở lại tiến trình cha.
4.  **Logic Timeout Chính xác:**
    - Sử dụng `process.join(timeout)` để chờ tiến trình.
    - Nếu tiến trình vẫn còn sống sau timeout (`process.is_alive()`), hãy chấm dứt nó một cách cưỡng bức bằng cách sử dụng `process.terminate()`.
    - Đảm bảo tiến trình luôn được join (`process.join()`) sau khi chấm dứt để ngăn chặn các tiến trình zombie.
5.  **Xử lý Lỗi Rõ ràng:** Nếu một timeout xảy ra, phương thức `execute` nên ném ra một `TimeoutError` tùy chỉnh. Nếu hàm mục tiêu ném ra một ngoại lệ, phương thức `execute` nên ném lại chính xác ngoại lệ đó trong tiến trình cha.
6.  **Giải thích Chi tiết:**
    - Giải thích *tại sao* `multiprocessing` là lựa chọn đúng đắn cho các tác vụ thực sự nặng về CPU, tham chiếu đến Global Interpreter Lock (GIL).
    - Giải thích sự nguy hiểm của `process.terminate()` và tại sao nó cần thiết ở đây (trích dẫn [4]). Thảo luận về các kịch bản mà nó có thể gây ra vấn đề (ví dụ: làm hỏng trạng thái được chia sẻ).
    - Cung cấp một ví dụ rõ ràng về việc sử dụng lớp để chạy một hàm thành công và một hàm khác bị timeout.

---

### **Yêu cầu Cốt lõi 2: Các Chiến lược Hiện đại cho các Tác vụ Nặng về I/O**

Đối với các tác vụ nặng về I/O, việc sử dụng một tiến trình đầy đủ thường quá nặng nề. Giải thích và cung cấp mã cho hai kịch bản chính.

**2.1. I/O Đồng bộ: Timeout Gốc của Thư viện**
- Giải thích rằng phương pháp tốt nhất cho I/O đồng bộ thường là sử dụng tham số timeout được tích hợp sẵn trong thư viện cụ thể.
- Cung cấp một ví dụ mã rõ ràng sử dụng thư viện `requests` [2], minh họa cách đặt `timeout` cho một yêu cầu HTTP GET và cách bắt ngoại lệ `requests.exceptions.Timeout`.

**2.2. I/O Bất đồng bộ: Hủy bỏ An toàn với `asyncio`**
- Giải thích rằng `asyncio` là tiêu chuẩn hiện đại cho I/O hiệu suất cao trong Python.
- Cung cấp một ví dụ có thể chạy được sử dụng `asyncio.timeout()` (trình quản lý ngữ cảnh có sẵn trong Python 3.11+, nhưng cũng giải thích phương án thay thế `asyncio.wait_for()` cho các phiên bản cũ hơn).[2, 6]
- Ví dụ nên hiển thị một hàm `async` mô phỏng một cuộc gọi mạng dài (`await asyncio.sleep(...)`).
- Minh họa cách trình quản lý ngữ cảnh `asyncio.timeout()` ném ra một `asyncio.CancelledError` để ngắt tác vụ một cách an toàn.
- Giải thích *tại sao* đây là "hủy bỏ hợp tác" và cách nó cho phép logic dọn dẹp trong các khối `try...finally`.

---

### **Yêu cầu Cốt lõi 3: Phương pháp `signal` Unix Kế thừa (Để tham khảo)**

Để có bối cảnh lịch sử và giáo dục, hãy giải thích phương pháp dựa trên `signal`.

1.  Cung cấp một ví dụ mã đơn giản, có thể chạy được sử dụng `signal.signal(signal.SIGALRM,...)` và `signal.alarm(...)` để bao bọc một lệnh gọi hàm trong một timeout.[9, 10, 11, 12] Trình xử lý nên ném ra một `TimeoutError` tùy chỉnh.
2.  Trong lời giải thích của bạn, bạn **phải** nêu rõ các hạn chế chính của nó:
    - Nó **chỉ dành cho Unix** và sẽ không hoạt động trên Windows.[13, 14]
    - Nó **không an toàn cho luồng** và phải được sử dụng trong luồng chính.[12]
3.  Kết luận bằng cách nói rằng mặc dù thông minh, phương pháp này thường không được khuyến nghị cho các ứng dụng đa nền tảng, hiện đại.

---

### **Yêu cầu Cốt lõi 4: Phân tích So sánh và các Thực tiễn Tốt nhất**

Kết thúc bài hướng dẫn của bạn bằng một bản tóm tắt giúp các nhà phát triển đưa ra quyết định kiến trúc đúng đắn.

1.  **Tạo một Bảng Markdown:** Tạo một bảng so sánh bốn phương pháp (`multiprocessing`, `asyncio`, `library-native`, `signal`) trên các tiêu chí sau:
    - **Trường hợp Sử dụng** (ví dụ: Nặng về CPU, Nặng về I/O)
    - **Loại Chấm dứt** (ví dụ: Ưu tiên, Hợp tác)
    - **Đa nền tảng** (Có/Không)
    - **An toàn cho Luồng** (Có/Không)
    - **Chi phí (Overhead)** (ví dụ: Cao, Thấp)
2.  **Cung cấp một Danh sách các Thực tiễn Tốt nhất:**
    - Cách chọn một giá trị timeout phù hợp (ví dụ: dựa trên giám sát hiệu suất, các phân vị).[2]
    - Tầm quan trọng của việc ghi nhật ký các timeout như các sự kiện riêng biệt để giám sát và cảnh báo.
    - "Ảo tưởng Timeout" của `concurrent.futures.ThreadPoolExecutor` [7, 8], giải thích rằng `future.result(timeout=...)` không dừng tác vụ và không nên được sử dụng để giới hạn tài nguyên cho các tác vụ chạy dài.

#