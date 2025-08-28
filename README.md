<br>

# F5TTS OpenAI-Compatible API Server (Tiếng Việt)

Dự án này triển khai một máy chủ Text-to-Speech (TTS) hiệu năng cao sử dụng mô hình **F5TTS**, được đóng gói trong một API tương thích với chuẩn của OpenAI. Điều này cho phép dễ dàng tích hợp vào các hệ thống hiện có đang sử dụng API của OpenAI mà không cần thay đổi nhiều ở phía client.

Server được xây dựng bằng **FastAPI**, hỗ trợ streaming audio và được tối ưu để triển khai dễ dàng thông qua **Docker**.

## ✨ Tính năng chính

- **Chất lượng giọng nói cao**: Sử dụng mô hình F5TTS để tạo ra giọng nói tiếng Việt tự nhiên và chất lượng.
- **Tương thích OpenAI**: Cung cấp endpoint `/v1/audio/speech` với cấu trúc request/response tương tự API của OpenAI.
- **Hỗ trợ Streaming**: Trả về âm thanh dưới dạng stream (luồng) để giảm độ trễ và cải thiện trải nghiệm người dùng. Hỗ trợ định dạng `wav` và `pcm`.
- **Bảo mật**: Yêu cầu xác thực bằng API Key qua header `Authorization: Bearer <YOUR_KEY>`.
- **Nhân bản giọng nói (Voice Cloning)**: Dễ dàng thêm giọng nói mới bằng cách tải lên file âm thanh tham chiếu thông qua API.
- **Triển khai dễ dàng**: Cấu hình sẵn sàng để triển khai với Docker và Docker Compose.
- **Quản lý tài nguyên**: Tích hợp cơ chế Semaphore và Lock để quản lý truy cập đồng thời vào model, đảm bảo sự ổn định.

## 🛠️ Công nghệ sử dụng

- **Backend**: FastAPI
- **Model TTS**: F5TTS (chạy trên PyTorch)
- **Containerization**: Docker, Docker Compose
- **Ngôn ngữ**: Python 3.12

## 🚀 Bắt đầu

### Yêu cầu tiên quyết

1.  **Docker** và **Docker Compose**: Cài đặt trên máy của bạn. Đây là phương pháp khuyến khích.
2.  **Git**: Để sao chép mã nguồn.
3.  **Model F5TTS**: Bạn cần có các file của model và đặt chúng vào đúng thư mục.
4.  **Python 3.12** (nếu bạn muốn chạy cục bộ không qua Docker).

### Cấu trúc thư mục cần chuẩn bị

Trước khi chạy, hãy đảm bảo cấu trúc thư mục của bạn trông như sau:

```
.
├── erax-ai_model/            # <-- Thư mục chứa model
│   ├── model_48000.safetensors
│   └── vocab.txt
├── female-vts.wav            # <-- File âm thanh tham chiếu
├── male_south_TEACH_chunk_0_segment_684.wav
├── client.html
├── docker-compose.yml
├── Dockerfile
├── f5tts_wrapper.py
├── main.py                   # <-- Tên file python chính của bạn
└── requirements.txt
```

### Cài đặt & Khởi chạy

#### Lựa chọn 1: Sử dụng Docker (Khuyến khích)

Đây là cách đơn giản và đáng tin cậy nhất để chạy server.

1.  **Sao chép mã nguồn:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Cấu hình API Key:**
    Mở file `docker-compose.yml`. Tìm đến dòng `API_KEY` và thay đổi `your-very-secret-key-12345` thành một khóa bí mật của riêng bạn.
    ```yaml
    version: '3'
    services:
      tts-server:
        build: .
        ports:
          - "8000:8000" # <-- Port bên ngoài là 8000
        environment:
          - API_KEY=thay-the-bang-key-bi-mat-cua-ban
    ```

3.  **Build và chạy container:**
    ```bash
    docker-compose up --build -d
    ```
    Lệnh này sẽ build Docker image và khởi chạy service trong nền.

4.  **Kiểm tra logs (tùy chọn):**
    ```bash
    docker-compose logs -f
    ```

5.  **Dừng server:**
    ```bash
    docker-compose down
    ```

#### Lựa chọn 2: Chạy cục bộ với Python

1.  **Sao chép mã nguồn** và di chuyển vào thư mục dự án.

2.  **Tạo môi trường ảo (khuyến khích):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\Scripts\activate
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Thiết lập biến môi trường API_KEY:**
    *   **Trên Linux/macOS:**
        ```bash
        export API_KEY="thay-the-bang-key-bi-mat-cua-ban"
        ```
    *   **Trên Windows (Command Prompt):**
        ```bash
        set API_KEY="thay-the-bang-key-bi-mat-cua-ban"
        ```    *   **Trên Windows (PowerShell):**
        ```bash
        $env:API_KEY="thay-the-bang-key-bi-mat-cua-ban"
        ```

5.  **Chạy server FastAPI:**
    (Giả sử file của bạn tên là `main.py` và port là `8000`)
    ```bash
    python main.py --host 0.0.0.0 --port 8000
    ```

Server của bạn bây giờ sẽ chạy tại `http://localhost:8000`.

## ⚙️ Cấu hình

Server được cấu hình thông qua các biến môi trường:

| Biến môi trường | Mô tả | Mặc định | Bắt buộc |
| :-------------- | :----------------------------------------------------------------------- | :-------- | :------- |
| `API_KEY` | Khóa bí mật để xác thực các yêu cầu API. | `None` | **Có** |

> **Cảnh báo**: Server sẽ không chấp nhận yêu cầu tới các endpoint được bảo vệ nếu biến môi trường `API_KEY` không được thiết lập.

## API Usage

### Xác thực

Tất cả các yêu cầu đến các endpoint được bảo vệ phải bao gồm header `Authorization`.

-   **Key**: `Authorization`
-   **Value**: `Bearer <YOUR_API_KEY>`

### Ví dụ: Tạo giọng nói (OpenAI Compatible)

Sử dụng `curl` để gửi yêu cầu đến endpoint `/v1/audio/speech`:

```bash
# Thay <YOUR_API_KEY> bằng key bạn đã cấu hình
curl -X POST http://localhost:8000/v1/audio/speech \
-H "Authorization: Bearer <YOUR_API_KEY>" \
-H "Content-Type: application/json" \
-d '{
    "model": "F5TTS_v1_Base",
    "input": "Xin chào thế giới! Đây là một bài kiểm tra API.",
    "voice": "female",
    "response_format": "wav"
}' --output test_audio.wav
```

File `test_audio.wav` sẽ được tạo ra trong thư mục hiện tại của bạn.

### Ví dụ: Tải lên giọng nói mới

Bạn có thể thêm một giọng nói mới một cách linh hoạt bằng cách tải lên một file âm thanh (`.wav` hoặc `.mp3`).

```bash
# Thay <YOUR_API_KEY> bằng key bạn đã cấu hình
curl -X POST http://localhost:8000/v1/upload_reference \
-H "Authorization: Bearer <YOUR_API_KEY>" \
-F "file=@/path/to/your/audio_sample.wav" \
-F "text=Nội dung văn bản tương ứng với file âm thanh."
```

Phản hồi sẽ chứa `ref_id` của giọng nói mới, ví dụ: `custom_1678886400`. Bạn có thể sử dụng `ref_id` này làm giá trị cho trường `voice` trong các yêu cầu tiếp theo.

## Endpoints

Tất cả các endpoint API đều có tiền tố `/v1`.

| Method | Path | Bảo vệ | Mô tả |
| :----- | :-------------------- | :----- | :------------------------------------------------------ |
| `POST` | `/audio/speech` | **Có** | Tạo âm thanh (Endpoint tương thích OpenAI). |
| `POST` | `/tts/stream` | **Có** | Endpoint streaming gốc của F5TTS. |
| `POST` | `/upload_reference` | **Có** | Tải lên một file âm thanh tham chiếu để tạo giọng nói mới. |
| `GET` | `/references` | Không | Lấy danh sách các giọng nói có sẵn và trạng thái của chúng. |
| `GET` | `/health` | Không | Kiểm tra trạng thái hoạt động của server. |
| `GET` | `/` | Không | Giao diện web đơn giản để thử nghiệm. |
# F5TTS OpenAI-Compatible API Server

This project deploys a high-performance Text-to-Speech (TTS) server using the **F5TTS** model, wrapped in an OpenAI-compatible API. This allows for easy integration into existing systems that use the OpenAI API without significant client-side changes.

The server is built with **FastAPI**, supports audio streaming, and is optimized for easy deployment via **Docker**.

## ✨ Key Features

- **High-Quality Voice**: Utilizes the F5TTS model to generate natural and high-quality Vietnamese speech.
- **OpenAI-Compatible**: Provides a `/v1/audio/speech` endpoint with a request/response structure similar to the official OpenAI API.
- **Streaming Support**: Returns audio as a stream to reduce latency and improve user experience. Supports `wav` and `pcm` formats.
- **Secure**: Requires API key authentication via the `Authorization: Bearer <YOUR_KEY>` header.
- **Voice Cloning**: Easily add new voices by uploading a reference audio file via an API endpoint.
- **Easy Deployment**: Ready-to-use configuration for deployment with Docker and Docker Compose.
- **Resource Management**: Integrates Semaphore and Lock mechanisms to manage concurrent access to the model, ensuring stability.

## 🛠️ Technology Stack

- **Backend**: FastAPI
- **TTS Model**: F5TTS (running on PyTorch)
- **Containerization**: Docker, Docker Compose
- **Language**: Python 3.12

## 🚀 Getting Started

### Prerequisites

1.  **Docker** and **Docker Compose**: Installed on your machine. This is the recommended method.
2.  **Git**: To clone the source code.
3.  **F5TTS Model**: You need the model files placed in the correct directory.
4.  **Python 3.12** (if you wish to run locally without Docker).

### Required Directory Structure

Before running, ensure your project directory structure looks like this:

```
.
├── erax-ai_model/            # <-- Directory for the model
│   ├── model_48000.safensors
│   └── vocab.txt
├── female-vts.wav            # <-- Reference audio file
├── male_south_TEACH_chunk_0_segment_684.wav
├── client.html
├── docker-compose.yml
├── Dockerfile
├── f5tts_wrapper.py
├── main.py                   # <-- Your main Python script file
└── requirements.txt
```

### Installation & Launch

#### Option 1: Using Docker (Recommended)

This is the simplest and most reliable way to run the server.

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Configure the API Key:**
    Open the `docker-compose.yml` file. Find the `API_KEY` line and replace `your-very-secret-key-12345` with your own secret key.
    ```yaml
    version: '3'
    services:
      tts-server:
        build: .
        ports:
          - "8000:8000" # <-- External port is 8000
        environment:
          - API_KEY=replace-with-your-secret-key
    ```

3.  **Build and run the container:**
    ```bash
    docker-compose up --build -d
    ```
    This command will build the Docker image and start the service in the background.

4.  **Check the logs (optional):**
    ```bash
    docker-compose logs -f
    ```

5.  **Stop the server:**
    ```bash
    docker-compose down
    ```

#### Option 2: Running Locally with Python

1.  **Clone the repository** and navigate into the project directory.

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set the API_KEY environment variable:**
    *   **On Linux/macOS:**
        ```bash
        export API_KEY="replace-with-your-secret-key"
        ```
    *   **On Windows (Command Prompt):**
        ```bash
        set API_KEY="replace-with-your-secret-key"
        ```
    *   **On Windows (PowerShell):**
        ```bash
        $env:API_KEY="replace-with-your-secret-key"
        ```

5.  **Run the FastAPI server:**
    (Assuming your script is named `main.py` and the port is `8000`)
    ```bash
    python main.py --host 0.0.0.0 --port 8000
    ```

Your server will now be running at `http://localhost:8000`.

## ⚙️ Configuration

The server is configured via environment variables:

| Environment Variable | Description | Default | Required |
| :------------------- | :----------------------------------------------------------------------- | :-------- | :------- |
| `API_KEY` | The secret key to authenticate API requests. | `None` | **Yes** |

> **Warning**: The server will reject requests to protected endpoints if the `API_KEY` environment variable is not set.

## API Usage

### Authentication

All requests to protected endpoints must include the `Authorization` header.

-   **Key**: `Authorization`
-   **Value**: `Bearer <YOUR_API_KEY>`

### Example: Generating Speech (OpenAI Compatible)

Use `curl` to send a request to the `/v1/audio/speech` endpoint:

```bash
# Replace <YOUR_API_KEY> with the key you configured
curl -X POST http://localhost:8000/v1/audio/speech \
-H "Authorization: Bearer <YOUR_API_KEY>" \
-H "Content-Type: application/json" \
-d '{
    "model": "F5TTS_v1_Base",
    "input": "Hello world! This is an API test.",
    "voice": "female",
    "response_format": "wav"
}' --output test_audio.wav
```

The file `test_audio.wav` will be created in your current directory.

### Example: Uploading a New Voice

You can dynamically add a new voice by uploading an audio file (`.wav` or `.mp3`).

```bash
# Replace <YOUR_API_KEY> with the key you configured
curl -X POST http://localhost:8000/v1/upload_reference \
-H "Authorization: Bearer <YOUR_API_KEY>" \
-F "file=@/path/to/your/audio_sample.wav" \
-F "text=The transcribed text corresponding to the audio file."
```

The response will contain the `ref_id` for the new voice, e.g., `custom_1678886400`. You can use this `ref_id` as the `voice` value in subsequent requests.

## Endpoints

All API endpoints are prefixed with `/v1`.

| Method | Path | Protected | Description |
| :----- | :-------------------- | :--------- | :------------------------------------------------------ |
| `POST` | `/audio/speech` | **Yes** | Generate audio (OpenAI-compatible endpoint). |
| `POST` | `/tts/stream` | **Yes** | The native F5TTS streaming endpoint. |
| `POST` | `/upload_reference` | **Yes** | Upload a reference audio file to create a new voice. |
| `GET` | `/references` | No | Get a list of available voices and their status. |
| `GET` | `/health` | No | Check the health status of the server. |
| `GET` | `/` | No | A simple web client for testing. |

---
