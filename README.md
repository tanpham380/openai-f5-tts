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
[![](https://img.shields.io/badge/Download%20Model-HuggingFace-blue)](https://huggingface.co/erax-ai/EraX-Smile-UnixSex-F5/tree/main/models)

### Tải model EraX F5TTS

**🎉 Tự động tải model:** Server sẽ tự động kiểm tra và tải model EraX từ HuggingFace khi khởi động! Không cần cấu hình gì thêm.

**🔄 Quá trình hoạt động:**
1. Khi khởi động, F5TTSWrapper sẽ kiểm tra thư mục `erax-ai_model/`
2. Nếu chưa có model → tự động tải từ `erax-ai/EraX-Smile-UnixSex-F5`
3. Nếu đã có model → sử dụng model cục bộ
4. Model được cache và chỉ tải một lần

**📁 Cấu trúc sau khi tải:**
```
erax-ai_model/
├── model_48000.safetensors
├── vocab.txt
```

**�️ Tùy chọn đường dẫn HuggingFace Cache:**
Bạn có thể chỉ định thư mục cache cho HuggingFace models:
```python
# Trong F5TTSWrapper
wrapper = F5TTSWrapper(hf_cache_dir="./custom_hf_cache")
```

**�🛠️ Tải thủ công (tùy chọn):**
Nếu muốn tải thủ công, bạn có thể tải từ: 
👉 [https://huggingface.co/erax-ai/EraX-Smile-UnixSex-F5/tree/main/models](https://huggingface.co/erax-ai/EraX-Smile-UnixSex-F5/tree/main/models)

**⚙️ Tùy chọn model path tùy chỉnh:**
Bạn có thể sử dụng model từ đường dẫn tùy chỉnh:
```python
# Sử dụng model từ đường dẫn khác
wrapper = F5TTSWrapper(
    ckpt_path="/path/to/your/custom_model.safetensors",
    vocab_file="/path/to/your/custom_vocab.txt"
)
```

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

### 🔧 Cấu hình nâng cao

**📂 Tùy chỉnh đường dẫn model:**
Bạn có thể cấu hình F5TTSWrapper với các đường dẫn tùy chỉnh trong `tts_server.py`:

```python
MODEL_CONFIG = {
    # Tự động tải EraX model (mặc định)
    "ckpt_path": None,  # None = auto-download EraX
    "vocab_file": None, # None = auto-download EraX vocab
    
    # Hoặc sử dụng model tùy chỉnh
    # "ckpt_path": "/path/to/custom_model.safetensors",
    # "vocab_file": "/path/to/custom_vocab.txt",
    
    # Cấu hình cache HuggingFace
    "hf_cache_dir": "./hf_cache",
    
    # Cấu hình device
    "device": "auto",  # auto, cuda, cpu
}
```

**📁 Tùy chỉnh thư mục lưu trữ:**
```python
# Trong tts_server.py
CUSTOM_REF_PATH = "./references"  # Thư mục lưu custom references
REF_AUDIO_CONFIGS = {
    "default_vi": {"path": "./ref_audios/default_vi.wav", "text": "..."},
    # Thêm references mặc định khác
}
```

**🌐 Cấu hình mạng:**
```bash
# Thay đổi host và port
python tts_server.py --host 0.0.0.0 --port 8080

# Hoặc qua Docker
docker run -p 8080:8000 your-tts-image
```

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

### Ví dụ: Tải lên giọng nói mới (Custom Reference)

Bạn có thể thêm một giọng nói mới một cách linh hoạt bằng cách tải lên một file âm thanh (`.wav`, `.mp3`, `.flac`, v.v.).

**📋 Yêu cầu cho Custom Reference:**
- File âm thanh: 5-15 giây, chất lượng tốt
- Định dạng hỗ trợ: `.wav`, `.mp3`, `.flac`, `.ogg`, `.aac`, `.mp4`
- Transcript (tùy chọn nhưng khuyến khích): Nội dung văn bản chính xác của audio

**💻 Sử dụng qua API:**
```bash
# Thay <YOUR_API_KEY> bằng key bạn đã cấu hình
curl -X POST http://localhost:8000/v1/upload_reference \
-H "Authorization: Bearer <YOUR_API_KEY>" \
-F "file=@/path/to/your/audio_sample.wav" \
-F "text=Nội dung văn bản tương ứng với file âm thanh."
```

**🌐 Sử dụng qua Web Interface:**
1. Mở trình duyệt tại `http://localhost:8000`
2. Chuyển sang tab "**Custom Reference**"
3. Chọn file âm thanh (5-15 giây)
4. Nhập transcript (tùy chọn)
5. Nhấn "**Upload & Process**"
6. Giọng nói mới sẽ xuất hiện trong danh sách Voice Reference

**📁 Vị trí lưu trữ:**
Custom references được lưu trong thư mục `./references/` với format:
```
references/
├── custom_1678886400.wav  # ID tự động tạo
├── custom_1678886401.mp3
└── ...
```

**🔄 Sử dụng Custom Reference:**
Sau khi upload thành công, sử dụng `ref_id` được trả về làm giá trị cho trường `voice`:
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
-H "Authorization: Bearer <YOUR_API_KEY>" \
-H "Content-Type: application/json" \
-d '{
    "model": "F5TTS_v1_Base",
    "input": "Văn bản cần tổng hợp giọng nói.",
    "voice": "custom_1678886400",  # <-- Sử dụng ref_id
    "response_format": "wav"
}'
```

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
[![](https://img.shields.io/badge/Download%20Model-HuggingFace-blue)](https://huggingface.co/erax-ai/EraX-Smile-UnixSex-F5/tree/main/models)

### Download EraX F5TTS Model

**🎉 Automatic Model Download:** The server will automatically check and download the EraX model from HuggingFace on startup! No additional configuration needed.

**🔄 How it works:**
1. On startup, F5TTSWrapper checks the `erax-ai_model/` directory
2. If model not found → automatically downloads from `erax-ai/EraX-Smile-UnixSex-F5`
3. If model exists → uses local model
4. Model is cached and downloaded only once

**📁 Directory structure after download:**
```
erax-ai_model/
├── model_48000.safetensors
├── vocab.txt
```

**🗂️ Custom HuggingFace Cache Path:**
You can specify a custom cache directory for HuggingFace models:
```python
# In F5TTSWrapper
wrapper = F5TTSWrapper(hf_cache_dir="./custom_hf_cache")
```

**🛠️ Manual Download (Optional):**
If you prefer manual download, you can get the files from: 
👉 [https://huggingface.co/erax-ai/EraX-Smile-UnixSex-F5/tree/main/models](https://huggingface.co/erax-ai/EraX-Smile-UnixSex-F5/tree/main/models)

**⚙️ Custom Model Path Options:**
You can use models from custom paths:
```python
# Use model from custom path
wrapper = F5TTSWrapper(
    ckpt_path="/path/to/your/custom_model.safetensors",
    vocab_file="/path/to/your/custom_vocab.txt"
)
```

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

### 🔧 Advanced Configuration

**📂 Custom Model Paths:**
You can configure F5TTSWrapper with custom paths in `tts_server.py`:

```python
MODEL_CONFIG = {
    # Auto-download EraX model (default)
    "ckpt_path": None,  # None = auto-download EraX
    "vocab_file": None, # None = auto-download EraX vocab
    
    # Or use custom model
    # "ckpt_path": "/path/to/custom_model.safetensors",
    # "vocab_file": "/path/to/custom_vocab.txt",
    
    # HuggingFace cache configuration
    "hf_cache_dir": "./hf_cache",
    
    # Device configuration
    "device": "auto",  # auto, cuda, cpu
}
```

**📁 Custom Storage Directories:**
```python
# In tts_server.py
CUSTOM_REF_PATH = "./references"  # Directory for custom references
REF_AUDIO_CONFIGS = {
    "default_vi": {"path": "./ref_audios/default_vi.wav", "text": "..."},
    # Add other default references
}
```

**🌐 Network Configuration:**
```bash
# Change host and port
python tts_server.py --host 0.0.0.0 --port 8080

# Or via Docker
docker run -p 8080:8000 your-tts-image
```

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

### Example: Uploading a New Voice (Custom Reference)

You can dynamically add a new voice by uploading an audio file (`.wav`, `.mp3`, `.flac`, etc.).

**📋 Requirements for Custom Reference:**
- Audio file: 5-15 seconds, good quality
- Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.aac`, `.mp4`
- Transcript (optional but recommended): Exact text content of the audio

**💻 Using API:**
```bash
# Replace <YOUR_API_KEY> with the key you configured
curl -X POST http://localhost:8000/v1/upload_reference \
-H "Authorization: Bearer <YOUR_API_KEY>" \
-F "file=@/path/to/your/audio_sample.wav" \
-F "text=The transcribed text corresponding to the audio file."
```

**🌐 Using Web Interface:**
1. Open browser at `http://localhost:8000`
2. Switch to "**Custom Reference**" tab
3. Select audio file (5-15 seconds)
4. Enter transcript (optional)
5. Click "**Upload & Process**"
6. New voice will appear in Voice Reference dropdown

**📁 Storage Location:**
Custom references are saved in `./references/` directory with format:
```
references/
├── custom_1678886400.wav  # Auto-generated ID
├── custom_1678886401.mp3
└── ...
```

**🔄 Using Custom Reference:**
After successful upload, use the returned `ref_id` as the `voice` value:
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
-H "Authorization: Bearer <YOUR_API_KEY>" \
-H "Content-Type: application/json" \
-d '{
    "model": "F5TTS_v1_Base",
    "input": "Text to synthesize speech.",
    "voice": "custom_1678886400",  # <-- Use ref_id
    "response_format": "wav"
}'
```

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

## 🔧 Troubleshooting

### Model Download Issues

**🚫 Download Failed:**
```bash
❌ Failed to download EraX model: HTTP 403 Forbidden
💡 Please manually download from: https://huggingface.co/erax-ai/EraX-Smile-UnixSex-F5/tree/main/models
```
**Solution:** Manually download and place files in `erax-ai_model/` directory.

**🐌 Slow Download:**
- Use custom HuggingFace cache: `hf_cache_dir="./hf_cache"`
- Check internet connection
- Try downloading during off-peak hours

### Custom Reference Issues

**📄 Unsupported File Format:**
```bash
❌ Error: Unsupported audio format
```
**Solution:** Convert to supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.aac`, `.mp4`

**⏱️ Audio Too Long/Short:**
```bash
❌ Audio duration should be 5-15 seconds
```
**Solution:** Trim audio to recommended length for best results.

**🔊 Poor Audio Quality:**
- Use high-quality audio (44.1kHz or 48kHz)
- Avoid background noise
- Ensure clear speech

### Performance Issues

**💾 Out of Memory:**
- Reduce `nfe_step` parameter (default: 32)
- Use smaller batch sizes
- Enable CPU offloading

**🐢 Slow Generation:**
- Use GPU if available
- Reduce `cfg_strength` for faster generation
- Optimize `cross_fade_duration`

---
