# AISmartMixer: 智能音视频匹配与片段导出工具【匹配VO与切片画面】 🎙️✂️🎞️

AISmartMixer 旨在自动化音视频内容处理流程。它能够分析主要的音频旁白，将其分割成带有时间戳的文本片段；同时分析视频素材库，提取视频的描述信息并生成特征向量（ embeddings）。然后，通过计算音频片段与视频片段之间的语义相似度，为每个音频片段匹配最相关的视频素材。最终，系统会导出一个包含这些匹配信息的编辑决策列表 (EDL)，并能够将最匹配的视频片段（或其子片段）导出到独立的文件夹中，方便后续的人工审查或视频编辑。

## 核心功能 ✨

*   **音频分析与转录**:
    *   使用 OpenAI Whisper 模型将输入的音频文件（如旁白、画外音）精准转录为带时间戳的文本片段。
*   **视频内容分析与嵌入**:
    *   扫描用户提供的视频素材库。
    *   从视频文件名中提取描述性文本 (当前版本主要依赖文件名结构进行解析)。
    *   使用 Sentence Transformers 模型 (如 LaBSE, all-MiniLM-L6-v2) 为视频描述生成语义嵌入向量。
    *   获取视频时长等元数据。
*   **语义匹配**:
    *   为每个音频文本片段生成语义嵌入向量。
    *   计算音频片段嵌入与视频片段嵌入之间的余弦相似度。
    *   为每个音频片段找出语义上最匹配的 Top-K 个视频片段。
*   **时间线/EDL 生成**:
    *   根据匹配结果生成一个编辑决策列表 (EDL)，详细记录每个音频片段及其对应的 Top-K 视频匹配（包含视频ID、文件名、相似度得分、建议的片段时长等）。
    *   支持视频重用策略配置 (如不重用、允许在不同音频段重用等)。
*   **视频片段导出**:
    *   根据生成的 EDL，将每个音频片段的最佳匹配视频（Top-N）导出到独立的文件夹中。
    *   支持导出原始视频的子片段 (subclip) 以精确匹配音频时长，并可配置前后填充时间。
    *   导出的文件名可包含排序、相似度得分等信息。
    *   自动生成 HTML 索引页面，方便快速预览和审查导出的视频片段。
*   **用户工作区管理**:
    *   为不同用户创建独立的工作目录 (`user/<username>/`)，包含各自的输入数据、处理中间文件和输出结果。
*   **灵活配置**:
    *   通过 `config.yaml` 文件集中管理所有路径、模型名称、处理参数、日志级别等。
*   **模块化设计**: 代码结构清晰，分为音频/视频分析、匹配、时间线生成、片段导出和工具类等模块。
*   **日志系统**: 详细记录程序运行过程中的信息、警告和错误，便于追踪和调试。

## 工作流程 ⚙️

AISmartMixer 的处理流程主要分为以下几个阶段：

1.  **阶段 0: 初始化与配置**
    *   用户通过命令行启动 `main.py`。
    *   程序加载 `config.yaml` 配置文件。
    *   提示用户输入用户名，并根据用户名创建或定位用户专属的工作目录。
    *   设置日志系统。
    *   （可选）清空用户上一次运行的输出文件夹。

2.  **阶段 1: 分析 (Analyzer - `phase_1_analyzer.py`)**
    *   **音频分析 (`AudioAnalyzer`)**:
        *   读取用户指定的音频文件 (例如 `user/<username>/data/input_audio/你的vo音频文件.mp3`)。
        *   使用 Whisper 模型进行语音转文字，生成包含各语音片段文本、开始和结束时间的转录数据。
        *   结果保存为 JSON 文件 (例如 `audio_transcription.json`)。
    *   **视频分析 (`VideoAnalyzer`)**:
        *   扫描用户指定的视频素材库文件夹 (例如 `user/<username>/data/input_video_clips/`)。
        *   对于每个视频文件：
            *   从文件名解析出描述性文本。 **(重要: 当前版本主要依赖文件名包含描述信息，例如 `片段XXX-描述A-描述B.mp4` 的格式)**
            *   使用 Sentence Transformer 模型为该描述文本生成语义嵌入向量。
            *   获取视频时长等元数据。
        *   所有视频的元数据和嵌入向量保存为 JSON 文件 (例如 `video_metadata_embeddings.json`)。

3.  **阶段 2: 匹配 (Matcher - `phase_2_matcher.py`)**
    *   加载音频转录数据和视频元数据（包含嵌入向量）。
    *   对于音频转录中的每一个文本片段：
        *   （如果未在阶段1生成）为其生成语义嵌入向量。
        *   计算其与视频库中所有视频片段描述的嵌入向量之间的余弦相似度。
    *   此阶段的核心是 `Matcher` 类，它为后续的 `TimelineGenerator` 提供按需匹配服务。

4.  **阶段 3: 时间线生成 (Timeline Generator - `phase_3_timeline_generator.py`)**
    *   `TimelineGenerator` 接收音频转录和视频元数据。
    *   对于每个音频片段，调用 `Matcher` 找到 Top-K (例如，默认为5个，由 `config.yaml` 中 `settings.top_k_candidates` 控制) 最相似的视频片段。
    *   考虑视频重用策略 (例如，一个视频片段是否可以在多个音频片段中使用，或是否可以重复使用)。
    *   生成一个编辑决策列表 (EDL)，这是一个 JSON 文件 (例如 `final_edit_list.json`)，其中每个条目对应一个音频片段，并列出其匹配的 Top-K 视频片段的详细信息（ID、文件名、相似度得分、建议的视频片段时长等）。

5.  **阶段 4: 片段导出 (Clip Exporter - `phase_4_clip_exporter.py`)**
    *   `ClipExporter` 读取生成的 EDL。
    *   对于 EDL 中的每一个音频片段及其匹配的视频列表：
        *   为该音频片段创建一个专属的输出子文件夹 (例如 `user/<username>/output/exported_clips/segment_001_天气真好/`)。
        *   将 Top-N (由 `config.yaml` 中 `clip_export.max_clips_per_segment` 控制) 个匹配度最高的视频片段复制或提取子片段到该文件夹。
            *   如果 `extract_subclips` 配置为 `true`，则会尝试使用 `moviepy` 提取与音频片段时长相匹配的视频子片段（可加前后 `padding_seconds`）。
            *   导出的文件名会包含排序编号、相似度得分（可选）和原始视频文件名的一部分。
        *   为每个子文件夹生成一个 `segment_metadata.json`，记录该音频段导出的所有片段信息。
    *   在 `exported_clips` 根目录下生成一个 `preview_metadata.json` 和一个 `index.html` 文件，用于在浏览器中快速预览所有导出的片段。

## 项目结构 📁

    AISmartMixer/
    ├── src/                          # 源代码目录 (应作为 Python 包运行)
    │   ├── __init__.py
    │   ├── main.py                   # 程序主入口
    │   ├── phase_1_analyzer.py       # 音频和视频分析模块
    │   ├── phase_2_matcher.py        # 语义匹配模块
    │   ├── phase_3_timeline_generator.py # 时间线/EDL生成模块
    │   ├── phase_4_clip_exporter.py  # 视频片段导出模块
    │   └── utils/                    # 工具类模块
    │       ├── __init__.py
    │       ├── config_loader.py      # YAML 配置文件加载
    │       ├── file_handler.py       # JSON 文件读写、目录创建
    │       └── logger_setup.py       # 日志系统设置
    ├── config.yaml                   # 主配置文件
    ├── requirements.txt              # (建议创建) Python 依赖列表
    └── user/                         # 用户工作区根目录 (自动创建)
        └── <username>/               # 特定用户的工作目录 (根据输入创建)
            ├── data/                 # 用户输入数据
            │   ├── input_audio/      # 存放输入的音频文件
            │   │   └── 你的vo音频文件.mp3 (示例)
            │   └── input_video_clips/ # 存放视频素材片段
            │       └── video1-描述.mp4 (示例)
            │       └── video2-描述.mp4 (示例)
            └── output/               # 用户输出结果
                ├── processed_data/   # 处理后的中间数据
                │   ├── audio_transcription.json
                │   └── video_metadata_embeddings.json
                ├── edl/              # 编辑决策列表
                │   └── final_edit_list.json
                ├── exported_clips/   # 导出的视频片段
                │   ├── segment_001_音频文本描述/
                │   │   ├── 01_score_0.95_video1-描述.mp4
                │   │   ├── 02_score_0.92_videoX-描述.mp4
                │   │   └── segment_metadata.json
                │   ├── segment_002_另一段音频描述/
                │   │   └── ...
                │   ├── index.html
                │   └── preview_metadata.json
                ├── final_video/      # (预留, 当前主要用于存放最终合成视频的目录，但此代码库侧重片段导出)
                └── app.log           # 应用程序日志文件


## 环境依赖 🛠️

*   **Python**: 3.8 或更高版本。
*   **核心 Python 库**:
    *   `openai-whisper`: 用于音频转录。
    *   `sentence-transformers`: 用于生成文本的语义嵌入向量。
    *   `moviepy`: 用于获取视频时长和提取视频子片段。
    *   `PyYAML`: 用于加载 `config.yaml` 配置文件。
    *   `numpy`: 用于数值计算，特别是处理嵌入向量。
*   **(可选但推荐) FFmpeg**: `moviepy` 和 `openai-whisper` 在某些情况下可能依赖系统中正确安装的 FFmpeg 来处理各种音视频编解码格式。建议安装。

## 安装与配置 ⚙️

1.  **克隆代码库**:
    ```bash
    git clone <your-repository-url>
    cd AISmartMixer
    ```

2.  **创建并激活虚拟环境 (推荐)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

3.  **安装 Python 依赖**:
    建议在项目根目录创建一个 `requirements.txt` 文件，内容如下：
    ```txt
    openai-whisper
    sentence-transformers
    moviepy
    PyYAML
    numpy
    ```
    然后运行:
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置文件 (`config.yaml`)**:
    *   项目根目录下应包含一个 `config.yaml` 文件。您可以基于提供的示例文件进行修改。
    *   **重要**:
        *   检查并根据需要修改 `paths` 部分，确保它们符合您的预期（尽管 `main.py` 会基于用户名动态构建绝对路径）。
        *   `models.whisper_model_name`: 选择合适的 Whisper 模型 (如 `tiny`, `base`, `small`, `medium`, `large-v3`)。首次运行时会自动下载。
        *   `models.embedding_model_name`: 选择合适的 Sentence Transformer 模型 (如 `LaBSE`, `all-MiniLM-L6-v2`)。首次运行时会自动下载。
        *   `settings.top_k_candidates` 和 `clip_export.max_clips_per_segment` 通常应保持一致，决定了为每个音频段匹配和导出多少个视频片段。
        *   `clip_export.video_path_map`: 如果您的视频素材库结构复杂，或者 `ClipExporter` 无法自动找到视频文件，可以在 `config.yaml` 中预定义一个视频文件名到其绝对路径的映射。例如：
            ```yaml
            clip_export:
              # ... 其他设置 ...
              video_path_map:
                "my_complex_video_name_001.mp4": "/mnt/nas/archive/project_x/vid001.mp4"
                "another_clip.mov": "D:/stock_footage/nature/waterfall.mov"
            ```

5.  **Google API 密钥 (idc-ipc-1dc332fa2fe3.json)**:
    *   您提供的 `idc-ipc-1dc332fa2fe3.json` 文件是一个 Google Cloud 服务账号密钥。
    *   **当前 AISmartMixer 代码库 (Whisper + Sentence Transformers) 本身不直接使用 Google Cloud 服务 (如 Gemini)。因此，此密钥文件对于运行此特定代码库不是必需的。**
    *   如果您计划将此代码库与其他使用 Google Cloud 服务的模块集成，则可能需要它。

## 使用方法 🚀

1.  **准备输入数据**:
    *   **音频文件**: 将您的主要音频旁白文件 (例如，一个 `.mp3` 文件) 放置到 `user/<您的用户名>/data/input_audio/` 目录下。确保 `config.yaml` 中的 `paths.input_audio_filename` 与您的文件名一致，或者在程序运行时选择。
    *   **视频素材库**: 将您的视频片段素材文件放置到 `user/<您的用户名>/data/input_video_clips/` 目录下。
        *   **文件名规范**: 为了获得最佳的语义匹配效果，建议视频文件名包含对视频内容的简短描述。例如，`海边日落美景.mp4` 比 `video_001.mp4` 效果更好，因为 `VideoAnalyzer` 当前主要从文件名提取描述文本。当前版本会尝试从形如 `xxx-无人物-有产品-具体描述文字.mp4` 的文件名中提取“具体描述文字”部分。

2.  **运行主程序**:
    确保您的终端当前目录位于项目根目录 (`AISmartMixer/`)。然后执行：
    ```bash
    python -m src.main
    ```
    *   **用户名输入**: 程序会首先提示您输入一个英文用户名。这将用于创建或访问您的个人工作区。
    *   **音频文件选择**: 如果用户的 `input_audio` 文件夹中存在多个 `.mp3` 文件，程序会列出它们并让用户选择一个进行处理。
    *   **命令行参数 (可选)**:
        *   `--config <路径>`: 指定 `config.yaml` 文件的路径 (默认为项目根目录下的 `config.yaml`)。
        *   `--force-reanalyze`: 强制重新进行音频和视频分析，即使已存在相应的 `.json` 中间文件。
        *   `--skip-clear`: 启动时不自动清空用户 `output` 目录下的子文件夹内容。
        *   `--debug`: 启用更详细的 DEBUG 级别日志输出。
        *   `--username <用户名>`: 直接指定用户名，跳过交互式输入提示 (方便脚本自动化)。

3.  **处理过程**:
    *   程序将按顺序执行分析、匹配、时间线生成和片段导出等阶段。
    *   详细的日志信息会输出到控制台和用户工作区下的 `output/app.log` 文件。

4.  **查看输出结果**:
    处理完成后，所有输出文件将位于 `user/<您的用户名>/output/` 目录下的相应子文件夹中：
    *   `processed_data/`: 包含 `audio_transcription.json` 和 `video_metadata_embeddings.json`。
    *   `edl/`: 包含 `final_edit_list.json`。
    *   `exported_clips/`: 包含为每个音频片段导出的视频片段，以及预览用的 `index.html`。打开 `index.html` 可以在浏览器中方便地查看和播放导出的片段。
    *   `app.log`: 完整的应用程序日志。

## 输出说明 📊

*   **`user/<username>/output/processed_data/audio_transcription.json`**:
    Whisper 生成的音频转录结果，包含完整的文本以及每个语音片段的开始时间、结束时间和文本内容。
*   **`user/<username>/output/processed_data/video_metadata_embeddings.json`**:
    视频库中每个视频的元数据列表，包括文件名、路径、从文件名提取的描述、语义嵌入向量和视频时长。
*   **`user/<username>/output/edl/final_edit_list.json`**:
    编辑决策列表。这是一个 JSON 数组，每个元素代表一个音频片段事件，并包含：
    *   `audio_segment_id`, `audio_start_time`, `audio_end_time`, `audio_text`
    *   `event_type`: 通常是 `video_match` 或 `match_failed`。
    *   `video_segments`: 一个列表，包含为该音频片段匹配到的 Top-K 个视频片段的信息：
        *   `video_id`, `filename`, `similarity_score`, `start_time` (通常为0，表示从视频头开始), `duration` (建议的片段时长), `original_video_full_duration`。
*   **`user/<username>/output/exported_clips/`**:
    *   **`<segment_id>_< sanitized_audio_text >/` (子文件夹)**: 每个音频片段对应一个子文件夹。
        *   **导出的视频文件**: 例如 `01_score_0.98_original_video_name.mp4`。文件名包含排序、相似度得分和原始文件名的一部分。这些是实际的视频片段（可能是子剪辑）。
        *   **`segment_metadata.json`**: 包含该音频片段导出的所有视频片段的详细元数据。
    *   **`index.html`**: 一个 HTML 文件，用于在浏览器中预览 `exported_clips` 文件夹下的所有导出片段。
    *   **`preview_metadata.json`**: `index.html` 使用的元数据源。
*   **`user/<username>/output/app.log`**:
    详细的应用程序运行日志。

## 配置详解 (`config.yaml`) ⚙️📄

`config.yaml` 文件是 AISmartMixer 的核心配置文件。以下是一些关键配置项的说明：

*   **`paths`**: 定义所有输入和输出的相对路径（相对于用户工作区内的 `data` 或 `output` 目录）和主要文件名。
    *   `input_audio_filename`: 要处理的主音频文件名。
    *   `input_video_dir`: 存放视频素材片段的目录名。
    *   `output_clips_dir`: 导出匹配视频片段的总目录名。
*   **`models`**:
    *   `whisper_model_name`: 指定 Whisper 模型的大小 (例如: "base", "small", "medium", "large-v3")。
    *   `embedding_model_name`: 指定 Sentence Transformer 模型的名称 (例如: "LaBSE", "all-MiniLM-L6-v2", "shibing624/text2vec-base-chinese")。
*   **`analysis`**:
    *   `recursive_video_search`: 是否递归搜索 `input_video_dir` 下的子目录以查找视频文件。
    *   `audio_language_code`: (可选) 指定音频的语言代码 (如 "en", "zh")，不指定则 Whisper 会自动检测。
*   **`settings`**:
    *   `video_extensions`: VideoAnalyzer 扫描视频文件时使用的扩展名列表。
    *   `min_clip_duration_for_timeline`: 音频片段的最小有效时长，短于此时长的片段在生成时间线时可能被忽略。
    *   `similarity_threshold`: 相似度阈值。在 `TimelineGenerator` 中，如果设置为 `-1.0`（如当前配置），则主要依赖 `top_k_candidates` 来选择，不过滤低分匹配。如果设置为一个正值 (0到1之间)，则只有相似度高于此阈值的视频才会被考虑。
    *   `top_k_candidates`: 对于每个音频片段，期望匹配器找出并由时间线生成器记录的候选视频数量。
    *   `video_reuse_strategy`: 视频重用策略。
        *   `"no_reuse"`: 一个视频片段在整个项目中最多只使用一次。
        *   `"reuse_different_segments"`: 一个视频片段可以在不同的音频片段中使用，但在同一个音频片段的 Top-K 列表中不应重复出现（此策略在当前实现中主要通过 `Matcher` 返回不同视频来间接实现）。
        *   `"allow_full_reuse"`: 允许视频片段被任意重复使用。
    *   `fallback_strategy`: 当没有找到合适的视频匹配时的回退策略 (当前配置为 `"blank"`，意味着如果一个音频段没有视频匹配，EDL中对应的视频列表会为空)。
*   **`clip_export`**:
    *   `max_clips_per_segment`: 为每个音频片段实际导出的视频数量上限。
    *   `include_score_in_filename`: 是否在导出的视频文件名中包含相似度得分。
    *   `max_folder_name_length`: 导出的片段子文件夹名称的最大长度。
    *   `max_filename_length`: 导出的视频文件名（不含路径）的最大长度。
    *   `extract_subclips`: 是否尝试从原始视频中提取与音频片段时长匹配的子片段。如果为 `false`，则复制整个匹配的视频文件。
    *   `padding_seconds`: 如果提取子片段，可以在片段的开始和结束处添加的额外时长（秒）。
    *   `create_preview_metadata`: 是否生成 `index.html` 和 `preview_metadata.json` 用于片段预览。
    *   `video_path_map`: (可选) 手动提供视频文件名到其完整路径的映射，用于 `ClipExporter` 在无法自动定位视频文件时查找。
*   **`logging`**:
    *   `level`: 日志级别 (例如 "DEBUG", "INFO", "WARNING", "ERROR")。
    *   `log_file`: 日志文件的名称 (会保存在用户工作区的 `output` 目录下)。
*   **`force_reanalyze`**: (布尔值) 如果为 `true`，程序启动时会忽略已存在的 `audio_transcription.json` 和 `video_metadata_embeddings.json` 文件，强制重新进行音频和视频分析。也可以通过命令行参数 `--force-reanalyze` 启用。

## 自定义与扩展 🛠️

*   **更改嵌入模型**: 在 `config.yaml` 中修改 `models.embedding_model_name`。确保您选择的模型与您的文本语言和应用场景相匹配。例如，处理中文内容时，`shibing624/text2vec-base-chinese` 是一个不错的选择。
*   **优化视频描述提取**: 当前 `VideoAnalyzer` 主要从文件名提取视频描述。您可以修改 `VideoAnalyzer.analyze_videos` 方法，从其他来源（如视频元数据文件、视频内容本身通过图像识别等）获取更丰富的描述。
*   **调整匹配参数**: 实验不同的 `similarity_threshold` 和 `top_k_candidates` 值，以找到最适合您素材的匹配效果。
*   **修改导出逻辑**: `ClipExporter` 中的文件名生成、子片段提取逻辑等都可以根据具体需求进行调整。
*   **集成其他AI服务**: 可以扩展项目以集成如场景识别、对象检测等其他 AI 服务，为视频打上更丰富的标签，从而改进匹配的维度。

## 故障排查与注意事项 ⚠️

*   **库安装问题**: 确保所有依赖库都已正确安装。特别是 `openai-whisper` 和 `sentence-transformers` 可能需要下载较大的模型文件。`moviepy` 可能需要系统中存在 FFmpeg。
*   **模型下载**: 首次运行或更改模型时，Whisper 和 Sentence Transformers 会自动下载所需的模型文件，这可能需要一些时间，具体取决于您的网络连接和模型大小。请耐心等待。
*   **视频文件路径**: `ClipExporter` 依赖于能够正确定位原始视频文件。如果视频库结构复杂，或文件名在处理过程中发生变化，可能需要使用 `config.yaml` 中的 `clip_export.video_path_map` 来手动指定路径。
*   **文件名和路径长度**: 操作系统对文件名和路径总长度有限制。如果您的音频描述非常长，或者视频素材库嵌套层级很深，可能会导致导出的文件名或路径过长而出错。`config.yaml` 中的 `max_folder_name_length` 和 `max_filename_length` 设置有助于缓解此问题。
*   **性能**:
    *   音频转录（尤其是使用较大的 Whisper 模型）和批量视频嵌入生成可能非常耗时。
    *   处理大量视频素材时，请确保有足够的磁盘空间和计算资源。
*   **MoviePy 与 FFmpeg**: 如果 `moviepy` 在提取子片段或获取视频时长时出错，通常与 FFmpeg 未正确安装或配置有关。
*   **编码问题**: 所有文本处理和文件读写都默认使用 UTF-8 编码。确保您的输入文件名、音频内容等也使用一致的编码。



