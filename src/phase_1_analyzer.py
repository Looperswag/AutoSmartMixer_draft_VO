# src/phase_1_analyzer.py

import os
import logging
import time
import numpy as np
import glob

try:
    import whisper
except ImportError:
    print("Whisper library not found. Please install it: pip install openai-whisper")
    whisper = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("SentenceTransformers library not found. Please install it: pip install sentence-transformers")
    SentenceTransformer = None

from .utils.file_handler import save_json, load_json # Corrected import path

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """
    Analyzes the main audio narration to produce a transcription with timestamps.
    """
    def __init__(self, config):
        """
        Initializes the AudioAnalyzer.

        Args:
            config (dict): The application configuration dictionary.
        """
        self.config = config
        self.audio_model_name = self.config.get("analysis", {}).get("audio_model", "base")
        self.whisper_model = None

        if whisper:
            try:
                logger.info(f"Loading Whisper audio model: {self.audio_model_name}...")
                self.whisper_model = whisper.load_model(self.audio_model_name)
                logger.info(f"Whisper model '{self.audio_model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Whisper model '{self.audio_model_name}': {e}")
                self.whisper_model = None
        else:
            logger.error("Whisper library is not available. Audio analysis will not work.")

    def analyze(self):
        """
        Transcribes the input audio file.

        Returns:
            dict: A dictionary containing the transcription results (text, segments with timestamps),
                  or None if analysis fails.
        """
        if not self.whisper_model:
            logger.error("Whisper model not loaded. Cannot perform audio analysis.")
            return None

        audio_input_dir = self.config["paths"]["input_audio_dir"]
        audio_filename = self.config["paths"]["input_audio_filename"]
        audio_path = os.path.join(audio_input_dir, audio_filename)

        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None

        logger.info(f"Starting audio transcription for: {audio_path}")
        start_time = time.time()
        try:
            # Set language if specified in config, otherwise Whisper will auto-detect
            language_code = self.config.get("analysis", {}).get("audio_language_code") # e.g., "en", "es"
            task = self.config.get("analysis", {}).get("audio_transcription_task", "transcribe") # "transcribe" or "translate"

            transcription_options = {"language": language_code, "task": task}
            # Remove None values from options
            transcription_options = {k: v for k, v in transcription_options.items() if v is not None}


            result = self.whisper_model.transcribe(audio_path, **transcription_options)
            # Whisper's result is a dictionary with keys like 'text', 'segments', 'language'
            # 'segments' is a list of dicts, each with 'start', 'end', 'text'

            # Ensure segments have unique IDs if not already present (Whisper usually provides them)
            for i, segment in enumerate(result.get("segments", [])):
                if "id" not in segment:
                    segment["id"] = i

            end_time = time.time()
            logger.info(f"Audio transcription completed in {end_time - start_time:.2f} seconds.")
            logger.debug(f"Transcription result: {result['text'][:100]}...") # Log first 100 chars
            return result
        except Exception as e:
            logger.error(f"Error during audio transcription: {e}", exc_info=True)
            return None

    def save_transcription(self, transcription_data, output_path):
        """
        Saves the transcription data to a JSON file.

        Args:
            transcription_data (dict): The transcription data from analyze().
            output_path (str): Path to save the JSON file.

        Returns:
            bool: True if successful, False otherwise.
        """
        return save_json(transcription_data, output_path)

    def load_transcription(self, input_path):
        """
        Loads transcription data from a JSON file.

        Args:
            input_path (str): Path to the JSON file.

        Returns:
            dict: Loaded transcription data, or None if loading fails.
        """
        return load_json(input_path)



class VideoAnalyzer:
    def __init__(self, config, embedding_model):
        """
        Initializes the VideoAnalyzer.
        Args:
            config (dict): The application configuration.
            embedding_model: The pre-loaded sentence transformer model.
        """
        self.config = config
        self.video_dir = config.get("paths", {}).get("input_video_dir")
        self.video_extensions = config.get("settings", {}).get("video_extensions", ["*.mp4"])
        
        analysis_settings = config.get("analysis", {})
        self.recursive_search = analysis_settings.get("recursive_video_search", False)
        
        self.embedding_model = embedding_model
        if not self.video_dir:
            logger.error("Video input directory 'paths.input_video_dir' not configured.")
            raise ValueError("Video input directory not configured.")
        if not self.embedding_model:
            logger.error("Embedding model not provided to VideoAnalyzer.")
            raise ValueError("Embedding model not provided.")

    def _scan_for_videos(self):
        """Scans the video directory for video files based on configured extensions."""
        video_files = []
        logger.info(f"Scanning for videos in '{self.video_dir}' (recursive: {self.recursive_search}) with extensions: {self.video_extensions}")
        for ext_pattern in self.video_extensions:
            if self.recursive_search:
                search_pattern = os.path.join(self.video_dir, "**", ext_pattern)
                found = glob.glob(search_pattern, recursive=True)
            else:
                search_pattern = os.path.join(self.video_dir, ext_pattern)
                found = glob.glob(search_pattern, recursive=False)
            
            video_files.extend(found)
        
        if video_files:
            logger.info(f"Found {len(video_files)} video files.")
        else:
            # 这个警告现在由 analyze_videos 在末尾处理，或者可以保留
            # logger.warning(f"No video files found with specified extensions in {self.video_dir}")
            pass
        return list(set(video_files))

    def analyze_videos(self):
        logger.info(f"Starting video analysis in directory: {self.video_dir}")
        logger.info(f"Scanning for videos in '{self.video_dir}' (recursive: {self.recursive_search}) with extensions: {self.video_extensions}")
        
        video_files = self._scan_for_videos()
        video_metadata_list = []

        if not video_files:
            logger.warning(f"No video files found in '{self.video_dir}' with specified extensions. Video analysis will yield no metadata.")
            return []

        config_video_input_dir_abs = os.path.abspath(self.video_dir)

        for video_path in video_files:
            try:
                filename_with_ext = os.path.basename(video_path)
                filename_no_ext = os.path.splitext(filename_with_ext)[0]
                
                meaningful_description = ""

                # 正则表达式匹配模式: 尝试找到最有意义的部分
                import re
                # 模式: 提取"-无人物-有产品-"或类似部分之后的所有内容
                pattern = r'.*?-(?:无人物|有人物|有产品|无产品)-(?:无人物|有人物|有产品|无产品)-(.*)'
                match = re.search(pattern, filename_no_ext)
                
                if match and match.group(1):
                    # 提取到了有意义的描述部分
                    meaningful_description = match.group(1)
                    # 替换下划线为空格，美化描述
                    meaningful_description = meaningful_description.replace("_", " ")
                    logger.debug(f"从文件名 '{filename_no_ext}' 提取到的有意义描述: '{meaningful_description}'")
                else:
                    # 如果正则表达式没有匹配成功，尝试使用简单的分割
                    parts = filename_no_ext.split('-')
                    if len(parts) >= 4:  # 假设描述部分从第4个部分开始
                        meaningful_description = '-'.join(parts[3:])
                        meaningful_description = meaningful_description.replace("_", " ")
                    else:
                        # 回退到使用完整文件名（清理后）
                        meaningful_description = filename_no_ext.replace("_", " ").replace("-", " ")
                        logger.warning(f"无法从 '{filename_no_ext}' 提取出有意义的描述部分，使用完整文件名")
                
                # 确保描述不为空
                if not meaningful_description.strip():
                    meaningful_description = filename_no_ext
                    logger.warning(f"提取的描述为空，回退到使用文件名: '{meaningful_description}'")
                
                # 计算视频时长（如果可能）
                video_duration = None
                try:
                    from moviepy.editor import VideoFileClip
                    with VideoFileClip(video_path) as clip:
                        video_duration = clip.duration
                    logger.debug(f"视频 {filename_with_ext} 的时长: {video_duration:.2f}s")
                except Exception as e:
                    logger.warning(f"无法获取视频 {filename_with_ext} 的时长: {e}. 时长信息将不可用。")
                
                # 生成嵌入向量
                embedding = self.embedding_model.encode(meaningful_description).tolist()
                
                video_metadata_list.append({
                    "id": filename_with_ext,  
                    "filename": filename_with_ext,
                    "path": video_path,
                    "description": meaningful_description, # 使用提取的有意义描述
                    "full_name": filename_no_ext, # 保留完整文件名以供参考
                    "embedding": embedding,
                    "duration": video_duration  # 添加时长信息
                })
                logger.debug(f"处理并嵌入: {filename_with_ext} 描述: '{meaningful_description}' id: '{filename_with_ext}'")

            except Exception as e:
                logger.error(f"处理视频文件 {video_path} 失败: {e}", exc_info=True)
        
        if video_metadata_list:
            logger.info(f"成功分析并生成了 {len(video_metadata_list)} 个视频片段的元数据。")
        else:
            if video_files: 
                logger.error("发现视频文件，但未能为任何文件生成元数据。")
            
        return video_metadata_list





    
    def save_video_metadata(self, video_metadata_list, output_path):
        """Saves the video metadata list to a JSON file."""
        from .utils.file_handler import save_json 
        if save_json(video_metadata_list, output_path):
            logger.info(f"Video metadata and embeddings saved to {output_path}")
        else:
            logger.error(f"Failed to save video metadata to {output_path}")

    def load_video_metadata(self, input_path):
        """Loads video metadata from a JSON file."""
        from .utils.file_handler import load_json
        data = load_json(input_path)
        if data is not None:
            logger.info(f"Video metadata successfully loaded from {input_path}")
            return data
        else:
            logger.error(f"Failed to load video metadata from {input_path}")
            return None



if __name__ == "__main__":
    # This is for basic testing of the analyzers.
    # A more complete test would involve creating a dummy config and dummy files.
    print("Testing AISmartMixer Analyzers (requires dummy config and data for full test)")

    # Setup basic logging for testing this module directly
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # --- Dummy Config for Testing ---
    dummy_config_for_analyzer = {
        "paths": {
            "input_audio_dir": "temp_test_data/audio_in",
            "input_audio_filename": "sample_audio.mp3", # Needs a real (short) audio file
            "input_video_clips_dir": "temp_test_data/videos_in", # Needs some dummy video files
            "output_processed_data_dir": "temp_test_data/processed",
            "audio_transcription_json": "audio_transcription_test.json",
            "video_metadata_embeddings_json": "video_metadata_test.json"
        },
        "analysis": {
            "audio_model": "tiny",  # Use a small model for quick testing
            "video_embedding_model": "all-MiniLM-L6-v2", # A common, relatively small model
            "video_file_extensions": [".mp4", ".txt"], # .txt for easy dummy file creation
            "audio_language_code": "en", # Specify English for testing
            "recursive_video_search": False
        }
    }

    # --- Create dummy directories and files for testing ---
    os.makedirs(os.path.join("temp_test_data", "audio_in"), exist_ok=True)
    os.makedirs(os.path.join("temp_test_data", "videos_in"), exist_ok=True)
    os.makedirs(os.path.join("temp_test_data", "processed"), exist_ok=True)

    # Create a dummy audio file (e.g., a silent or very short mp3)
    # For a real test, place a small mp3 file here: temp_test_data/audio_in/sample_audio.mp3
    # As a placeholder, let's create an empty file if whisper can handle it or skip audio test
    dummy_audio_path = os.path.join(dummy_config_for_analyzer["paths"]["input_audio_dir"],
                                    dummy_config_for_analyzer["paths"]["input_audio_filename"])
    if not os.path.exists(dummy_audio_path):
        with open(dummy_audio_path, "w") as f: # Creating an empty file, Whisper might fail.
            f.write("This is not a real audio file.") # Whisper will likely fail, but tests path handling.
        logger.info(f"Created dummy (empty) audio file: {dummy_audio_path} for path testing.")
        # For a proper test, you'd need a small, actual audio file.
        # For now, we expect Whisper to fail gracefully or for the test to be limited.


    # Create dummy video files (can be empty text files with video extensions for testing metadata extraction)
    dummy_video_files = ["nature_walk_01.mp4", "city_drive.txt", "old_film.avi"]
    for fname in dummy_video_files:
        with open(os.path.join(dummy_config_for_analyzer["paths"]["input_video_clips_dir"], fname), "w") as f:
            f.write(f"This is a dummy video file: {fname}")

    logger.info("--- Testing AudioAnalyzer ---")
    if whisper:
        audio_analyzer = AudioAnalyzer(dummy_config_for_analyzer)
        if audio_analyzer.whisper_model: # Check if model loaded
            # Check if the dummy audio file exists, even if it's not valid audio
            if os.path.exists(dummy_audio_path):
                logger.info(f"Attempting to analyze dummy audio file: {dummy_audio_path}")
                transcription = audio_analyzer.analyze()
                if transcription:
                    logger.info("Audio analysis produced a result (may be empty or error for dummy file).")
                    logger.info(f"Transcription Text (first 50 chars): {transcription.get('text', '')[:50]}")
                    # Save and load test
                    output_audio_json = os.path.join(dummy_config_for_analyzer["paths"]["output_processed_data_dir"],
                                                     dummy_config_for_analyzer["paths"]["audio_transcription_json"])
                    if audio_analyzer.save_transcription(transcription, output_audio_json):
                        logger.info(f"Audio transcription saved to {output_audio_json}")
                        loaded_transcription = audio_analyzer.load_transcription(output_audio_json)
                        if loaded_transcription and loaded_transcription.get("text") == transcription.get("text"):
                            logger.info("Audio transcription successfully loaded and verified.")
                        else:
                            logger.error("Failed to load or verify audio transcription.")
                    else:
                        logger.error("Failed to save audio transcription.")
                else:
                    logger.warning(f"Audio analysis failed for {dummy_audio_path} (as expected for a non-audio file or if Whisper setup issue).")
            else:
                logger.warning(f"Dummy audio file {dummy_audio_path} not found. Skipping audio analysis test.")
        else:
            logger.warning("AudioAnalyzer's Whisper model not loaded. Skipping audio analysis test.")
    else:
        logger.warning("Whisper library not available. Skipping AudioAnalyzer tests.")


    logger.info("\n--- Testing VideoAnalyzer ---")
    if SentenceTransformer:
        video_analyzer = VideoAnalyzer(dummy_config_for_analyzer)
        if video_analyzer.st_model: # Check if model loaded
            video_metadata = video_analyzer.analyze_videos()
            if video_metadata:
                logger.info(f"Video analysis successful. Found {len(video_metadata)} 'video' files.")
                for meta in video_metadata:
                    logger.debug(f"  Video: {meta['filename']}, Desc: '{meta['description_for_embedding']}', Embedding shape: {np.array(meta['embedding']).shape}")

                # Save and load test
                output_video_json = os.path.join(dummy_config_for_analyzer["paths"]["output_processed_data_dir"],
                                                 dummy_config_for_analyzer["paths"]["video_metadata_embeddings_json"])
                if video_analyzer.save_video_metadata(video_metadata, output_video_json):
                    logger.info(f"Video metadata saved to {output_video_json}")
                    loaded_metadata = video_analyzer.load_video_metadata(output_video_json)
                    if loaded_metadata and len(loaded_metadata) == len(video_metadata):
                        logger.info("Video metadata successfully loaded and count matches.")
                        # Further verification could compare content
                    else:
                        logger.error("Failed to load or verify video metadata.")
                else:
                    logger.error("Failed to save video metadata.")
            else:
                logger.warning("Video analysis did not return any metadata.")
        else:
            logger.warning("VideoAnalyzer's SentenceTransformer model not loaded. Skipping video analysis test.")

    else:
        logger.warning("SentenceTransformers library not available. Skipping VideoAnalyzer tests.")

    # Basic cleanup (optional, could be done manually or in a test runner)
    # print("\nConsider manually cleaning up the 'temp_test_data' directory.")