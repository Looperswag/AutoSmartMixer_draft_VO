import argparse
import logging
import os
import time
import shutil
import re # 用于用户输入清理

# 假设您的项目结构是 .../AISmartMixer/src/main.py
# 那么 utils 等模块的导入应该是 from .utils...
from .utils.config_loader import load_config
from .utils.logger_setup import setup_logging
from .phase_1_analyzer import AudioAnalyzer, VideoAnalyzer
from .phase_2_matcher import Matcher
from .phase_3_timeline_generator import TimelineGenerator
from .phase_4_clip_exporter import ClipExporter

# sentence_transformers 应该在需要它的模块（如VideoAnalyzer, Matcher）中导入
# 但如果 main.py 直接实例化并传递模型，则在这里导入是合适的
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("IMPORTANT: SentenceTransformers library not found. Video analysis and matching will not work.")
    print("Please install it: pip install sentence-transformers")


# --- Constants ---
DEFAULT_CONFIG_PATH = "config.yaml" # 如果 main.py 在 src 外层，路径可能是 "src/config.yaml" 或只 "config.yaml"
                                    # 假设 main.py 和 config.yaml 在同一级或通过 PYTHONPATH 能找到
USER_BASE_DIR = "user"

def sanitize_username(name):
    """Cleans a username to be filesystem-friendly."""
    name = name.lower()
    name = re.sub(r'\s+', '_', name)  # Replace spaces with underscores
    name = re.sub(r'[^\w_.-]', '', name)  # Remove non-alphanumeric (except _, ., -)
    return name if name else "default_user"


def clear_output_folders(config, logger, current_user_output_dir_abs):
    """
    清空指定用户output文件夹中的相关子文件夹。
    """
    # 这些是 config.yaml 中定义的相对于 user/output/ 的文件夹名
    relative_folders_to_clear = [
        os.path.basename(config["paths"].get("output_processed_data_dir", "processed_data").strip('/\\')),
        os.path.basename(config["paths"].get("output_edl_dir", "edl").strip('/\\')),
        os.path.basename(config["paths"].get("output_clips_dir", "exported_clips").strip('/\\')),
        # output_final_video_dir 也可以考虑，但通常里面是最终视频，可能不希望每次清空
        # os.path.basename(config["paths"].get("output_final_video_dir", "final_video").strip('/\\'))
    ]

    for rel_folder in relative_folders_to_clear:
        folder_to_clear_abs = os.path.join(current_user_output_dir_abs, rel_folder)
        if os.path.exists(folder_to_clear_abs):
            try:
                logger.info(f"正在清空用户文件夹: {folder_to_clear_abs}")
                for filename in os.listdir(folder_to_clear_abs):
                    file_path = os.path.join(folder_to_clear_abs, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        logger.warning(f"删除 {file_path} 时出错: {e}")
                logger.info(f"文件夹 {folder_to_clear_abs} 已清空")
            except Exception as e:
                logger.error(f"清空文件夹 {folder_to_clear_abs} 时出错: {e}")
        # else:
            # logger.debug(f"用户输出子文件夹不存在，无需清空: {folder_to_clear_abs}")


def get_audio_files(audio_dir):
    """获取目录中所有的MP3文件"""
    mp3_files = []
    if os.path.exists(audio_dir):
        for file in os.listdir(audio_dir):
            if file.lower().endswith('.mp3'):
                mp3_files.append(file)
    return sorted(mp3_files)  # 返回排序后的文件列表


def select_audio_file(config, current_user_data_dir):
    """让用户选择一个音频文件，并更新配置"""
    input_audio_dir = os.path.join(current_user_data_dir, "input_audio")
    mp3_files = get_audio_files(input_audio_dir)
    
    if not mp3_files:
        print(f"警告: 在 {input_audio_dir} 中未找到MP3文件。")
        return None
    
    print("\n可用的音频文件:")
    for i, file in enumerate(mp3_files, 1):
        print(f"{i}. {file}")
    
    selection = None
    while selection is None:
        try:
            choice = input("\n请输入要使用的音频文件编号 (1-{}): ".format(len(mp3_files)))
            index = int(choice) - 1
            if 0 <= index < len(mp3_files):
                selection = mp3_files[index]
            else:
                print(f"请输入一个有效的数字 (1-{len(mp3_files)})")
        except ValueError:
            print("请输入一个有效的数字")
    
    # 更新配置
    config["paths"]["input_audio_filename"] = selection
    print(f"已选择: {selection}")
    
    return selection


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="AISmartMixer: AI-Powered Video Clip Assembler")
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH,
        help=f"Path to the configuration file (default: {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument(
        "--force-reanalyze", action="store_true",
        help="Force re-analysis of audio and video even if intermediate files exist."
    )
    parser.add_argument(
        "--skip-clear", action="store_true",
        help="Skip clearing user's output folders before processing."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode with more verbose logging."
    )
    parser.add_argument(
        "--username", type=str, default=None,
        help="Specify username to bypass prompt (e.g., for automated scripts)."
    )
    args = parser.parse_args()

    # --- Get Username ---
    if args.username:
        username_raw = args.username
    else:
        username_raw = input("请输入您的英文名 (例如: mike): ")
    
    sanitized_username = sanitize_username(username_raw)
    if not sanitized_username:
        print("无效的用户名。退出。")
        return

    current_user_dir = os.path.abspath(os.path.join(USER_BASE_DIR, sanitized_username))
    current_user_data_dir = os.path.join(current_user_dir, "data")
    current_user_output_dir = os.path.join(current_user_dir, "output")

    # --- Create User Directories ---
    try:
        os.makedirs(current_user_data_dir, exist_ok=True)
        os.makedirs(current_user_output_dir, exist_ok=True)
        print(f"工作目录设置为: {current_user_dir}")
    except OSError as e:
        print(f"创建用户目录 {current_user_dir} 失败: {e}")
        return

    # --- Create Input Directories ---
    try:
        # Create input_audio and input_video_clips directories
        input_audio_dir = os.path.join(current_user_data_dir, "input_audio")
        input_video_dir = os.path.join(current_user_data_dir, "input_video_clips")
        os.makedirs(input_audio_dir, exist_ok=True)
        os.makedirs(input_video_dir, exist_ok=True)
        print(f"创建用户输入目录: {input_audio_dir} 和 {input_video_dir}")
    except OSError as e:
        print(f"创建用户输入目录失败: {e}")
        return

    # --- Configuration Loading ---
    try:
        config = load_config(args.config)
        if config is None: return
    except Exception as e:
        print(f"加载配置文件 {args.config} 出错: {e}")
        return

    # --- 检查用户是否存在并选择音频文件 ---
    user_exists = os.path.exists(current_user_dir)
    if user_exists:
        print(f"检测到已存在的用户目录: {current_user_dir}")
        selected_audio_file = select_audio_file(config, current_user_data_dir)
        if selected_audio_file:
            print(f"将使用选定的音频文件: {selected_audio_file}")

    # --- Override Config Paths for User ---
    original_paths_config = config.get("paths", {}).copy() # Keep a copy of original relative paths
    user_specific_paths = {}

    # Input paths
    user_specific_paths["input_audio_dir"] = os.path.join(current_user_data_dir, original_paths_config.get("input_audio_dir", "input_audio/").strip('/\\'))
    user_specific_paths["input_audio_filename"] = original_paths_config.get("input_audio_filename", "audio.mp3") # This is relative to input_audio_dir
    user_specific_paths["input_video_dir"] = os.path.join(current_user_data_dir, original_paths_config.get("input_video_dir", "input_video_clips/").strip('/\\'))

    # Output paths
    user_specific_paths["output_processed_data_dir"] = os.path.join(current_user_output_dir, original_paths_config.get("output_processed_data_dir", "processed_data/").strip('/\\'))
    user_specific_paths["output_edl_dir"] = os.path.join(current_user_output_dir, original_paths_config.get("output_edl_dir", "edl/").strip('/\\'))
    user_specific_paths["output_final_video_dir"] = os.path.join(current_user_output_dir, original_paths_config.get("output_final_video_dir", "final_video/").strip('/\\'))
    user_specific_paths["output_clips_dir"] = os.path.join(current_user_output_dir, original_paths_config.get("output_clips_dir", "exported_clips/").strip('/\\'))
    
    # Output filenames (these will be inside the above output directories)
    user_specific_paths["audio_transcription_json"] = original_paths_config.get("audio_transcription_json", "audio_transcription.json")
    user_specific_paths["video_metadata_embeddings_json"] = original_paths_config.get("video_metadata_embeddings_json", "video_metadata_embeddings.json")
    user_specific_paths["edit_decision_list_json"] = original_paths_config.get("edit_decision_list_json", "final_edit_list.json")

    config["paths"] = user_specific_paths # Update config with user-specific absolute paths

    # --- Logging Setup ---
    # Update log file path in config for the logger setup
    logging_config = config.get("logging", {})
    original_log_filename = logging_config.get("log_file", "app.log")
    logging_config["log_file"] = os.path.join(current_user_output_dir, os.path.basename(original_log_filename))
    config["logging"] = logging_config # Put it back into config

    if args.debug:
        config["logging"]["level"] = "DEBUG"
    
    setup_logging(config.get("logging", {})) # setup_logging will use the updated path
    logger = logging.getLogger(__name__) # Get logger after setup
    logger.info(f"AISmartMixer application started for user: {sanitized_username}")
    logger.info(f"使用配置文件: {args.config} (路径已针对用户调整)")
    
    if args.force_reanalyze:
        config["force_reanalyze"] = True # Ensure this is in config for other modules
        logger.info("强制重新分析已启用。")
    if args.debug:
        logger.info("调试模式已启用。")


    # --- 清空用户输出文件夹 ---
    if not args.skip_clear:
        logger.info(f"清空用户 {sanitized_username} 的输出文件夹...")
        clear_output_folders(config, logger, current_user_output_dir) # Pass absolute user output dir
    else:
        logger.info(f"跳过清空用户 {sanitized_username} 的输出文件夹（因为 --skip-clear 标志）。")

    # --- Ensure User-Specific Output Directories Exist ---
    try:
        # Base output subdirs are already created by clear_output_folders or will be by file_handler
        # We need to ensure the specific ones from config exist
        for key, path_val in config["paths"].items():
            if key.startswith("output_") and key.endswith("_dir"):
                 os.makedirs(path_val, exist_ok=True)
        logger.info("用户特定的输出目录已确保存在。")
    except OSError as e:
        logger.error(f"创建用户输出目录时出错: {e}")
        return

    # --- 验证用户输入路径 ---
    # Input audio directory and file
    abs_input_audio_dir = config["paths"]["input_audio_dir"]
    abs_input_audio_path = os.path.join(abs_input_audio_dir, config["paths"]["input_audio_filename"])
    
    if not os.path.exists(abs_input_audio_dir):
        logger.error(f"用户音频输入目录不存在: {abs_input_audio_dir}")
        logger.info(f"请在 {abs_input_audio_dir} 中创建并放入您的音频文件。")
        return
    if not os.path.exists(abs_input_audio_path):
        logger.error(f"用户音频文件不存在: {abs_input_audio_path}")
        logger.info(f"请确保音频文件 '{config['paths']['input_audio_filename']}' 存在于 {abs_input_audio_dir} 中。")
        return
    logger.info(f"用户音频文件已验证: {abs_input_audio_path}")

    # Input video directory
    abs_input_video_dir = config["paths"]["input_video_dir"]
    if not os.path.exists(abs_input_video_dir):
        logger.error(f"用户视频输入目录不存在: {abs_input_video_dir}")
        logger.info(f"请在 {abs_input_video_dir} 中创建并放入您的视频切片。")
        return
    logger.info(f"用户视频目录已验证: {abs_input_video_dir}")


    # --- PHASE 1: ANALYSIS ---
    logger.info("启动阶段 1: 分析")
    audio_analyzer = AudioAnalyzer(config) # Pass the user-specific config
    
    # Path for transcription JSON (now user-specific)
    audio_transcription_path = os.path.join(
        config["paths"]["output_processed_data_dir"], # This is now user/mike/output/processed_data/
        config["paths"]["audio_transcription_json"]
    )

    if not os.path.exists(audio_transcription_path) or config.get("force_reanalyze"):
        logger.info("正在分析音频...")
        transcription_result = audio_analyzer.analyze() # analyze() uses paths from config
        if transcription_result:
            audio_analyzer.save_transcription(transcription_result, audio_transcription_path)
            logger.info(f"音频转录已保存到 {audio_transcription_path}")
        else:
            logger.error("音频分析失败。正在退出。")
            return
    else:
        logger.info(f"使用已存在的音频转录: {audio_transcription_path}")
        transcription_result = audio_analyzer.load_transcription(audio_transcription_path)
        if not transcription_result:
            logger.error(f"从 {audio_transcription_path} 加载已存在的转录失败。请考虑重新分析。")
            return

    # Video Analysis
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("SentenceTransformers库不可用。无法进行视频分析和匹配。正在退出。")
        return
        
    embedding_model_name = config.get("models", {}).get("embedding_model_name", "all-MiniLM-L6-v2") # Default from config
    embedding_model = None
    try:
        logger.info(f"正在加载用于视频分析的句子转换器模型: {embedding_model_name}...")
        embedding_model = SentenceTransformer(embedding_model_name)
        logger.info(f"句子转换器模型 '{embedding_model_name}' 加载成功。")
    except Exception as e:
        logger.error(f"加载句子转换器模型 '{embedding_model_name}' 失败: {e}", exc_info=True)
        return

    # 检查用户视频文件夹是否包含视频文件
    video_files_in_user_dir = []
    video_extensions_cfg = config.get("settings", {}).get("video_extensions", ["*.mp4"])
    normalized_extensions = [ext[1:] if ext.startswith("*.") else ext for ext in video_extensions_cfg]
    
    for root, _, files in os.walk(config["paths"]["input_video_dir"]): # This is now user's video dir
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in normalized_extensions):
                video_files_in_user_dir.append(os.path.join(root, file))
    
    if not video_files_in_user_dir:
        logger.warning(f"警告：在用户视频目录 {config['paths']['input_video_dir']} 中未找到任何匹配的视频文件。")
        logger.info(f"检查的视频扩展名: {', '.join(normalized_extensions)}")
    else:
        logger.info(f"在用户视频目录中找到 {len(video_files_in_user_dir)} 个视频文件。")

    video_analyzer = VideoAnalyzer(config, embedding_model)
    video_metadata_path = os.path.join(
        config["paths"]["output_processed_data_dir"],
        config["paths"]["video_metadata_embeddings_json"]
    )
    if not os.path.exists(video_metadata_path) or config.get("force_reanalyze"):
        logger.info("正在分析视频片段...")
        video_metadata_list = video_analyzer.analyze_videos()
        if video_metadata_list: # analyze_videos can return empty list
            video_analyzer.save_video_metadata(video_metadata_list, video_metadata_path)
            logger.info(f"视频元数据和嵌入向量已保存到 {video_metadata_path}")
        else: # Handles case where analyze_videos returns None or empty list
            logger.warning("没有处理任何视频片段或视频分析失败。")
            video_metadata_list = [] 
    else:
        logger.info(f"使用已存在的视频元数据: {video_metadata_path}")
        video_metadata_list = video_analyzer.load_video_metadata(video_metadata_path)
        if not video_metadata_list: # Handles case where file exists but is empty/invalid
            logger.warning(f"从 {video_metadata_path} 加载已存在的视频元数据失败或文件为空。")
            video_metadata_list = []


    if not transcription_result or not transcription_result.get("segments"):
        logger.error("分析后未找到音频片段。无法继续。")
        return
    # No need to check video_metadata_list for emptiness here, subsequent phases handle it

    logger.info("阶段 1: 分析完成。")

    # --- PHASE 2: MATCHER ---
    logger.info("启动阶段 2: 匹配器初始化")
    # 确保这些设置在config中，以便Matcher和TimelineGenerator使用
    if "settings" not in config: config["settings"] = {}
    config["settings"]["similarity_threshold"] = config["settings"].get("similarity_threshold", -1.0)
    config["settings"]["top_k_candidates"] = config["settings"].get("top_k_candidates", 5)
    
    matcher = Matcher(config)
    logger.info("阶段 2: 匹配器已使用确保每个片段有前5个匹配的设置进行初始化。")

    # --- PHASE 3: TIMELINE GENERATOR ---
    logger.info("启动阶段 3: 时间线生成")
    if "clip_export" not in config: config["clip_export"] = {}
    config["clip_export"]["max_clips_per_segment"] = config["clip_export"].get("max_clips_per_segment", 5)
    
    timeline_generator = TimelineGenerator(config, matcher)
    
    edit_decision_list_path = os.path.join(
        config["paths"]["output_edl_dir"], # User-specific EDL dir
        config["paths"]["edit_decision_list_json"]
    )

    edit_decision_list = timeline_generator.generate_timeline(
        transcription_result,
        video_metadata_list if video_metadata_list is not None else []
    )

    if edit_decision_list:
        # 文件名修正逻辑应该不需要了，因为 VideoAnalyzer 现在存储的是 filename
        # for event in edit_decision_list:
        #     if event.get("event_type") == "video_match" and "video_segments" in event:
        #         for segment in event["video_segments"]:
        #             if "filename" in segment:
        #                 segment["filename"] = os.path.basename(segment["filename"])
        
        if timeline_generator.save_timeline(edit_decision_list, edit_decision_list_path):
            logger.info(f"时间线 (编辑决策列表) 已保存到 {edit_decision_list_path}")
        else:
            logger.error(f"保存时间线 (编辑决策列表) 到 {edit_decision_list_path} 失败。正在退出。")
            return
    else:
        logger.error("时间线生成失败或产生空的EDL。正在退出。")
        return
    logger.info("阶段 3: 时间线生成完成。")

    # --- PHASE 4: CLIP EXPORTER ---
    logger.info("启动阶段 4: 片段导出器")
    # 验证用户输入视频文件夹是否存在和包含文件 (已在上面验证过)
    # video_path_map 的构建需要基于用户视频目录
    user_video_path_map = {}
    for video_file_abs_path in video_files_in_user_dir: # Use the list from earlier scan
        video_filename = os.path.basename(video_file_abs_path)
        user_video_path_map[video_filename] = video_file_abs_path
    
    config["clip_export"]["video_path_map"] = user_video_path_map
    config["clip_export"]["include_score_in_filename"] = config["clip_export"].get("include_score_in_filename", True)
    
    clip_exporter = ClipExporter(config) # config now has user-specific paths
    success = clip_exporter.export_top_clips(edit_decision_list)

    if success:
        logger.info(f"最匹配的片段已成功导出到 {config['paths']['output_clips_dir']}")
    else:
        logger.error("片段导出失败。")
    logger.info("阶段 4: 片段导出器完成。")

    end_time = time.time()
    logger.info(f"AISmartMixer 用户 {sanitized_username} 的处理在 {end_time - start_time:.2f} 秒内完成。")

if __name__ == "__main__":
    if not SENTENCE_TRANSFORMERS_AVAILABLE and "VideoAnalyzer" not in globals(): # Simple check
        # This check is a bit weak, but tries to prevent running if core deps are missing
        print("关键库 (如 SentenceTransformers) 未加载，某些功能可能无法运行。")
    main()