import logging
import os
import shutil
import re
import unicodedata
import json
import time
import numpy as np
import uuid
import hashlib

try:
    from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    print("--------------------------------------------------------------------")
    print("IMPORTANT: MoviePy library not found or an issue occurred importing it.")
    print("Please install it: pip install moviepy")
    print("Needed for getting video durations and creating subclips.")
    print("--------------------------------------------------------------------")
    MOVIEPY_AVAILABLE = False
    VideoFileClip = None
    CompositeVideoClip = None
    ColorClip = None

logger = logging.getLogger(__name__)

class ClipExporter:
    """
    Exports top matching video clips for each audio segment to separate folders.
    Replaces the synthesizer to allow manual review of matched clips.
    """
    def __init__(self, config):
        """
        Initializes the ClipExporter.

        Args:
            config (dict): The application configuration dictionary.
        """
        self.config = config
        self.video_clips_base_dir = self.config.get("paths", {}).get("input_video_dir")
        self.output_clips_base_dir = self.config.get("paths", {}).get("output_clips_dir", 
                                                    os.path.join(self.config.get("paths", {}).get("output_final_video_dir", "output/clips")))
        
        # Get clip export specific settings
        clip_export_config = self.config.get("clip_export", {})
        self.max_clips_per_segment = clip_export_config.get("max_clips_per_segment", 5)
        self.include_score_in_filename = clip_export_config.get("include_score_in_filename", True)
        self.max_folder_name_length = clip_export_config.get("max_folder_name_length", 50)
        # New setting to limit filename length
        self.max_filename_length = clip_export_config.get("max_filename_length", 100)
        self.extract_subclips = clip_export_config.get("extract_subclips", True)
        self.padding_seconds = clip_export_config.get("padding_seconds", 2.0)
        self.create_preview_metadata = clip_export_config.get("create_preview_metadata", True)
        self.normalize_audio = clip_export_config.get("normalize_audio", False)
        
        # 获取预先提供的视频路径映射
        self.video_path_map = clip_export_config.get("video_path_map", {})
        if self.video_path_map:
            logger.info(f"已加载 {len(self.video_path_map)} 个视频文件的路径映射")
        
        # Ensure the output directory exists
        if not os.path.exists(self.output_clips_base_dir):
            try:
                os.makedirs(self.output_clips_base_dir, exist_ok=True)
                logger.info(f"Created output clips directory: {self.output_clips_base_dir}")
            except Exception as e:
                logger.error(f"Failed to create output directory {self.output_clips_base_dir}: {e}")
        
        # Initialize folder mapping cache for video lookup
        self._folder_mapping_cache = None
        
        # Keep track of all exported clips for generating preview metadata
        self.exported_clips_metadata = {}

    def _sanitize_folder_name(self, text, max_length=None):
        """
        Creates a safe folder name from the provided text.
        
        Args:
            text (str): The folder name to sanitize
            max_length (int, optional): Maximum length of the folder name
            
        Returns:
            str: A sanitized folder name safe for filesystem use
        """
        if max_length is None:
            max_length = self.max_folder_name_length
        
        # Make sure we're working with a string
        if text is None:
            text = "unnamed_segment"
        
        # Replace apostrophes and other common problematic characters
        text = text.replace("'", "").replace('"', "").replace('`', "")
        
        # Remove invalid characters for file systems
        text = re.sub(r'[\\/*?:"<>|]', '', text)
        
        # Replace spaces and other separators with underscores
        text = re.sub(r'[\s\.,;:!@#$%^&()]', '_', text)
        
        # Remove consecutive underscores
        text = re.sub(r'_{2,}', '_', text)
        
        # Normalize unicode characters and remove non-ASCII
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        
        # Limit length but try to keep whole words
        if len(text) > max_length:
            # Try to cut at the last underscore before max_length
            last_underscore = text[:max_length].rfind('_')
            if last_underscore > max_length // 2:  # Only use underscore if reasonably positioned
                text = text[:last_underscore]
            else:
                text = text[:max_length]
        
        # Remove leading/trailing underscores
        text = text.strip('_')
        
        # Ensure the folder name is not empty or just underscores
        if not text or text.isspace():
            text = "unnamed_segment"
        
        return text

    def _truncate_filename(self, filename, max_length=None):
        """
        Truncates a filename to a reasonable length while preserving extension.
        
        Args:
            filename (str): Original filename
            max_length (int, optional): Maximum length for the filename
            
        Returns:
            str: Truncated filename
        """
        if max_length is None:
            max_length = self.max_filename_length
            
        if not filename:
            return "unknown_file"
            
        # Get file extension
        base, ext = os.path.splitext(filename)
        
        # Calculate max length for base part (reserving space for extension)
        max_base_length = max_length - len(ext)
        
        if len(base) <= max_base_length:
            return filename
            
        # If base is too long, truncate it
        truncated_base = base[:max_base_length - 4]  # Leave room for hash
        
        # Add a short hash to avoid naming conflicts
        name_hash = hashlib.md5(base.encode()).hexdigest()[:4]
        
        return f"{truncated_base}_{name_hash}{ext}"

    def _find_video_path(self, video_filename):
        """
        Finds the full path to a video file based on its filename.
        
        Args:
            video_filename (str): The filename of the video
            
        Returns:
            str: The full path to the video file, or None if not found
        """
        if not video_filename:
            logger.warning("Empty video filename provided")
            return None
            
        if not self.video_clips_base_dir:
            logger.warning("Video clips base directory not configured")
            return None
        
        # 首先检查预先提供的路径映射
        basename = os.path.basename(video_filename)
        if basename in self.video_path_map:
            path = self.video_path_map[basename]
            logger.debug(f"Found video in path map: {basename} -> {path}")
            return path
        
        # 如果没找到，检查原始文件名
        if video_filename in self.video_path_map:
            path = self.video_path_map[video_filename]
            logger.debug(f"Found video in path map with original name: {video_filename} -> {path}")
            return path
            
        logger.debug(f"Looking for video file: {video_filename}")
        logger.debug(f"Base directory: {self.video_clips_base_dir}")
        
        # 尝试其他方式查找
        # 首先尝试直接路径
        direct_path = os.path.join(self.video_clips_base_dir, video_filename)
        if os.path.exists(direct_path):
            logger.debug(f"Found video at direct path: {direct_path}")
            return direct_path
            
        direct_path_basename = os.path.join(self.video_clips_base_dir, basename)
        if os.path.exists(direct_path_basename):
            logger.debug(f"Found video using basename at: {direct_path_basename}")
            return direct_path_basename
        
        # 初始化文件夹映射缓存
        if self._folder_mapping_cache is None:
            self._folder_mapping_cache = {}
            try:
                # 扫描所有目录和子目录
                for root, dirs, files in os.walk(self.video_clips_base_dir):
                    for dirname in dirs:
                        folder_path = os.path.join(root, dirname)
                        if os.path.isdir(folder_path):
                            self._folder_mapping_cache[dirname] = folder_path
                            logger.debug(f"Added folder to cache: {dirname} -> {folder_path}")
            except Exception as e:
                logger.error(f"Error scanning video directory structure: {e}")
        
        # 在映射的子目录中寻找视频
        for folder_name, folder_path in self._folder_mapping_cache.items():
            # 检查完整文件名
            potential_path = os.path.join(folder_path, video_filename)
            if os.path.exists(potential_path):
                logger.debug(f"Found video in mapped folder: {potential_path}")
                return potential_path
                
            # 检查基础文件名
            potential_basename_path = os.path.join(folder_path, basename)
            if os.path.exists(potential_basename_path):
                logger.debug(f"Found video using basename in mapped folder: {potential_basename_path}")
                return potential_basename_path
        
        # 尝试通过搜索所有子目录中的所有文件来查找视频
        for root, dirs, files in os.walk(self.video_clips_base_dir):
            # 检查完整文件名
            if video_filename in files:
                full_path = os.path.join(root, video_filename)
                logger.debug(f"Found video by full directory scan: {full_path}")
                return full_path
                
            # 检查基础文件名
            if basename in files:
                full_path = os.path.join(root, basename)
                logger.debug(f"Found video by basename in full scan: {full_path}")
                return full_path
        
        # 作为最后的手段，检查基本目录中的任何文件是否包含
        # 视频文件名的主要部分
        filename_stem = os.path.splitext(basename)[0]
        if len(filename_stem) > 10:  # 只有当词干合理长，以避免太多误匹配
            for root, dirs, files in os.walk(self.video_clips_base_dir):
                for file in files:
                    # 过滤只留下视频文件
                    if not file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')):
                        continue
                        
                    # 检查部分匹配
                    if filename_stem[:10] in file:  # 匹配前10个字符
                        full_path = os.path.join(root, file)
                        logger.debug(f"Found video by partial name match: {full_path} (matched with {filename_stem[:10]})")
                        return full_path
        
        logger.warning(f"无法找到视频文件: {video_filename}")
        return None

    def _extract_subclip(self, source_path, output_path, start_time=0, duration=None, target_duration=None):
        """
        Extract a portion of a video file as a new clip.
        
        Args:
            source_path (str): Path to the source video file
            output_path (str): Path where the subclip will be saved
            start_time (float): Start time in seconds for the subclip
            duration (float, optional): Duration of the subclip in seconds
            target_duration (float, optional): Target duration for the audio segment
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not MOVIEPY_AVAILABLE:
            return False
            
        try:
            logger.debug(f"Loading video: {source_path}")
            video_clip = VideoFileClip(source_path)
            
            # Determine clip duration
            clip_duration = duration
            if clip_duration is None:
                if target_duration:
                    # Use the full video or the target duration, whichever is shorter
                    clip_duration = min(video_clip.duration - start_time, target_duration)
                else:
                    # Use the full video from start_time
                    clip_duration = video_clip.duration - start_time
            
            # Ensure we don't try to extract beyond the video's duration
            if start_time >= video_clip.duration:
                logger.warning(f"Start time {start_time}s is beyond video duration {video_clip.duration}s")
                start_time = max(0, video_clip.duration - clip_duration)
                
            end_time = min(start_time + clip_duration, video_clip.duration)
            actual_duration = end_time - start_time
            
            logger.info(f"Extracting subclip from {source_path}: {start_time:.2f}s to {end_time:.2f}s (duration: {actual_duration:.2f}s)")
            
            # Extract the subclip
            subclip = video_clip.subclip(start_time, end_time)
            
            # Generate a unique ID for temp files to avoid long filenames
            temp_id = str(uuid.uuid4())[:8]
            temp_audiofile = f"temp_{temp_id}.m4a"
            
            # If the actual duration is less than target, create a black background to fill
            if target_duration and actual_duration < target_duration:
                logger.info(f"Padding clip to match target duration {target_duration:.2f}s")
                background = ColorClip(
                    size=video_clip.size,
                    color=(0, 0, 0),
                    duration=target_duration
                )
                # Center the actual clip on the background
                composite = CompositeVideoClip(
                    [background, subclip.set_start(0)]
                )
                composite.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile=temp_audiofile,
                    remove_temp=True,
                    threads=4,
                    preset="medium",
                    logger=None  # Suppress progress bar
                )
                composite.close()
            else:
                # Save the subclip directly
                subclip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile=temp_audiofile,
                    remove_temp=True,
                    threads=4,
                    preset="medium",
                    logger=None  # Suppress progress bar
                )
            
            # Clean up
            subclip.close()
            video_clip.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error extracting subclip: {e}")
            return False

    def _get_segment_time_range(self, event):
        """
        Get the time range for an audio segment including padding.
        
        Args:
            event (dict): The edit decision list event
            
        Returns:
            tuple: (start_time, end_time, duration) in seconds
        """
        audio_start_time = event.get("audio_start_time", 0)
        audio_end_time = event.get("audio_end_time", 0)
        duration = audio_end_time - audio_start_time
        
        return audio_start_time, audio_end_time, duration

    def _save_preview_metadata(self, metadata_dict):
        """
        Save preview metadata to a JSON file for the HTML preview generator.
        
        Args:
            metadata_dict (dict): Dictionary containing the exported clips metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            metadata_path = os.path.join(self.output_clips_base_dir, "preview_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Preview metadata saved to {metadata_path}")
            
            # Also create a simple HTML index file
            html_path = os.path.join(self.output_clips_base_dir, "index.html")
            html_content = self._generate_html_index(metadata_dict)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"HTML preview index saved to {html_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving preview metadata: {e}")
            return False

    def _generate_html_index(self, metadata_dict):
        """
        Generate a simple HTML index file for browsing exported clips.
        
        Args:
            metadata_dict (dict): Dictionary containing the exported clips metadata
            
        Returns:
            str: HTML content
        """
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exported Clips Preview</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
        h1 { text-align: center; color: #333; }
        h2 { color: #555; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
        .segment { margin-bottom: 40px; background: #f9f9f9; padding: 15px; border-radius: 5px; }
        .clip-list { display: flex; flex-wrap: wrap; gap: 20px; }
        .clip { width: 300px; }
        .clip video { width: 100%; border-radius: 5px; }
        .clip-info { font-size: 0.9em; margin-top: 5px; }
        .score { font-weight: bold; color: #0066cc; }
        .metadata { margin-top: 5px; font-size: 0.8em; color: #666; }
    </style>
</head>
<body>
    <h1>Exported Clips Preview</h1>
"""
        
        # Sort segments by ID
        sorted_segments = sorted(metadata_dict.items(), key=lambda x: x[0])
        
        for segment_id, segment_data in sorted_segments:
            audio_text = segment_data.get("audio_text", "No text available")
            clips = segment_data.get("clips", [])
            
            html += f"""
    <div class="segment">
        <h2>Segment: {segment_id}</h2>
        <p><strong>Audio Text:</strong> {audio_text}</p>
        <div class="clip-list">
"""
            
            for clip in clips:
                rel_path = os.path.relpath(clip.get("output_path", ""), self.output_clips_base_dir)
                score = clip.get("similarity_score", 0)
                filename = os.path.basename(clip.get("source_path", ""))
                
                html += f"""
            <div class="clip">
                <video controls>
                    <source src="{rel_path}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <div class="clip-info">
                    <div>Score: <span class="score">{score:.4f}</span></div>
                    <div class="metadata">Source: {filename}</div>
                </div>
            </div>
"""
            
            html += """
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html

    def export_top_clips(self, edl):
        """
        Exports top matching video clips for each audio segment to separate folders.
        
        Args:
            edl (list): The edit decision list with matching information
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        if not MOVIEPY_AVAILABLE:
            logger.error("MoviePy library is not available. Cannot export video clips.")
            return False
            
        if not self.video_clips_base_dir:
            logger.error("Input video directory not configured. Cannot locate original videos.")
            return False
            
        if not edl:
            logger.error("Edit decision list (EDL) is empty. No matches to export.")
            return False
            
        logger.info(f"Starting clip export to: {self.output_clips_base_dir}")
        logger.info(f"Video clips base directory: {self.video_clips_base_dir}")
        
        # Print directory structure for debugging
        try:
            logger.info("Scanning video input directory structure:")
            for root, dirs, files in os.walk(self.video_clips_base_dir):
                rel_path = os.path.relpath(root, self.video_clips_base_dir)
                if rel_path == '.':
                    logger.info(f"- Root directory: {len(files)} files, {len(dirs)} subdirectories")
                    if len(files) < 10:  # If not too many files, log them
                        logger.info(f"  Files in root: {', '.join(files)}")
                else:
                    level = rel_path.count(os.sep)
                    indent = '  ' * (level + 1)
                    logger.info(f"{indent}Subdir: {os.path.basename(root)} - {len(files)} files")
                    if level < 2 and len(files) < 5:  # Only log files for first two levels
                        logger.info(f"{indent}  Files: {', '.join(files)}")
        except Exception as e:
            logger.warning(f"Error scanning directory structure: {e}")
        
        # Reset export metadata
        self.exported_clips_metadata = {}
        
        # Track export statistics
        total_segments = 0
        total_exported_clips = 0
        failed_exports = 0
        
        # Process each audio segment in the EDL
        for event_idx, event in enumerate(edl):
            event_type = event.get("event_type", "unknown")
            audio_segment_id = event.get("audio_segment_id", f"segment_{event_idx}")
            audio_text = event.get("audio_text", "")
            
            logger.info(f"Processing EDL event {event_idx} (id: {audio_segment_id})")
            logger.debug(f"Event details: type={event_type}, text='{audio_text[:50]}'")
            
            # Skip events that aren't video matches or don't have proper segments
            if event_type != "video_match" or not event.get("video_segments"):
                logger.info(f"Skipping event {event_idx} (type: {event_type}) - not a video match or has no segments")
                continue
                
            total_segments += 1
            
            # Create folder name based on audio segment text and ID
            folder_prefix = f"{audio_segment_id}_"
            folder_text = self._sanitize_folder_name(audio_text)
            segment_folder_name = folder_prefix + folder_text
            segment_folder_path = os.path.join(self.output_clips_base_dir, segment_folder_name)
            
            # Add to metadata
            self.exported_clips_metadata[audio_segment_id] = {
                "audio_text": audio_text,
                "folder_path": segment_folder_path,
                "audio_start_time": event.get("audio_start_time", 0),
                "audio_end_time": event.get("audio_end_time", 0),
                "clips": []
            }
            
            # Ensure segment folder exists
            try:
                os.makedirs(segment_folder_path, exist_ok=True)
                logger.info(f"Created/verified folder for segment {audio_segment_id}: {segment_folder_path}")
            except Exception as e:
                logger.error(f"Failed to create folder for segment {audio_segment_id}: {e}")
                failed_exports += 1
                continue
                
            # Get the video segments (should already be sorted by score in descending order)
            video_segments = event.get("video_segments", [])
            
            # Get audio segment time range
            audio_start, audio_end, audio_duration = self._get_segment_time_range(event)
            
            # Limit to top N matches
            top_segments = video_segments[:self.max_clips_per_segment]
            logger.info(f"Exporting top {len(top_segments)} matches for segment {audio_segment_id}")
            
            # Create a metadata file for this segment
            segment_metadata = {
                "audio_segment_id": audio_segment_id,
                "audio_text": audio_text,
                "audio_start_time": audio_start,
                "audio_end_time": audio_end,
                "audio_duration": audio_duration,
                "clips": []
            }
            
            # Export each of the top matches
            for match_idx, video_segment in enumerate(top_segments):
                video_id = video_segment.get("video_id")
                filename = video_segment.get("filename")
                similarity_score = video_segment.get("similarity_score", 0)
                start_time = video_segment.get("start_time", 0)
                segment_duration = video_segment.get("duration", None)
                
                logger.info(f"  Processing match {match_idx+1}: video={filename}, score={similarity_score:.4f}")
                
                # Find the original video file
                source_path = self._find_video_path(filename)
                if not source_path:
                    logger.warning(f"Could not find source video {filename} for segment {audio_segment_id}")
                    # Check if any video file exists at all
                    logger.info(f"Checking if any video files exist in {self.video_clips_base_dir}")
                    video_files = []
                    for root, _, files in os.walk(self.video_clips_base_dir):
                        for file in files:
                            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                                video_files.append(os.path.join(root, file))
                                if len(video_files) >= 5:  # Limit to 5 examples
                                    break
                    if video_files:
                        logger.info(f"Found {len(video_files)} video files in input directory. Examples: {', '.join(os.path.basename(f) for f in video_files[:3])}")
                    else:
                        logger.warning("No video files found in input directory!")
                    
                    failed_exports += 1
                    continue
                
                # Format the output filename with rank and score
                rank_str = f"{match_idx+1:02d}"
                score_part = f"_score_{similarity_score:.4f}".replace(".", "_") if self.include_score_in_filename else ""
                
                # Use a shorter version of the original filename to avoid path length issues
                original_filename_part = self._truncate_filename(os.path.basename(filename), 80)
                output_basename = f"{rank_str}{score_part}_{original_filename_part}"
                
                # Make final check for length and truncate if necessary
                if len(output_basename) > self.max_filename_length:
                    output_basename = self._truncate_filename(output_basename, self.max_filename_length)
                
                output_path = os.path.join(segment_folder_path, output_basename)
                
                try:
                    if self.extract_subclips and segment_duration is not None:
                        # Extract a portion of the video matching the audio segment's duration
                        logger.info(f"  Extracting subclip from {os.path.basename(source_path)} (start={start_time}s, duration={segment_duration}s)")
                        success = self._extract_subclip(
                            source_path, 
                            output_path, 
                            start_time=start_time,
                            duration=segment_duration, 
                            target_duration=audio_duration
                        )
                        
                        if not success:
                            logger.warning(f"  Failed to extract subclip, falling back to copying entire file")
                            shutil.copy2(source_path, output_path)
                    else:
                        # Just copy the entire video file
                        logger.info(f"  Copying entire video file from {source_path} to {output_path}")
                        shutil.copy2(source_path, output_path)
                    
                    logger.info(f"  Exported clip {match_idx+1}/{len(top_segments)} for segment {audio_segment_id}: {os.path.basename(output_path)}")
                    total_exported_clips += 1
                    
                    # Add to segment metadata
                    clip_metadata = {
                        "rank": match_idx + 1,
                        "video_id": video_id,
                        "filename": os.path.basename(filename),
                        "similarity_score": similarity_score,
                        "source_path": source_path,
                        "output_path": output_path,
                        "start_time": start_time,
                        "duration": segment_duration
                    }
                    segment_metadata["clips"].append(clip_metadata)
                    self.exported_clips_metadata[audio_segment_id]["clips"].append(clip_metadata)
                    
                except Exception as e:
                    logger.error(f"Failed to export clip {match_idx+1} for segment {audio_segment_id}: {e}")
                    logger.debug(f"Exception details:", exc_info=True)
                    failed_exports += 1
            
            # Save segment metadata
            segment_metadata_path = os.path.join(segment_folder_path, "segment_metadata.json")
            try:
                with open(segment_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(segment_metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved segment metadata to {segment_metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to save segment metadata: {e}")
        
        # Save preview metadata if requested
        if self.create_preview_metadata:
            self._save_preview_metadata(self.exported_clips_metadata)
        
        # Log export summary
        logger.info(f"Clip export complete. Summary:")
        logger.info(f"- Total audio segments processed: {total_segments}")
        logger.info(f"- Total clips exported: {total_exported_clips}")
        logger.info(f"- Failed exports: {failed_exports}")
        
        return True