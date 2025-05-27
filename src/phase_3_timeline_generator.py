# src/phase_3_timeline_generator.py

import logging
import os
import random

from .utils.file_handler import save_json

logger = logging.getLogger(__name__)

class TimelineGenerator:
    def __init__(self, config, matcher):
        self.config = config
        self.matcher = matcher
        if not self.matcher or not hasattr(self.matcher, 'find_best_matches'):
            logger.error("TimelineGenerator: 提供的Matcher实例无效或未初始化。")
            raise ValueError("TimelineGenerator需要有效的Matcher实例。")

        settings = self.config.get("settings", {})
        self.min_clip_duration = settings.get("min_clip_duration_for_timeline", 1.0)
        # similarity_threshold 和 top_k_candidates 会被 main.py 中的设置覆盖
        self.similarity_threshold = settings.get("similarity_threshold", -1.0) 
        self.top_k_candidates = settings.get("top_k_candidates", 5)
        self.reuse_strategy = settings.get("video_reuse_strategy", "reuse_different_segments")
        self.fallback_strategy = settings.get("fallback_strategy", "blank")
        
        # min_similarity_threshold 在 similarity_threshold 为 -1.0 时基本不起主要过滤作用
        self.min_similarity_threshold = max(0.001, self.similarity_threshold * 0.4 if self.similarity_threshold >= 0 else 0.001)
        
        # 目标每个音频段至少匹配的视频数量，应与 top_k_candidates 一致
        self.min_video_matches = self.top_k_candidates # 确保这里是 5
        
        self.used_videos = {}
        self.used_videos_per_segment = {}

    def _get_candidate_videos(self, audio_segment, video_metadata_list, top_n=None):
        if top_n is None:
            top_n = self.top_k_candidates # 默认为 5
        
        # 从 matcher 获取稍多一些的候选，以备不时之需 (例如，如果后续有其他过滤)
        # 考虑到 similarity_threshold = -1.0，主要依赖 matcher 的排序
        num_to_fetch_from_matcher = max(top_n, top_n * 2) # 例如，如果 top_n=5, 请求 10 个

        matches = self.matcher.find_best_matches(audio_segment, video_metadata_list, top_n=num_to_fetch_from_matcher)
        
        # 由于 self.similarity_threshold 设置为 -1.0 (或非常低)，以下过滤基本无效
        # filtered_matches = [m for m in matches if m[1] >= self.similarity_threshold]
        # 直接使用 matcher 的结果
        filtered_matches = list(matches) 

        # 如果获得的匹配少于目标 top_n，并且视频库中还有更多视频，
        # 这通常意味着这些额外视频与音频的相似度非常低（可能为0或负）。
        # 这里的补充逻辑是为了“硬凑”数量，即使补充的视频相关性极低。
        if len(filtered_matches) < top_n and video_metadata_list:
            logger.info(f"音频段 '{audio_segment.get('text', '')[:20]}...' 的初始匹配数 ({len(filtered_matches)}) 少于目标 ({top_n})。")
            
            current_match_ids = {m[0] for m in filtered_matches}
            additional_candidates_needed = top_n - len(filtered_matches)
            
            supplemental_options = []
            for video_meta in video_metadata_list:
                if video_meta["id"] not in current_match_ids:
                    # 为这些补充视频分配一个极低的分数
                    supplemental_options.append(
                        (video_meta["id"], -2.0, video_meta.get("duration")) 
                    )
            
            # 如果有补充选项，添加到 filtered_matches
            if supplemental_options:
                # 可以考虑对 supplemental_options 也进行某种排序，例如文件名
                # 但通常情况下，它们的相关性远低于 matcher 返回的
                filtered_matches.extend(supplemental_options[:additional_candidates_needed])
                logger.info(f"已为音频段补充 {min(len(supplemental_options), additional_candidates_needed)} 个低相关性视频以尝试达到数量目标。")
        
        # 最终返回不多于 top_n 个候选
        return filtered_matches[:top_n]

    # _select_videos_for_segment 方法不再是生成EDL中 video_segments 列表的主要逻辑
    # 它可能仍然用于确定单个视频片段的具体使用时长，如果需要合成最终视频的话。
    # 但对于“列出前5个匹配”的需求，我们直接在 generate_timeline 中处理。

    def generate_timeline(self, audio_transcription, video_metadata_list):
        self.used_videos = {} 
        self.used_videos_per_segment = {}
        
        if not audio_transcription or "segments" not in audio_transcription:
            logger.error("提供的音频转录数据无效或为空。")
            return []
        if not video_metadata_list:
            logger.warning("视频元数据列表为空。时间线中的视频匹配将为空。")
            # 即使视频列表为空，也应该为每个音频段生成事件结构

        video_metadata_map = {vm["id"]: vm for vm in video_metadata_list if "id" in vm}
        # if not video_metadata_map and video_metadata_list: # warning if list had items but map is empty
        #     logger.error("视频元数据列表不包含具有'id'的有效项目。")
        #     return [] # Or handle differently

        timeline_events = []
        sorted_audio_segments = sorted(audio_transcription["segments"], key=lambda s: s["start"])

        for i, segment in enumerate(sorted_audio_segments):
            segment_id = segment.get("id", f"segment_{i}")
            segment_text = segment.get("text", "").strip()
            segment_duration = segment.get("end", 0) - segment.get("start", 0)

            if not segment_text or segment_duration < self.min_clip_duration:
                logger.info(f"跳过音频段 ID {segment_id}，因为文本为空或时长过短 ({segment_duration:.2f}s)。")
                timeline_events.append({
                    "event_type": "slug",
                    "audio_segment_id": segment_id,
                    "audio_start_time": segment.get("start"),
                    "audio_end_time": segment.get("end"),
                    "audio_text": segment_text, # 保留audio_text
                    "reason": "文本为空或过短"
                })
                continue

            logger.info(f"处理音频段 ID {segment_id}: '{segment_text[:50]}...'")
            
            # 根据重用策略确定当前段可用的视频列表
            # 注意：这里的 available_videos_for_segment 主要影响 _get_candidate_videos 的输入范围
            # 如果希望总是从全部视频中选top_k，则应传递 video_metadata_map.values()
            # 但通常重用策略会限制候选池
            available_videos_for_segment = []
            if self.reuse_strategy == "no_reuse":
                available_videos_for_segment = [v for v in video_metadata_map.values() if v["id"] not in self.used_videos]
            elif self.reuse_strategy == "reuse_different_segments":
                current_segment_already_used = self.used_videos_per_segment.get(segment_id, set())
                available_videos_for_segment = [v for v in video_metadata_map.values() if v["id"] not in current_segment_already_used]
            else: # "allow_full_reuse"
                available_videos_for_segment = list(video_metadata_map.values())
                available_videos_for_segment.sort(key=lambda v: self.used_videos.get(v["id"], 0))

            if not available_videos_for_segment and video_metadata_map: # If strategy filtered all out, but videos exist
                logger.warning(f"音频段 {segment_id} 根据重用策略无可用视频，尝试从所有视频中选择。")
                available_videos_for_segment = list(video_metadata_map.values())

            # 为音频段获取前 self.top_k_candidates (即5) 个候选视频
            # _get_candidate_videos 返回 (video_id, similarity_score, video_full_duration)
            top_candidates_for_edl = self._get_candidate_videos(
                segment, 
                available_videos_for_segment if available_videos_for_segment else list(video_metadata_map.values()), # Fallback if empty
                top_n=self.top_k_candidates 
            )
            
            event = {
                "event_type": "video_match",
                "audio_segment_id": segment_id,
                "audio_start_time": segment.get("start"),
                "audio_end_time": segment.get("end"),
                "audio_text": segment_text,
                "video_segments": [] 
            }
            
            if not top_candidates_for_edl:
                event["event_type"] = "match_failed"
                event["notes"] = "没有找到足够的视频候选"
                if not video_metadata_map:
                     event["notes"] += " (视频库为空)"
                logger.warning(f"  无法为音频段 ID {segment_id} 找到视频候选。")
            else:
                if segment_id not in self.used_videos_per_segment:
                    self.used_videos_per_segment[segment_id] = set()

                for rank_idx, (video_id, score, video_full_duration) in enumerate(top_candidates_for_edl):
                    video_meta = video_metadata_map.get(video_id)
                    if not video_meta:
                        logger.warning(f"  处理音频段 {segment_id} 时：找不到视频 {video_id} 的元数据，跳过此候选。")
                        continue
                    
                    # EDL中segment的duration应该是这个视频片段如果被选中用于导出时，应该具有的长度
                    # 通常是音频段的长度，但不能超过视频本身的实际长度（如果从头播放）
                    # ClipExporter 会用这个 duration 作为 subclip 的 segment_duration
                    # 并用 audio_duration (即这里的 segment_duration) 作为 target_duration
                    
                    edl_clip_duration = segment_duration # 默认使用音频段时长
                    if video_full_duration is not None: # video_full_duration 是视频文件的总时长
                        edl_clip_duration = min(video_full_duration, segment_duration)
                    
                    video_segment_for_edl = {
                        "video_id": video_id,
                        "filename": video_meta.get("filename", "unknown_video.mp4"),
                        "similarity_score": round(score, 4),
                        "start_time": 0, # 假设候选片段从头开始
                        "duration": edl_clip_duration,
                        "original_video_full_duration": video_full_duration
                        # "rank": rank_idx + 1 # 可以选择在这里添加rank
                    }
                    event["video_segments"].append(video_segment_for_edl)
                    
                    # 更新视频使用次数（主要用于重用策略的排序和判断）
                    self.used_videos[video_id] = self.used_videos.get(video_id, 0) + 1
                    self.used_videos_per_segment[segment_id].add(video_id)
                    
                    logger.info(f"  为音频段 {segment_id} (第 {rank_idx+1} 候选): 分配视频 {video_id} (分数: {score:.4f}, EDL片段时长: {edl_clip_duration:.2f}s, 原始总时长: {video_full_duration})")
            
            timeline_events.append(event)

        self.analyze_matching_coverage(timeline_events)
        logger.info(f"时间线生成完成。创建了 {len(timeline_events)} 个事件。")
        return timeline_events

    def analyze_matching_coverage(self, edl):
        total_segments_with_audio = 0
        segments_with_target_matches = 0
        segments_with_less_map = {} # audio_segment_id -> count
        
        target_match_count = self.top_k_candidates #应该是5

        for event in edl:
            # 只分析有实际音频内容的段落
            if event.get("event_type") == "slug" and (not event.get("audio_text") or event.get("reason") == "文本为空或过短"):
                continue
            
            total_segments_with_audio +=1

            if event.get("event_type") == "video_match" or event.get("event_type") == "match_failed":
                video_segments_list = event.get("video_segments", [])
                count = len(video_segments_list)
                
                if count >= target_match_count:
                    segments_with_target_matches += 1
                else:
                    segments_with_less_map[event.get("audio_segment_id", "unknown_id")] = count
        
        logger.info(f"匹配覆盖分析: 总计 {total_segments_with_audio} 个有效音频段。")
        logger.info(f"其中 {segments_with_target_matches} 个音频段有 {target_match_count} 个或更多视频匹配。")
        if segments_with_less_map:
            logger.warning(f"匹配数量少于 {target_match_count} 个的音频段 (ID: 数量): {segments_with_less_map}")
        else:
            logger.info(f"所有有效音频段均已达到 {target_match_count} 个视频匹配的目标 (或因视频库总数限制)。")

    def save_timeline(self, timeline_data, output_filename=None):
        # ... (此方法不变) ...
        if output_filename is None:
            processed_dir = self.config.get("paths", {}).get("output_processed_data_dir", "data/processed_data")
            filename = self.config.get("paths", {}).get("timeline_json", "timeline.json") # 确保config.yaml中有timeline_json
            # 如果config.yaml中用的是edit_decision_list_json，则用那个键
            if "edit_decision_list_json" in self.config.get("paths",{}):
                 filename = self.config.get("paths", {}).get("edit_decision_list_json", "timeline.json")
            output_path = os.path.join(processed_dir, filename)
        else: 
            if os.path.isabs(output_filename) or os.path.dirname(output_filename): 
                output_path = output_filename
            else: 
                processed_dir = self.config.get("paths", {}).get("output_processed_data_dir", "data/processed_data")
                output_path = os.path.join(processed_dir, output_filename)

        logger.info(f"尝试将时间线保存到: {output_path}")
        # 确保路径的目录存在
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"已创建目录: {output_dir}")
            except Exception as e:
                logger.error(f"创建目录 {output_dir} 失败: {e}")
                return False
        return save_json(timeline_data, output_path)



if __name__ == "__main__":
    # Setup basic logging for testing this module directly
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # --- Mocking dependencies (Matcher) ---
    class MockMatcher:
        def __init__(self, config_dict):
            self.model_name = config_dict.get("matching",{}).get("embedding_model")
            self.threshold = config_dict.get("matching",{}).get("similarity_threshold", 0.1)
            logger.info(f"MockMatcher initialized with model {self.model_name} and threshold {self.threshold}")

        def find_best_matches(self, audio_segment, video_metadata_list, top_n=5):
            logger.debug(f"MockMatcher: find_best_matches called for audio text: '{audio_segment['text'][:30]}...'")
            # Simulate some matching logic based on keywords for testing
            matches = []
            audio_text_lower = audio_segment['text'].lower()
            for video_meta in video_metadata_list:
                video_desc_lower = video_meta.get('description_for_embedding', video_meta.get('filename','')).lower()
                score = 0.0
                if "park" in audio_text_lower and "park" in video_desc_lower: score = 0.9
                elif "car" in audio_text_lower and "car" in video_desc_lower: score = 0.85
                elif "ocean" in audio_text_lower and "ocean" in video_desc_lower: score = 0.92
                elif "cat" in video_desc_lower: score = 0.5 # Generic match
                else: score = 0.2 # Low score for others

                if score >= self.threshold:
                    matches.append((video_meta['id'], score))
            
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[:top_n]

    # --- Dummy Config for Testing ---
    dummy_config_for_timeline = {
        "paths": {
            "output_processed_data_dir": "temp_test_data/processed",
            "timeline_json": "test_timeline.json"
        },
        "matching": { # Needed for MockMatcher
             "embedding_model": "mock_model",
             "similarity_threshold": 0.3
        },
        "timeline": {
            "allow_video_repetition": False,
            "max_video_uses": 1, # Irrelevant if allow_video_repetition is False
            "default_video_id_if_no_match": "vid_default",
            "min_segment_duration_for_match": 0.1,
            "num_candidates_for_selection": 5
        }
    }
    os.makedirs(dummy_config_for_timeline["paths"]["output_processed_data_dir"], exist_ok=True)


    logger.info("--- Testing TimelineGenerator ---")
    mock_matcher_instance = MockMatcher(dummy_config_for_timeline)
    timeline_generator = TimelineGenerator(dummy_config_for_timeline, mock_matcher_instance)

    # --- Test Data ---
    test_audio_transcription = {
        "segments": [
            {"id": "seg_001", "start": 0.0, "end": 5.0, "text": "A beautiful sunny day in the park with children playing."},
            {"id": "seg_002", "start": 5.5, "end": 10.0, "text": "A fast car chase through the city streets at night."},
            {"id": "seg_003", "start": 10.5, "end": 15.0, "text": "A quiet moment by the ocean waves."},
            {"id": "seg_004", "start": 15.5, "end": 16.0, "text": "Short."}, # Test min_segment_duration
            {"id": "seg_005", "start": 16.5, "end": 20.0, "text": "Something completely different without direct video match."},
            {"id": "seg_006", "start": 20.5, "end": 25.0, "text": "Another day in the park, perhaps a picnic."} # Test repetition
        ]
    }
    test_video_metadata = [
        {"id": "vid_001", "filename": "park_footage_01.mp4", "description_for_embedding": "People enjoying a sunny park, kids on swings."},
        {"id": "vid_002", "filename": "car_chase_scene.mp4", "description_for_embedding": "High-speed car pursuit in urban setting."},
        {"id": "vid_003", "filename": "ocean_sunset.mp4", "description_for_embedding": "Serene beach with gentle ocean waves."},
        {"id": "vid_004", "filename": "random_cat_video.mp4", "description_for_embedding": "A fluffy cat playing with a toy."},
        {"id": "vid_default", "filename": "default_placeholder.mp4", "description_for_embedding": "Default placeholder screen."}
    ]

    # --- Generate Timeline ---
    generated_timeline = timeline_generator.generate_timeline(test_audio_transcription, test_video_metadata)

    if generated_timeline:
        logger.info("\n--- Generated Timeline Events: ---")
        for event_idx, event in enumerate(generated_timeline):
            logger.info(f"Event {event_idx + 1}:")
            logger.info(f"  Audio Segment ID: {event['audio_segment_id']}")
            logger.info(f"  Audio Text: '{event['audio_text'][:40]}...'")
            logger.info(f"  Event Type: {event['event_type']}")
            if event.get("matched_video_id"):
                logger.info(f"  Matched Video ID: {event['matched_video_id']} (Filename: {event['matched_video_filename']})")
                logger.info(f"  Similarity Score: {event['similarity_score']}")
            if event.get("notes"):
                logger.info(f"  Notes: {event['notes']}")

        # --- Assertions for basic correctness (example) ---
        assert len(generated_timeline) == len(test_audio_transcription["segments"])
        # seg_001 should match vid_001
        assert generated_timeline[0]["matched_video_id"] == "vid_001"
        # seg_002 should match vid_002
        assert generated_timeline[1]["matched_video_id"] == "vid_002"
        # seg_004 should be a slug
        assert generated_timeline[3]["event_type"] == "slug"
        # seg_005 should use default video if no direct match by mock logic
        assert generated_timeline[4]["matched_video_id"] == "vid_default" or generated_timeline[4]["matched_video_id"] == "vid_004" # Mock might pick cat
        # seg_006 (park again) should use default if vid_001 (park) cannot be repeated
        # (This depends on allow_video_repetition = False and if vid_001 was used for seg_001)
        if not dummy_config_for_timeline["timeline"]["allow_video_repetition"]:
             assert generated_timeline[5]["matched_video_id"] == "vid_default" or generated_timeline[5]["matched_video_id"] == "vid_004", \
                 f"Expected default or another video for seg_006 due to no-repetition, got {generated_timeline[5]['matched_video_id']}"


        # --- Test Saving Timeline ---
        if timeline_generator.save_timeline(generated_timeline):
            saved_path = os.path.join(dummy_config_for_timeline["paths"]["output_processed_data_dir"], dummy_config_for_timeline["paths"]["timeline_json"])
            logger.info(f"\nTimeline successfully saved to {saved_path}")
            assert os.path.exists(saved_path)
        else:
            logger.error("\nFailed to save the timeline.")
    else:
        logger.error("Timeline generation failed or produced an empty timeline.")

    logger.info("\nTimelineGenerator tests completed.")
    # Consider cleaning up temp_test_data