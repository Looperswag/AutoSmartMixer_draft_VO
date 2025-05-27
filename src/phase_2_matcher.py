# src/phase_2_matcher.py

import logging
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
except ImportError:
    print("SentenceTransformers library not found. Please install it: pip install sentence-transformers")
    SentenceTransformer = None
    cos_sim = None

logger = logging.getLogger(__name__)

class Matcher:
    """
    Matches audio segments with video clips based on semantic similarity
    of their textual content or descriptions.
    """
    def __init__(self, config):
        """
        Initializes the Matcher.

        Args:
            config (dict): The application configuration dictionary.
        """
        self.config = config
        self.st_model = None
        # Harmonize config key for embedding model name
        self.embedding_model_name = self.config.get("models", {}).get(
            "embedding_model_name", "all-MiniLM-L6-v2" # Default if not found
        )

        if SentenceTransformer:
            try:
                logger.info(f"Matcher: Loading sentence transformer model: {self.embedding_model_name}...")
                self.st_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Matcher: Sentence transformer model '{self.embedding_model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Matcher: Failed to load sentence transformer model '{self.embedding_model_name}': {e}")
                self.st_model = None
        else:
            logger.error("Matcher: SentenceTransformers library is not available. Matching will not work.")

        self.similarity_threshold = self.config.get("settings", {}).get( # Corrected from "matching" to "settings" as per config.yaml
            "similarity_threshold", 0.5
        )

    def get_embedding_for_text(self, text):
        """
        Generates an embedding for a given text string.

        Args:
            text (str): The text to embed.

        Returns:
            numpy.ndarray: The embedding vector (as float32), or None if model not loaded or text is empty.
        """
        if not self.st_model:
            logger.error("Matcher: Sentence transformer model not loaded. Cannot generate embeddings.")
            return None
        if not text or not text.strip():
            logger.warning("Matcher: Attempted to get embedding for empty text.")
            return None
        try:
            embedding = self.st_model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32) # Ensure float32
        except Exception as e:
            logger.error(f"Matcher: Error generating embedding for text '{text[:50]}...': {e}")
            return None

    def find_best_matches(self, audio_segment, video_metadata_list, top_n=5):
        """
        Finds the best matching video clips for a given audio segment.

        Args:
            audio_segment (dict): A dictionary representing an audio segment,
                                  must contain 'text' (or 'embedding' if pre-computed).
                                  Example: {'id': 0, 'start': 0.0, 'end': 5.0, 'text': 'some transcribed text'}
            video_metadata_list (list): A list of video metadata dictionaries,
                                        where each must contain 'embedding' and 'id'.
                                        Example: [{'id': 'video_0000', ..., 'embedding': [0.1, ...]}, ...]
            top_n (int): The number of top matches to return.

        Returns:
            list: A list of tuples, where each tuple is (video_id, similarity_score),
                  sorted by similarity_score in descending order.
                  Returns an empty list if no suitable matches are found or an error occurs.
        """
        if not self.st_model:
            logger.error("Matcher: Sentence transformer model not loaded. Cannot find matches.")
            return []
        if not video_metadata_list:
            logger.warning("Matcher: Video metadata list is empty. No videos to match against.")
            return []
        if not audio_segment or ('text' not in audio_segment and 'embedding' not in audio_segment) :
            logger.error("Matcher: Audio segment is invalid or missing text/embedding.")
            return []

        # Get or generate audio segment embedding
        audio_embedding_val = audio_segment.get("embedding")
        if audio_embedding_val is None:
            if not audio_segment.get("text"):
                logger.error("Matcher: Audio segment has no text to generate embedding from.")
                return []
            audio_embedding_val = self.get_embedding_for_text(audio_segment["text"]) # Returns float32
            if audio_embedding_val is None:
                logger.error(f"Matcher: Failed to generate embedding for audio segment ID {audio_segment.get('id')}.")
                return []
        else: # If embedding is pre-computed, ensure it's a numpy array of float32
             if isinstance(audio_embedding_val, list):
                audio_embedding_val = np.array(audio_embedding_val, dtype=np.float32)
             elif isinstance(audio_embedding_val, np.ndarray):
                audio_embedding_val = audio_embedding_val.astype(np.float32)


        # Prepare video embeddings
        video_embeddings = []
        video_ids = []
        video_durations = []


        for video_meta in video_metadata_list:
            if "embedding" in video_meta and "id" in video_meta:
                video_emb_val = video_meta["embedding"]
                if isinstance(video_emb_val, list):
                    video_emb_val = np.array(video_emb_val, dtype=np.float32)
                elif isinstance(video_emb_val, np.ndarray):
                    video_emb_val = video_emb_val.astype(np.float32)
                
                if video_emb_val is not None and video_emb_val.ndim == 1:
                    video_embeddings.append(video_emb_val)
                    video_ids.append(video_meta["id"])
                    
                    # 获取视频时长 - 从视频文件中读取或使用预存储的值
                    video_duration = video_meta.get("duration", None)
                    if video_duration is None and os.path.exists(video_meta.get("path", "")):
                        try:
                            from moviepy.editor import VideoFileClip
                            with VideoFileClip(video_meta["path"]) as clip:
                                video_duration = clip.duration
                            # 可选：将时长缓存回元数据
                            video_meta["duration"] = video_duration
                        except Exception as e:
                            logger.warning(f"无法获取视频 {video_meta['id']} 的时长: {e}")
                            video_duration = None
                    
                    video_durations.append(video_duration)
                elif video_emb_val is not None and video_emb_val.ndim == 2 and video_emb_val.shape[0] == 1:
                    video_embeddings.append(video_emb_val.reshape(-1).astype(np.float32))
                    video_ids.append(video_meta["id"])
                    
                    # 同样获取时长
                    video_duration = video_meta.get("duration", None)
                    if video_duration is None and os.path.exists(video_meta.get("path", "")):
                        try:
                            from moviepy.editor import VideoFileClip
                            with VideoFileClip(video_meta["path"]) as clip:
                                video_duration = clip.duration
                            video_meta["duration"] = video_duration
                        except Exception as e:
                            logger.warning(f"无法获取视频 {video_meta['id']} 的时长: {e}")
                            video_duration = None
                    
                    video_durations.append(video_duration)
                else:
                    logger.warning(f"Matcher: Video ID {video_meta['id']} has invalid or missing embedding format. Skipping.")
            else:
                logger.warning(f"Matcher: Video metadata item is missing 'embedding' or 'id'. Skipping: {video_meta.get('filename', 'Unknown Filename')}")

        if not video_embeddings:
            logger.warning("Matcher: No valid video embeddings found to match against.")
            return []

        # Calculate cosine similarities
        # Reshape audio_embedding_val to 2D array for cos_sim if it's 1D
        if audio_embedding_val.ndim == 1:
            audio_embedding_2d = audio_embedding_val.reshape(1, -1)
        else:
            audio_embedding_2d = audio_embedding_val
        
        # Ensure audio_embedding_2d is float32
        audio_embedding_2d = audio_embedding_2d.astype(np.float32)

        try:
            # Ensure video_embeddings is a 2D numpy array of float32
            video_embeddings_matrix = np.array(video_embeddings, dtype=np.float32)
            if video_embeddings_matrix.ndim == 1: 
                logger.error("Matcher: Video embeddings matrix is unexpectedly 1D.")
                return []

            similarities = cos_sim(audio_embedding_2d, video_embeddings_matrix)
            # cos_sim returns a tensor of shape (1, num_videos); convert to list of scores
            similarity_scores = similarities[0].cpu().tolist() 
        except Exception as e:
            logger.error(f"Matcher: Error calculating cosine similarities: {e}", exc_info=True)
            return []

        # Combine video IDs with their scores
        scored_videos = []
        for i, (video_id, score) in enumerate(zip(video_ids, similarity_scores)):
            scored_videos.append((video_id, score, video_durations[i]))

        # Sort by similarity score in descending order
        scored_videos.sort(key=lambda x: x[1], reverse=True)
        top_matches = scored_videos[:top_n]

        logger.debug(f"Matcher: Found top {len(top_matches)} matches for audio segment ID {audio_segment.get('id', 'N/A')} (text: '{audio_segment.get('text', '')[:30]}...').")
        return top_matches


if __name__ == "__main__":
    # Setup basic logging for testing this module directly
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # --- Dummy Config for Testing ---
    dummy_config_for_matcher = {
        "models": { # Changed from "matching" to "models"
            "embedding_model_name": "all-MiniLM-L6-v2", # Ensure this model is available
        },
        "settings": { # Changed from "matching" to "settings"
             "similarity_threshold": 0.3 # Lower threshold for testing
        }
    }

    logger.info("--- Testing Matcher ---")

    if not SentenceTransformer or not cos_sim:
        logger.error("SentenceTransformers library not available. Cannot run Matcher tests.")
    else:
        matcher = Matcher(dummy_config_for_matcher)

        if not matcher.st_model:
            logger.error("Matcher model failed to load. Aborting tests.")
        else:
            # --- Test Data ---
            # Mock audio segments (typically from Whisper)
            audio_segments_test = [
                {"id": "seg_001", "start": 0.0, "end": 5.0, "text": "A beautiful sunny day in the park with children playing."},
                {"id": "seg_002", "start": 5.5, "end": 10.0, "text": "A fast car chase through the city streets at night."},
                {"id": "seg_003", "start": 10.5, "end": 15.0, "text": "A quiet moment by the ocean waves."},
                {"id": "seg_004", "start": 15.5, "end": 20.0, "text": ""}, # Empty text
                {"id": "seg_005", "start": 20.5, "end": 25.0} # Missing text
            ]

            # Mock video metadata (typically from VideoAnalyzer)
            video_texts_for_embeddings = [
                "People enjoying a sunny park, kids on swings.", # Matches seg_001
                "High-speed pursuit with cars in urban setting.", # Matches seg_002
                "Serene beach with gentle waves.", # Matches seg_003
                "Random unrelated footage of a cat.",
                "Cooking show in a kitchen."
            ]
            video_metadata_test = []
            for i, v_text in enumerate(video_texts_for_embeddings):
                emb = matcher.get_embedding_for_text(v_text) # emb is now float32
                if emb is not None:
                    video_metadata_test.append({
                        "id": f"vid_{i:03d}",
                        "filename": f"video_{i}.mp4",
                        "description_for_embedding": v_text,
                        "embedding": emb.tolist() # Store as list, matcher will convert back
                    })
                else:
                    logger.error(f"Could not generate embedding for test video text: {v_text}")

            if not video_metadata_test:
                logger.error("Failed to create test video metadata with embeddings. Aborting further tests.")

            else:
                logger.info(f"\n--- Test Case 1: Matching first audio segment ---")
                matches1 = matcher.find_best_matches(audio_segments_test[0], video_metadata_test, top_n=3)
                logger.info(f"Matches for '{audio_segments_test[0]['text'][:30]}...': {matches1}")
                assert len(matches1) > 0 and matches1[0][0] == "vid_000", f"Expected vid_000 as best match for seg_001, got {matches1}"

                logger.info(f"\n--- Test Case 2: Matching second audio segment ---")
                matches2 = matcher.find_best_matches(audio_segments_test[1], video_metadata_test, top_n=3)
                logger.info(f"Matches for '{audio_segments_test[1]['text'][:30]}...': {matches2}")
                assert len(matches2) > 0 and matches2[0][0] == "vid_001", f"Expected vid_001 as best match for seg_002, got {matches2}"

                logger.info(f"\n--- Test Case 3: Matching with a lower similarity (should get more results if available) ---")
                original_threshold = matcher.similarity_threshold
                matcher.similarity_threshold = 0.1
                matches3 = matcher.find_best_matches(audio_segments_test[2], video_metadata_test, top_n=5)
                logger.info(f"Matches for '{audio_segments_test[2]['text'][:30]}...' (low threshold): {matches3}")
                assert len(matches3) > 0 and matches3[0][0] == "vid_002", f"Expected vid_002 as best match for seg_003, got {matches3}"
                matcher.similarity_threshold = original_threshold # Reset threshold

                logger.info(f"\n--- Test Case 4: Audio segment with empty text ---")
                matches4 = matcher.find_best_matches(audio_segments_test[3], video_metadata_test, top_n=3)
                logger.info(f"Matches for audio segment with empty text: {matches4}")
                assert len(matches4) == 0, "Expected no matches for empty text audio segment."

                logger.info(f"\n--- Test Case 5: Audio segment missing text field ---")
                matches5 = matcher.find_best_matches(audio_segments_test[4], video_metadata_test, top_n=3)
                logger.info(f"Matches for audio segment with missing text: {matches5}")
                assert len(matches5) == 0, "Expected no matches for audio segment missing text."

                logger.info(f"\n--- Test Case 6: No video metadata ---")
                matches6 = matcher.find_best_matches(audio_segments_test[0], [], top_n=3)
                logger.info(f"Matches with empty video metadata list: {matches6}")
                assert len(matches6) == 0, "Expected no matches when video metadata is empty."

                logger.info("\nMatcher tests completed.")