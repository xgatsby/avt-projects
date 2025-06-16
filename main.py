
#!/usr/bin/env python3
"""
AVT Subtitler Pro - Professional AI-Powered Subtitle Generation System
==================================================================

A comprehensive subtitle generation pipeline that combines:
- Advanced Whisper ASR for transcription
- Context-aware translation engine
- Netflix-quality subtitle optimization
- Multi-layer quality assurance
- Professional output formatting

Author: AVT Subtitler Pro Team
Version: 1.0.0
Date: January 2025
"""

import os
import sys
import json
import logging
import warnings
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# Core ML Libraries
import torch
import numpy as np
from transformers import (
    pipeline,
)

import whisperx

# Subtitle Processing
import pysrt
from pysrt import SubRipFile, SubRipItem, SubRipTime

# Configuration & Utilities
import yaml
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import language_tool_python

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./avt_subtitler.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SubtitleSegment:
    """Represents a subtitle segment with timing and content information."""
    start_time: float
    end_time: float
    text: str
    confidence: float = 0.0
    speaker_id: Optional[str] = None
    language: str = "en"
    
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_srt_time(self, time_seconds: float) -> SubRipTime:
        """Convert seconds to SubRipTime format."""
        hours = int(time_seconds // 3600)
        minutes = int((time_seconds % 3600) // 60)
        seconds = int(time_seconds % 60)
        milliseconds = int((time_seconds % 1) * 1000)
        return SubRipTime(hours, minutes, seconds, milliseconds)

@dataclass
class Config:
    """Configuration for the AVT Subtitler Pro system."""
    asr_model: str = "tiny"
    alignment_model: Optional[str] = None
    diarization_enabled: bool = False
    translation_enabled: bool = False
    translation_model: str = "Helsinki-NLP/opus-mt-en-id"
    target_language: str = "en"
    batch_size: int = 8
    output_dir: str = "./output"
    audio_file: Optional[str] = None

class SubtitleStandards:
    """Netflix and international subtitle standards compliance."""
    
    # Netflix Standards
    MIN_DURATION = 5/6  # 5/6 seconds minimum
    MAX_DURATION = 7.0  # 7 seconds maximum
    MAX_CHARS_PER_LINE = 42  # Netflix standard
    MAX_LINES = 2
    MIN_GAP = 0.083  # 2 frames at 24fps
    
    # Reading speeds (characters per second)
    READING_SPEED_SLOW = 12
    READING_SPEED_MEDIUM = 17
    READING_SPEED_FAST = 21
    
    @classmethod
    def validate_segment(cls, segment: SubtitleSegment) -> Dict[str, Any]:
        """Validate segment against Netflix standards."""
        issues = []
        
        # Duration checks
        if segment.duration() < cls.MIN_DURATION:
            issues.append(f"Duration too short: {segment.duration():.2f}s < {cls.MIN_DURATION}s")
        if segment.duration() > cls.MAX_DURATION:
            issues.append(f"Duration too long: {segment.duration():.2f}s > {cls.MAX_DURATION}s")
        
        # Character count checks
        lines = segment.text.split("\n")
        if len(lines) > cls.MAX_LINES:
            issues.append(f"Too many lines: {len(lines)} > {cls.MAX_LINES}")
        
        for line in lines:
            if len(line) > cls.MAX_CHARS_PER_LINE:
                issues.append(f"Line too long: {len(line)} > {cls.MAX_CHARS_PER_LINE}")
        
        # Reading speed check
        chars_count = len(segment.text.replace("\n", " "))
        if segment.duration() > 0: # Avoid division by zero
            reading_speed = chars_count / segment.duration()
            if reading_speed > cls.READING_SPEED_FAST:
                issues.append(f"Reading speed too fast: {reading_speed:.1f} cps > {cls.READING_SPEED_FAST}")
        else:
            issues.append("Segment duration is zero, cannot calculate reading speed.")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'reading_speed': reading_speed if segment.duration() > 0 else 0.0,
            'character_count': chars_count
        }

class ContextAwareTranslationEngine:
    """Advanced translation engine with context awareness."""
    
    def __init__(self, device: str = "cpu", model_name: str = "Helsinki-NLP/opus-mt-en-id"):
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    def initialize(self):
        """Initialize translation models."""
        try:
            logger.info(f"Loading translation model: {self.model_name}")
            
            # Try to load with pipeline first (more robust)
            self.pipeline = pipeline(
                "translation",
                model=self.model_name,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            logger.info("Translation engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize translation engine: {e}")
            # Fallback to basic model
            try:
                self.model_name = "Helsinki-NLP/opus-mt-en-mul"
                self.pipeline = pipeline("translation", model=self.model_name, device=-1)
                logger.info("Fallback translation model loaded")
                return True
            except Exception as e2:
                logger.error(f"Fallback translation model also failed: {e2}")
                return False
    
    def translate_segments(self, segments: List[SubtitleSegment], target_language: str = "id", batch_size: int = 8) -> List[SubtitleSegment]:
        """Translate subtitle segments with context awareness and batch processing."""
        if not self.pipeline:
            logger.error("Translation pipeline not initialized")
            return segments
        
        translated_segments = []
        texts_to_translate = [segment.text.strip() for segment in segments]
        
        # Filter out empty texts to avoid errors in translation pipeline
        non_empty_texts = [text for text in texts_to_translate if text]
        original_indices = [i for i, text in enumerate(texts_to_translate) if text]

        if not non_empty_texts:
            return segments # No text to translate

        try:
            # Perform translation in batches
            translated_results = self.pipeline(non_empty_texts, batch_size=batch_size)
            
            translated_text_map = {}
            for i, result in zip(original_indices, translated_results):
                translated_text_map[i] = result[0]["translation_text"] if isinstance(result, list) else result["translation_text"]

            # Reconstruct segments with translated text
            for i, segment in enumerate(segments):
                if i in translated_text_map:
                    new_segment = SubtitleSegment(
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        text=translated_text_map[i],
                        confidence=segment.confidence,
                        speaker_id=segment.speaker_id,
                        language=target_language
                    )
                    translated_segments.append(new_segment)
                else:
                    # If original text was empty or translation failed, keep original segment
                    translated_segments.append(segment)

        except Exception as e:
            logger.warning(f"Batch translation failed: {e}. Falling back to segment-by-segment translation.")
            # Fallback to segment-by-segment translation if batch fails
            for segment in segments:
                try:
                    input_text = segment.text.strip()
                    if not input_text:
                        translated_segments.append(segment)
                        continue
                    
                    result = self.pipeline(input_text)
                    translated_text = result[0]["translation_text"] if isinstance(result, list) else result["translation_text"]
                    
                    new_segment = SubtitleSegment(
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        text=translated_text,
                        confidence=segment.confidence,
                        speaker_id=segment.speaker_id,
                        language=target_language
                    )
                    translated_segments.append(new_segment)
                    
                except Exception as e_single:
                    logger.warning(f"Single segment translation failed: {e_single}")
                    translated_segments.append(segment)
        
        logger.info(f"Translated {len(translated_segments)} segments")
        return translated_segments

class AdvancedSubtitleOptimizer:
    """Advanced subtitle optimization with line breaking and timing adjustments."""
    
    def __init__(self):
        self.standards = SubtitleStandards()

    def optimize_line_breaks(self, text: str) -> str:
        """Optimize line breaks for better readability, considering punctuation and character limits."""
        if len(text) <= self.standards.MAX_CHARS_PER_LINE:
            return text

        # Try to split by common punctuation
        sentences = re.split(r'([.?!])\s*', text)
        if len(sentences) > 1:
            processed_sentences = []
            for i in range(0, len(sentences), 2):
                sentence = sentences[i]
                if i + 1 < len(sentences):
                    sentence += sentences[i+1]
                processed_sentences.append(sentence.strip())
            
            # Try to combine sentences into two lines, respecting max chars per line
            line1 = ""
            line2 = ""
            for s in processed_sentences:
                if len(line1) == 0 or len(line1) + len(s) + 1 <= self.standards.MAX_CHARS_PER_LINE:
                    line1 += (" " if line1 else "") + s
                elif len(line2) == 0 or len(line2) + len(s) + 1 <= self.standards.MAX_CHARS_PER_LINE:
                    line2 += (" " if line2 else "") + s
                else:
                    # If a sentence is too long for either line, fall back to word splitting
                    break
            else:
                if line1 and line2:
                    return f"{line1}\n{line2}"
                elif line1:
                    return line1

        # Fallback to word-based splitting if punctuation-based splitting doesn't work or is insufficient
        words = text.split()
        current_line = []
        lines = []
        for word in words:
            if len(' '.join(current_line + [word])) <= self.standards.MAX_CHARS_PER_LINE:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        lines.append(' '.join(current_line))

        if len(lines) > self.standards.MAX_LINES:
            # If still too many lines, try to combine or truncate (as a last resort)
            # For now, we'll just take the first two lines, or combine if possible
            if len(lines) == 3 and len(lines[0] + lines[1]) <= self.standards.MAX_CHARS_PER_LINE:
                return f"{lines[0]} {lines[1]}\n{lines[2]}"
            return "\n".join(lines[:self.standards.MAX_LINES])
        
        return "\n".join(lines)

    def adjust_timing(self, segment: SubtitleSegment) -> SubtitleSegment:
        """Adjust timing to meet standards."""
        # Calculate optimal duration based on reading speed
        chars_count = len(segment.text.replace("\n", " "))
        optimal_duration = max(
            chars_count / self.standards.READING_SPEED_MEDIUM,
            self.standards.MIN_DURATION
        )
        optimal_duration = min(optimal_duration, self.standards.MAX_DURATION)
        
        # Adjust end time if needed
        new_end_time = segment.start_time + optimal_duration
        
        return SubtitleSegment(
            start_time=segment.start_time,
            end_time=new_end_time,
            text=segment.text,
            confidence=segment.confidence,
            speaker_id=segment.speaker_id,
            language=segment.language
        )
    
    def process_segments(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Process segments with advanced optimization."""
        optimized_segments = []
        
        for segment in segments:
            # Optimize line breaks
            optimized_text = self.optimize_line_breaks(segment.text)
            
            # Create segment with optimized text
            optimized_segment = SubtitleSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=optimized_text,
                confidence=segment.confidence,
                speaker_id=segment.speaker_id,
                language=segment.language
            )
            
            # Adjust timing
            final_segment = self.adjust_timing(optimized_segment)
            optimized_segments.append(final_segment)
        
        # Ensure no overlaps
        optimized_segments = self._resolve_overlaps(optimized_segments)
        
        logger.info(f"Optimized {len(optimized_segments)} segments")
        return optimized_segments
    
    def _resolve_overlaps(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Resolve timing overlaps between segments."""
        if len(segments) <= 1:
            return segments
        
        resolved_segments = [segments[0]]
        
        for i in range(1, len(segments)):
            current = segments[i]
            previous = resolved_segments[-1]
            
            # Check for overlap
            if current.start_time < previous.end_time:
                # Adjust previous segment end time
                gap = self.standards.MIN_GAP
                new_end_time = max(
                    current.start_time - gap,
                    previous.start_time + self.standards.MIN_DURATION
                )
                
                resolved_segments[-1] = SubtitleSegment(
                    start_time=previous.start_time,
                    end_time=new_end_time,
                    text=previous.text,
                    confidence=previous.confidence,
                    speaker_id=previous.speaker_id,
                    language=previous.language
                )
            
            resolved_segments.append(current)
        
        return resolved_segments

class MultiLayerValidationPipeline:
    """Multi-layer quality assurance and validation pipeline."""
    
    def __init__(self):
        self.standards = SubtitleStandards()
        self.validation_report = {
            'total_segments': 0,
            'valid_segments': 0,
            'auto_fixed': 0,
            'issues': [],
            'quality_score': 0.0
        }
        self.lang_tool = None # Initialized lazily

    def initialize_language_tool(self, lang: str = "en"):
        """Initialize LanguageTool for grammar and spell checking."""
        if lang != "en":
            logger.warning(f"LanguageTool does not support language: {lang}. Skipping grammar and spell checking.")
            self.lang_tool = None
            return
        try:
            self.lang_tool = language_tool_python.LanguageTool(lang)
            logger.info(f"LanguageTool initialized for language: {lang}")
        except Exception as e:
            logger.error(f"Failed to initialize LanguageTool: {e}")
            self.lang_tool = None

    def validate_and_fix(self, segments: List[SubtitleSegment], target_language: str = "en") -> List[SubtitleSegment]:
        """Validate and auto-fix segments based on standards and grammar checks."""
        self.validation_report['total_segments'] = len(segments)
        fixed_segments = []

        if self.lang_tool is None:
            self.initialize_language_tool(target_language)

        for i, segment in enumerate(segments):
            # Validate against Netflix standards
            validation_result = self.standards.validate_segment(segment)
            if not validation_result['valid']:
                self.validation_report['issues'].append({
                    'segment_index': i,
                    'time': f"{segment.start_time:.2f}-{segment.end_time:.2f}",
                    'text': segment.text,
                    'type': 'Standard Violation',
                    'details': validation_result['issues']
                })

            # Grammar and spell check
            if self.lang_tool:
                try:
                    matches = self.lang_tool.check(segment.text)
                    if matches:
                        original_text = segment.text
                        for match in matches:
                            # Simple auto-fix: apply first suggestion if available
                            if match.replacements:
                                segment.text = language_tool_python.utils.correct(segment.text, [match])
                                self.validation_report['auto_fixed'] += 1
                                logger.debug(f"Auto-fixed: '{original_text}' to '{segment.text}'")
                        if original_text != segment.text:
                            self.validation_report['issues'].append({
                                'segment_index': i,
                                'time': f"{segment.start_time:.2f}-{segment.end_time:.2f}",
                                'original_text': original_text,
                                'fixed_text': segment.text,
                                'type': 'Grammar/Spell Check',
                                'details': [f"Corrected: {m.message}" for m in matches]
                            })
                except Exception as e:
                    logger.warning(f"LanguageTool check failed for segment {i}: {e}")
            
            fixed_segments.append(segment)
            if validation_result['valid'] and (not self.lang_tool or not matches):
                self.validation_report['valid_segments'] += 1

        # Calculate quality score (simple example)
        if self.validation_report['total_segments'] > 0:
            self.validation_report['quality_score'] = (
                self.validation_report['valid_segments'] / self.validation_report['total_segments']
            ) * 100
        
        logger.info(f"Validation complete. Total segments: {self.validation_report['total_segments']}, Valid: {self.validation_report['valid_segments']}, Auto-fixed: {self.validation_report['auto_fixed']}")
        return fixed_segments

class AVTSubtitlerPro:
    """Main class for the AVT Subtitler Pro system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.asr_model = None
        self.alignment_model = None
        self.diarization_pipeline = None
        self.translation_engine = None
        self.optimizer = AdvancedSubtitleOptimizer()
        self.validator = MultiLayerValidationPipeline()
        
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ASR, alignment, diarization, and translation models."""
        try:
            logger.info(f"Loading ASR model: {self.config.asr_model}")
            self.asr_model = whisperx.load_model(self.config.asr_model, self.device, compute_type="int8")
            logger.info("ASR model loaded successfully")

            if self.config.alignment_model:
                logger.info(f"Loading alignment model: {self.config.alignment_model}")
                self.alignment_model = whisperx.load_align_model(model_name=self.config.alignment_model, device=self.device)
                logger.info("Alignment model loaded successfully")
            else:
                logger.info("No alignment model specified, skipping alignment.")

            if self.config.diarization_enabled:
                logger.info("Loading diarization pipeline")
                # Diarization model loading can be resource intensive
                # self.diarization_pipeline = DiarizationPipeline(device=self.device)
                logger.warning("Diarization is enabled but the pipeline is not implemented. Skipping diarization.")

            if self.config.translation_enabled:
                self.translation_engine = ContextAwareTranslationEngine(device=self.device, model_name=self.config.translation_model)
                if not self.translation_engine.initialize():
                    logger.error("Failed to initialize translation engine. Translation will be skipped.")
                    self.config.translation_enabled = False

        except Exception as e:
            logger.critical(f"Failed to load one or more models: {e}")
            raise

    def transcribe_audio(self, audio_path: str) -> List[SubtitleSegment]:
        """Transcribe audio using WhisperX and return subtitle segments.""" 
        if not self.asr_model:
            logger.error("ASR model not loaded.")
            return []

        try:
            logger.info(f"Loading audio: {audio_path}")
            audio = whisperx.load_audio(audio_path)
            
            logger.info("Transcribing audio...")
            result = self.asr_model.transcribe(audio, batch_size=self.config.batch_size)
            logger.info("Transcription complete.")
            
            segments = []
            for segment_data in result["segments"]:
                segments.append(SubtitleSegment(
                    start_time=segment_data["start"],
                    end_time=segment_data["end"],
                    text=segment_data["text"],
                    confidence=segment_data.get("confidence", 0.0)
                ))
            logger.info(f"Initial transcription yielded {len(segments)} segments.")

            if self.alignment_model and segments:
                logger.info("Aligning transcription...")
                # Ensure the audio is passed correctly for alignment
                aligned_result = whisperx.align(segments, self.asr_model.model.tokenizer, audio, self.device, model=self.alignment_model)
                logger.info("Alignment complete.")
                
                aligned_segments = []
                for segment_data in aligned_result["segments"]:
                    aligned_segments.append(SubtitleSegment(
                        start_time=segment_data["start"],
                        end_time=segment_data["end"],
                        text=segment_data["text"],
                        confidence=segment_data.get("confidence", 0.0)
                    ))
                logger.info(f"Alignment resulted in {len(aligned_segments)} segments.")
                return aligned_segments
            
            return segments

        except Exception as e:
            logger.error(f"Error during transcription or alignment: {e}")
            traceback.print_exc()
            return []

    def generate_srt(self, segments: List[SubtitleSegment], output_filename: str) -> str:
        """Generate an SRT file from subtitle segments."""
        srt_file = SubRipFile()
        for i, segment in enumerate(segments):
            start_time = segment.to_srt_time(segment.start_time)
            end_time = segment.to_srt_time(segment.end_time)
            srt_item = SubRipItem(index=i+1, start=start_time, end=end_time, text=segment.text)
            srt_file.append(srt_item)
        
        output_path = Path(self.config.output_dir) / output_filename
        try:
            srt_file.save(str(output_path), encoding="utf-8")
            logger.info(f"SRT file saved to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save SRT file: {e}")
            return ""

def load_config(config_path: str) -> Config:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return Config(**config_data)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)

def main():
    logger.info("Starting AVT Subtitler Pro")
    
    # Ensure output directory exists
    Path("./output").mkdir(parents=True, exist_ok=True)

    config_path = "/home/ubuntu/upload/config_1.yaml"
    config = load_config(config_path)

    if config.audio_file:
        logger.info(f"Audio file path from config: {config.audio_file}")
        subtitler = AVTSubtitlerPro(config)
        segments = subtitler.transcribe_audio(config.audio_file)

        if segments:
            logger.info("Applying subtitle optimization...")
            optimized_segments = subtitler.optimizer.process_segments(segments)
            logger.info("Optimization complete.")

            if config.translation_enabled and subtitler.translation_engine:
                logger.info("Translating segments...")
                translated_segments = subtitler.translation_engine.translate_segments(
                    optimized_segments, 
                    target_language=config.target_language,
                    batch_size=config.batch_size
                )
                logger.info("Translation complete.")
                final_segments = translated_segments
            else:
                final_segments = optimized_segments

            logger.info("Validating and auto-fixing segments...")
            validated_segments = subtitler.validator.validate_and_fix(final_segments, target_language=config.target_language)
            logger.info("Validation complete.")
            logger.info(f"Validation Report: {subtitler.validator.validation_report}")

            # Determine output filename
            audio_filename = Path(config.audio_file).stem
            output_srt_filename = f"{audio_filename}_final.srt"
            
            srt_output_path = subtitler.generate_srt(validated_segments, output_srt_filename)
            if srt_output_path:
                logger.info(f"Processing complete. Final SRT saved to: {srt_output_path}")
            else:
                logger.error("SRT file generation failed.")
        else:
            logger.warning("No segments transcribed. Skipping further processing.")
    else:
        logger.error("No audio file specified in the configuration. Please update config.yaml.")

if __name__ == "__main__":
    main()


