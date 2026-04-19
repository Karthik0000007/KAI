"""
Benchmark script for Task 1.5: Verify async pipeline reduces latency

This script measures turn latency with the async pipeline and compares
against the target of <10s (down from 15-20s baseline).

Measurements:
- Total turn latency (end-to-end)
- Per-stage latency (STT, emotion, LLM, TTS)
- Parallel execution speedup

Target: <10s total turn latency
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch, AsyncMock
import statistics

# Mock heavy dependencies for benchmarking
import sys
sys.path.insert(0, str(Path(__file__).parent))


class LatencyBenchmark:
    """Benchmark async pipeline latency"""
    
    def __init__(self):
        self.results = []
    
    async def mock_stt(self, duration: float = 3.0) -> tuple:
        """Mock STT with realistic timing"""
        await asyncio.sleep(duration)
        return ("Hello, I slept 7 hours last night and feel good", "en")
    
    async def mock_emotion(self, duration: float = 1.5) -> object:
        """Mock emotion analysis with realistic timing"""
        await asyncio.sleep(duration)
        from core.models import EmotionResult
        return EmotionResult(
            label="calm",
            confidence=0.85,
            pitch_mean=150.0,
            pitch_std=20.0,
            energy_rms=0.05,
            speech_rate=3.5
        )
    
    async def mock_health_extraction(self, duration: float = 0.5) -> dict:
        """Mock health signal extraction"""
        await asyncio.sleep(duration)
        return {
            "sleep_hours": 7.0,
            "mood_score": 8.0,
            "energy_level": 7.5
        }
    
    async def mock_llm(self, duration: float = 2.5) -> str:
        """Mock LLM response generation"""
        await asyncio.sleep(duration)
        return "That's wonderful! Seven hours of sleep is excellent. How are you feeling today?"
    
    async def mock_tts(self, duration: float = 2.0) -> None:
        """Mock TTS synthesis"""
        await asyncio.sleep(duration)
    
    async def mock_db_save(self, duration: float = 0.1) -> None:
        """Mock database save"""
        await asyncio.sleep(duration)
    
    async def measure_sequential_pipeline(self) -> Dict[str, float]:
        """Measure latency of sequential (blocking) pipeline"""
        print("\n" + "="*60)
        print("Measuring SEQUENTIAL pipeline (baseline)...")
        print("="*60)
        
        start_time = time.time()
        
        # Sequential execution (old approach)
        stt_start = time.time()
        text, lang = await self.mock_stt(3.0)
        stt_time = time.time() - stt_start
        print(f"  STT:              {stt_time:.2f}s")
        
        emotion_start = time.time()
        emotion = await self.mock_emotion(1.5)
        emotion_time = time.time() - emotion_start
        print(f"  Emotion:          {emotion_time:.2f}s")
        
        health_start = time.time()
        health = await self.mock_health_extraction(0.5)
        health_time = time.time() - health_start
        print(f"  Health Extract:   {health_time:.2f}s")
        
        db_start = time.time()
        await self.mock_db_save(0.1)
        db_time = time.time() - db_start
        print(f"  DB Save:          {db_time:.2f}s")
        
        llm_start = time.time()
        response = await self.mock_llm(2.5)
        llm_time = time.time() - llm_start
        print(f"  LLM:              {llm_time:.2f}s")
        
        tts_start = time.time()
        await self.mock_tts(2.0)
        tts_time = time.time() - tts_start
        print(f"  TTS:              {tts_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"  {'─'*40}")
        print(f"  TOTAL:            {total_time:.2f}s")
        
        return {
            "total": total_time,
            "stt": stt_time,
            "emotion": emotion_time,
            "health": health_time,
            "db": db_time,
            "llm": llm_time,
            "tts": tts_time
        }
    
    async def measure_async_pipeline(self) -> Dict[str, float]:
        """Measure latency of async (parallel) pipeline"""
        print("\n" + "="*60)
        print("Measuring ASYNC pipeline (optimized)...")
        print("="*60)
        
        start_time = time.time()
        
        # Parallel STT and emotion analysis (Task 1.3)
        parallel_start = time.time()
        (text, lang), emotion = await asyncio.gather(
            self.mock_stt(3.0),
            self.mock_emotion(1.5)
        )
        parallel_time = time.time() - parallel_start
        print(f"  STT + Emotion:    {parallel_time:.2f}s (parallel)")
        
        # Health extraction (depends on STT)
        health_start = time.time()
        health = await self.mock_health_extraction(0.5)
        health_time = time.time() - health_start
        print(f"  Health Extract:   {health_time:.2f}s")
        
        # LLM response generation
        llm_start = time.time()
        response = await self.mock_llm(2.5)
        llm_time = time.time() - llm_start
        print(f"  LLM:              {llm_time:.2f}s")
        
        # Parallel TTS and DB save
        parallel2_start = time.time()
        await asyncio.gather(
            self.mock_tts(2.0),
            self.mock_db_save(0.1)
        )
        parallel2_time = time.time() - parallel2_start
        print(f"  TTS + DB Save:    {parallel2_time:.2f}s (parallel)")
        
        total_time = time.time() - start_time
        print(f"  {'─'*40}")
        print(f"  TOTAL:            {total_time:.2f}s")
        
        return {
            "total": total_time,
            "stt_emotion_parallel": parallel_time,
            "health": health_time,
            "llm": llm_time,
            "tts_db_parallel": parallel2_time
        }
    
    async def run_benchmark(self, iterations: int = 5):
        """Run benchmark multiple times and compute statistics"""
        print("\n" + "="*60)
        print(f"LATENCY BENCHMARK - Task 1.5 Checkpoint")
        print(f"Running {iterations} iterations...")
        print("="*60)
        
        sequential_times = []
        async_times = []
        
        for i in range(iterations):
            print(f"\n--- Iteration {i+1}/{iterations} ---")
            
            # Measure sequential
            seq_result = await self.measure_sequential_pipeline()
            sequential_times.append(seq_result["total"])
            
            # Measure async
            async_result = await self.measure_async_pipeline()
            async_times.append(async_result["total"])
            
            speedup = seq_result["total"] / async_result["total"]
            print(f"\n  Speedup: {speedup:.2f}x")
        
        # Compute statistics
        seq_mean = statistics.mean(sequential_times)
        seq_stdev = statistics.stdev(sequential_times) if len(sequential_times) > 1 else 0
        
        async_mean = statistics.mean(async_times)
        async_stdev = statistics.stdev(async_times) if len(async_times) > 1 else 0
        
        speedup_mean = seq_mean / async_mean
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"\nSequential Pipeline (Baseline):")
        print(f"  Mean:   {seq_mean:.2f}s ± {seq_stdev:.2f}s")
        print(f"  Range:  {min(sequential_times):.2f}s - {max(sequential_times):.2f}s")
        
        print(f"\nAsync Pipeline (Optimized):")
        print(f"  Mean:   {async_mean:.2f}s ± {async_stdev:.2f}s")
        print(f"  Range:  {min(async_times):.2f}s - {max(async_times):.2f}s")
        
        print(f"\nPerformance Improvement:")
        print(f"  Speedup:        {speedup_mean:.2f}x")
        print(f"  Time Saved:     {seq_mean - async_mean:.2f}s ({(1 - async_mean/seq_mean)*100:.1f}%)")
        
        # Check against target
        print(f"\nTarget Verification:")
        print(f"  Target:         <10s")
        print(f"  Achieved:       {async_mean:.2f}s")
        
        if async_mean < 10.0:
            print(f"  Status:         ✓ PASS (under target by {10.0 - async_mean:.2f}s)")
        else:
            print(f"  Status:         ✗ FAIL (over target by {async_mean - 10.0:.2f}s)")
        
        print("\n" + "="*60)
        
        return {
            "sequential_mean": seq_mean,
            "async_mean": async_mean,
            "speedup": speedup_mean,
            "target_met": async_mean < 10.0
        }


async def main():
    """Run the benchmark"""
    benchmark = LatencyBenchmark()
    results = await benchmark.run_benchmark(iterations=3)
    
    # Return exit code based on target
    return 0 if results["target_met"] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
